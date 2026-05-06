import ants
import numpy as np
import matplotlib.pyplot as plt
import os


def auto_crop_head(ct_image, head_height_mm=250):
    """自动屏蔽手臂，裁剪真实头部物理空间"""
    print("    -> [裁剪模块] 正在探测真实头骨顶部...")
    arr = ct_image.numpy()
    spacing = ct_image.spacing

    body_mask = arr > -500
    pixel_area_mm2 = spacing[0] * spacing[1]
    area_per_slice = np.sum(body_mask, axis=(0, 1)) * pixel_area_mm2

    valid_z = np.where(area_per_slice > 1000)[0]
    if len(valid_z) == 0: return ct_image

    real_top_z = valid_z[-1]
    for z in range(valid_z[-1], valid_z[0], -1):
        if area_per_slice[z] > 12000:
            real_top_z = z
            break

    slices_needed = int(head_height_mm / spacing[2])
    z_start = max(valid_z[0], real_top_z - slices_needed)
    z_end = min(real_top_z + int(20 / spacing[2]), arr.shape[2] - 1)

    return ants.crop_indices(ct_image, [0, 0, z_start], [arr.shape[0] - 1, arr.shape[1] - 1, z_end])


def extract_cranial_cavity(ct_head):
    """
    【新增核心先验模块】：自动提取 CT 的颅腔内部几何先验（脑外轮廓）
    """
    print("    -> [先验模块] 正在提取 CT 颅腔/脑实质几何先验...")
    # 1. 抓取脑组织软组织的初步范围 [0, 80] HU
    brain_window = ants.threshold_image(ct_head, low_thresh=0, high_thresh=80)

    # 2. 形态学开运算：切断脑部与眼球、面部肌肉的微小粘连
    brain_opened = ants.morphology(brain_window, operation='open', radius=2)

    # 3. 提取最大连通域：头颅中最大的一坨 [0-80] 组织，必然是大脑实质！彻底抛弃下巴和脖子。
    brain_blob = ants.get_mask(brain_opened, cleanup=2)

    # 4. 形态学闭运算：填平脑室和沟回的坑洼，使其变成一个光滑的“实心脑壳模型”
    brain_solid = ants.morphology(brain_blob, operation='close', radius=5)

    # 稍微向外膨胀一点点，完美贴合头骨内壁
    cranial_prior = ants.morphology(brain_solid, operation='dilate', radius=2)
    return cranial_prior


def visualize_registration(fixed, moving, title, save_path):
    """可视化工具"""
    moving_resampled = ants.resample_image_to_target(image=moving, target=fixed, interp_type='linear')
    z_mid = fixed.shape[2] // 2
    plt.figure(figsize=(8, 8))
    plt.imshow(fixed.numpy()[:, :, z_mid], cmap='gray', origin='lower')
    plt.imshow(moving_resampled.numpy()[:, :, z_mid], cmap='hot', alpha=0.4, origin='lower')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[*] 快照已保存: {os.path.basename(save_path)}")


def run_registration_pipeline(ct_path, template_path, template_label_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 阶段 0: 加载数据与物理裁剪 ===")
    original_ct = ants.image_read(ct_path)
    moving_template = ants.image_read(template_path)
    moving_labels = ants.image_read(template_label_path)

    fixed_ct_head = auto_crop_head(original_ct, head_height_mm=250)
    ants.image_write(fixed_ct_head, os.path.join(output_dir, "0_CT_Cropped_Head.nii.gz"))

    print("\n=== 阶段 1: 几何先验强制对齐 (Mask-to-Mask Affine) ===")
    # 构建 CT 的实心颅腔先验
    fixed_cranial_prior = extract_cranial_cavity(fixed_ct_head)
    # 构建 MNI 的实心颅腔先验 (纯二值化)
    moving_cranial_prior = ants.get_mask(moving_template)

    # 保存一下 CT 的先验掩模，方便你肉眼检查它有没有完美避开牙齿
    ants.image_write(fixed_cranial_prior, os.path.join(output_dir, "1_CT_Cranial_Prior_Mask.nii.gz"))

    # 【关键打法】：只用纯几何模型进行配准，使用 MSE 误差。彻底免疫所有灰度干扰！
    prior_reg = ants.registration(
        fixed=fixed_cranial_prior,
        moving=moving_cranial_prior,
        type_of_transform='Affine',  # 允许平移、旋转、缩放
        aff_metric='meansquares',  # 几何形状拟合
        reg_iterations=(120, 100, 50)
    )

    # 可视化先验配准结果 (把变形后的 MNI 模板贴回原 CT 上看位置)
    moving_prior_aligned = ants.apply_transforms(fixed_ct_head, moving_template, prior_reg['fwdtransforms'])
    visualize_registration(fixed_ct_head, moving_prior_aligned,
                           "Stage 1: A Priori Geometrical Alignment",
                           os.path.join(output_dir, "1_prior_aligned.png"))

    print("\n=== 阶段 2: 颅腔约束下的内部精细微调 (SyNOnly) ===")
    # 为了防止形变时跑出去，把 CT 洗白到 [0,80]
    arr_windowed = np.clip(fixed_ct_head.numpy(), 0, 80)
    fixed_ct_windowed = fixed_ct_head.new_image_like(arr_windowed)

    # 【绝招】：使用 SyNOnly！绝不能再做 Affine 了，死死锁定第一阶段算出来的正确位置！
    syn_reg = ants.registration(
        fixed=fixed_ct_windowed,
        moving=moving_template,
        type_of_transform='SyNOnly',
        initial_transform=prior_reg['fwdtransforms'],  # 继承先验位置
        syn_metric='mattes',  # 颅腔内部使用互信息对齐脑室
        fixed_mask=fixed_cranial_prior,  # 把形变严格囚禁在这个结界里，防止被外界牵拉
        reg_iterations=(100, 70, 50, 20),
        grad_step=0.1
    )

    moving_syn = syn_reg['warpedmovout']
    visualize_registration(fixed_ct_head, moving_syn,
                           "Stage 2: Confined SyN Deformable",
                           os.path.join(output_dir, "2_syn_deformable.png"))

    print("\n=== 阶段 3: 映射图谱至全身 CT 空间 ===")
    warped_labels_full_body = ants.apply_transforms(
        fixed=original_ct,
        moving=moving_labels,
        transformlist=syn_reg['fwdtransforms'],  # 包含了先验矩阵和形变场
        interpolator='nearestNeighbor'
    )

    output_label_path = os.path.join(output_dir, "BN_Atlas_246_in_FullBody.nii.gz")
    ants.image_write(warped_labels_full_body, output_label_path)
    print(f"[*] 先验约束下的 246区 ROI 已保存至: {output_label_path}")

    visualize_registration(fixed_ct_head, warped_labels_full_body,
                           "Stage 3: 246 Labels on CT",
                           os.path.join(output_dir, "3_final_labels.png"))

    return output_label_path


if __name__ == "__main__":
    MY_CT_IMAGE = "E:/data/New/WANG JUAN/CT_WB_NoHead_s202.nii.gz"
    MY_TEMPLATE = "E:/data/brain 246/MNI152_T1_2mm_brain.nii.gz"
    MY_TEMPLATE_LABELS = "E:/data/brain 246/BN_Atlas_246_2mm.nii.gz"
    OUTPUT_FOLDER = "D:/Work1/project/registration_results"

    run_registration_pipeline(MY_CT_IMAGE, MY_TEMPLATE, MY_TEMPLATE_LABELS, OUTPUT_FOLDER)