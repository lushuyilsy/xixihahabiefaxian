import SimpleITK as sitk
import numpy as np
import os


def register_liver_template_and_get_bbx(target_mask_path, template_labels_path, output_dir="."):
    """
    通过非刚性配准将肝脏左右叶模板填充到患者肝脏Mask中，并提取BBX。

    参数:
    target_mask_path (str): 患者的完整肝脏Mask文件路径 (例如 .nii.gz)
                           (假设 0=背景, 1=肝脏)
    template_labels_path (str): 模板Mask文件路径，已标注左右 (例如 .nii.gz)
                                (假设 0=背景, 1=左叶, 2=右叶)
    output_dir (str): 保存中间和最终结果的目录。
    """

    print("--- 1. 加载影像 ---")

    # 加载固定影像 (患者Mask)，并确保为浮点型以便配准
    fixed_image = sitk.ReadImage(target_mask_path, sitk.sitkFloat32)

    # 加载移动影像 (模板标签)，并确保为整数型
    moving_labels = sitk.ReadImage(template_labels_path, sitk.sitkUInt8)

    # 从模板标签创建用于配准的二值Mask (移动影像)
    # 我们只配准肝脏的整体形状
    moving_image = sitk.Cast(moving_labels > 0, sitk.sitkFloat32)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print("--- 2. 设置Elastix非刚性配准 ---")

    # 使用SimpleITK的Elastix进行配准
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)

    # 设置参数: 依次执行 Affine (仿射) 和 B-spline (非刚性) 配准
    # 这是实现非刚性配准的标准且稳健的方法
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
    parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))

    elastixImageFilter.SetParameterMap(parameterMapVector)

    # (可选) 记录日志
    # elastixImageFilter.LogToConsoleOn()

    print("--- 3. 执行配准 (这可能需要几分钟) ---")
    elastixImageFilter.Execute()

    # 获取结果 (这是配准后的 *二值* 模板，主要用于调试)
    # result_image = elastixImageFilter.GetResultImage()
    # sitk.WriteImage(result_image, os.path.join(output_dir, "debug_warped_binary_mask.nii.gz"))

    # 获取最重要的输出：变换参数图
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    print("--- 4. 将变换应用到原始的左右肝标签 ---")

    # 使用 Transformix 来应用已计算的变换
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(moving_labels)
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)

    # *** 关键步骤 ***
    # 当变换 *标签* 时，必须使用 "NearestNeighbor" 插值 (Order=0)
    # 否则，标签值（如1和2）会被插值成1.5, 1.8等
    # 我们修改参数图的最后一个B-spline变换
    # GetDefaultParameterMap("bspline") 默认使用 Order=3

    # 修改B-spline参数图
    final_bspline_map = transformParameterMap[-1]  # 获取最后一个参数图 (bspline)
    final_bspline_map["FinalBSplineInterpolationOrder"] = ("0",)

    # 更新Transformix的参数
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)

    # 执行变换
    warped_labels = transformixImageFilter.Execute()

    # 保存形变后的标签图 (可选)
    warped_labels_path = os.path.join(output_dir, "warped_left_right_labels.nii.gz")
    sitk.WriteImage(warped_labels, warped_labels_path)
    print(f"已保存形变后的标签图: {warped_labels_path}")

    print("--- 5. 后处理：使用患者Mask约束标签 ---")

    # 加载患者Mask为整数型用于相乘
    target_mask_int = sitk.ReadImage(target_mask_path, sitk.sitkUInt8)

    # 使用二值乘法，确保标签只存在于患者的肝脏Mask内
    # (target_mask_int > 0) 会创建一个 0/1 的Mask
    final_labels = warped_labels * (target_mask_int > 0)

    # 保存最终的标签图
    final_labels_path = os.path.join(output_dir, "final_patient_left_right_labels.nii.gz")
    sitk.WriteImage(final_labels, final_labels_path)
    print(f"已保存最终约束后的标签图: {final_labels_path}")

    print("--- 6. 提取Bounding Boxes (BBX) ---")

    # 使用 SimpleITK 的 LabelShapeStatisticsImageFilter 来获取BBX
    # 这是最安全的方法，因为它处理图像的元数据（如origin, spacing）
    label_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    label_stats_filter.Execute(final_labels)

    bounding_boxes = {}

    # 假设 1=左叶, 2=右叶
    # 检查标签是否存在 (以防配准后某个标签丢失)
    if label_stats_filter.HasLabel(1):
        # BBX格式: (x_start_index, y_start_index, z_start_index, x_size, y_size, z_size)
        bbx_left = label_stats_filter.GetBoundingBox(1)
        bounding_boxes['left_lobe'] = bbx_left
        print(f"左叶 (Label 1) BBX: {bbx_left}")

    if label_stats_filter.HasLabel(2):
        bbx_right = label_stats_filter.GetBoundingBox(2)
        bounding_boxes['right_lobe'] = bbx_right
        print(f"右叶 (Label 2) BBX: {bbx_right}")

    return bounding_boxes, final_labels_path


# --- 如何使用 ---

if __name__ == "__main__":

    # 1. 定义您的文件路径
    # 替换
    PATIENT_LIVER_MASK = "path/to/patient_liver_mask.nii.gz"
    TEMPLATE_LABELS = "path/to/template_left_right_labels.nii.gz"
    OUTPUT_DIRECTORY = "registration_output"

    # 2. 检查文件是否存在 (示例)
    if not os.path.exists(PATIENT_LIVER_MASK) or not os.path.exists(TEMPLATE_LABELS):
        print("=" * 50)
        print("错误：请先在 `if __name__ == \"__main__\":` 部分")
        print(f"将 'PATIENT_LIVER_MASK' (当前: {PATIENT_LIVER_MASK})")
        print(f"和 'TEMPLATE_LABELS' (当前: {TEMPLATE_LABELS})")
        print("替换为您自己的文件路径！")
        print("=" * 50)
    else:
        # 3. 执行主函数
        try:
            bboxes, final_mask_path = register_liver_template_and_get_bbx(
                target_mask_path=PATIENT_LIVER_MASK,
                template_labels_path=TEMPLATE_LABELS,
                output_dir=OUTPUT_DIRECTORY
            )

            print("\n--- 任务完成 ---")
            print(f"最终左右肝分割图已保存至: {final_mask_path}")
            print("提取到的Bounding Boxes (voxel-based):")
            print(bboxes)

            # 解释BBX格式
            print("\nBBX 格式为 (x_start, y_start, z_start, x_size, y_size, z_size)")
            if 'left_lobe' in bboxes:
                bb = bboxes['left_lobe']
                print(f"例如，左叶的X范围是: [{bb[0]}, {bb[0] + bb[3] - 1}]")

        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请确保 SimpleITK 已正确安装，并且文件路径无误。")