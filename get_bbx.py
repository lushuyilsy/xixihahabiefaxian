import numpy as np
import nibabel as nib
import os
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class OrganBoundingBoxExtractor:
    """
    从CT图像的像素级标注中提取器官的bounding box
    支持NIfTI格式（.nii, .nii.gz）的3D图像
    """

    def __init__(self,
                 label_map: Optional[Dict[int, str]] = None,
                 background_value: int = 0):
        """
        初始化提取器

        Parameters:
            label_map: 标签映射字典，键为标注像素值，值为器官名称
                      例：{1: '肝脏', 2: '脾脏', 3: '肾脏'}
            background_value: 背景像素值，默认0
        """
        self.label_map = label_map if label_map is not None else {}
        self.background_value = background_value

    def load_medical_image(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载医学图像（NIfTI格式）

        Parameters:
            file_path: 图像文件路径

        Returns:
            data: 3D图像数据数组 (z, y, x) 或 (x, y, z)，取决于图像存储格式
            affine:  affine矩阵，用于坐标转换
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        img = nib.load(file_path)
        data = img.get_fdata().astype(np.int32)  # 标注数据转换为整数
        affine = img.affine

        print(f"成功加载图像: {file_path}")
        print(f"图像形状: {data.shape}")
        print(f"像素值范围: {np.min(data)} - {np.max(data)}")

        return data, affine

    def get_organ_bounding_boxes(self, mask_data: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        从标注数据中提取所有器官的bounding box

        Parameters:
            mask_data: 3D标注数据数组

        Returns:
            bboxes: 字典，键为器官标签值，值为包含bounding box信息的字典
                   格式: {label: {'min': np.array([xmin, ymin, zmin]),
                                 'max': np.array([xmax, ymax, zmax]),
                                 'center': np.array([xcenter, ycenter, zcenter]),
                                 'size': np.array([width, height, depth])}}
        """
        # 获取所有非背景的标签值
        unique_labels = np.unique(mask_data)
        organ_labels = [label for label in unique_labels if label != self.background_value]

        if not organ_labels:
            raise ValueError("未找到任何器官标注（所有像素都是背景）")

        bboxes = {}

        for label in organ_labels:
            # 找到该器官的所有像素坐标
            coords = np.where(mask_data == label)

            # 计算bounding box的最小和最大值（注意坐标顺序）
            # 假设坐标顺序为 (z, y, x) 或 (x, y, z)，保持与输入一致
            min_coords = np.array([np.min(coord) for coord in coords])
            max_coords = np.array([np.max(coord) for coord in coords])

            # 计算中心点和尺寸
            center = (min_coords + max_coords) / 2.0
            size = max_coords - min_coords + 1  # +1 因为包含边界像素

            bboxes[label] = {
                'min': min_coords,
                'max': max_coords,
                'center': center,
                'size': size
            }

        return bboxes

    def convert_to_world_coords(self, bboxes: Dict[int, Dict[str, np.ndarray]],
                                affine: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
        """
        将图像坐标的bounding box转换为世界坐标（基于affine矩阵）

        Parameters:
            bboxes: 图像坐标的bounding box字典
            affine: NIfTI图像的affine矩阵

        Returns:
            world_bboxes: 世界坐标的bounding box字典
        """
        world_bboxes = {}

        for label, bbox in bboxes.items():
            # 转换最小和最大坐标到世界坐标
            # 注意：需要在齐次坐标下进行转换 (x, y, z, 1)
            min_img = np.append(bbox['min'], 1)
            max_img = np.append(bbox['max'], 1)
            center_img = np.append(bbox['center'], 1)

            min_world = np.dot(affine, min_img)[:3]
            max_world = np.dot(affine, max_img)[:3]
            center_world = np.dot(affine, center_img)[:3]

            world_bboxes[label] = {
                'min': min_world,
                'max': max_world,
                'center': center_world,
                'size': bbox['size'],  # 尺寸在世界坐标下需要重新计算
                'world_size': np.abs(max_world - min_world)
            }

        return world_bboxes

    def print_bbox_info(self, bboxes: Dict[int, Dict[str, np.ndarray]],
                        use_world_coords: bool = False):
        """
        打印bounding box信息
        """
        coord_type = "世界坐标" if use_world_coords else "图像坐标"
        print(f"\n{'=' * 50}")
        print(f"器官Bounding Box信息 ({coord_type})")
        print(f"{'=' * 50}")

        for label, bbox in bboxes.items():
            organ_name = self.label_map.get(label, f"器官_{label}")
            print(f"\n{organ_name} (标签值: {label}):")
            print(f"  最小坐标: {np.round(bbox['min'], 3)}")
            print(f"  最大坐标: {np.round(bbox['max'], 3)}")
            print(f"  中心点: {np.round(bbox['center'], 3)}")
            print(f"  尺寸: {np.round(bbox['size'], 3)}")
            if use_world_coords and 'world_size' in bbox:
                print(f"  世界尺寸: {np.round(bbox['world_size'], 3)} mm")

    def visualize_bbox_slice(self, ct_data: np.ndarray, mask_data: np.ndarray,
                             bboxes: Dict[int, Dict[str, np.ndarray]],
                             slice_idx: Optional[int] = None,
                             axis: int = 2):
        """
        可视化某个切片上的CT图像、标注和bounding box

        Parameters:
            ct_data: CT图像数据
            mask_data: 标注数据
            bboxes: bounding box字典
            slice_idx: 要显示的切片索引，如果为None则显示中心切片
            axis: 切片轴 (0: z轴, 1: y轴, 2: x轴)
        """
        # 设置默认切片为中心切片
        if slice_idx is None:
            slice_idx = ct_data.shape[axis] // 2

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 显示CT图像
        if axis == 0:
            ct_slice = ct_data[slice_idx, :, :]
        elif axis == 1:
            ct_slice = ct_data[:, slice_idx, :]
        else:  # axis == 2
            ct_slice = ct_data[:, :, slice_idx]

        ax1.imshow(ct_slice, cmap='gray', origin='lower')
        ax1.set_title(f'CT图像 - {["Z", "Y", "X"][axis]}轴切片 {slice_idx}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        # 显示标注和bounding box
        if axis == 0:
            mask_slice = mask_data[slice_idx, :, :]
        elif axis == 1:
            mask_slice = mask_data[:, slice_idx, :]
        else:  # axis == 2
            mask_slice = mask_data[:, :, slice_idx]

        ax2.imshow(mask_slice, cmap='viridis', origin='lower', alpha=0.7)
        ax2.set_title(f'器官标注 + Bounding Box - {["Z", "Y", "X"][axis]}轴切片 {slice_idx}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')

        # 绘制每个器官的bounding box在当前切片上的投影
        colors = plt.cm.get_cmap('tab10', len(bboxes))

        for i, (label, bbox) in enumerate(bboxes.items()):
            organ_name = self.label_map.get(label, f"器官_{label}")
            color = colors(i)

            # 根据当前切片轴，获取bounding box在该切片上的范围
            if axis == 0:  # Z轴切片，显示Y-X平面
                y_min, x_min = bbox['min'][1], bbox['min'][2]
                y_max, x_max = bbox['max'][1], bbox['max'][2]
            elif axis == 1:  # Y轴切片，显示Z-X平面
                z_min, x_min = bbox['min'][0], bbox['min'][2]
                z_max, x_max = bbox['max'][0], bbox['max'][2]
            else:  # X轴切片，显示Z-Y平面
                z_min, y_min = bbox['min'][0], bbox['min'][1]
                z_max, y_max = bbox['max'][0], bbox['max'][1]

            # 绘制矩形
            rect = plt.Rectangle((x_min if axis != 2 else z_min,
                                  y_min if axis != 1 else z_min),
                                 (x_max - x_min + 1) if axis != 2 else (z_max - z_min + 1),
                                 (y_max - y_min + 1) if axis != 1 else (y_max - y_min + 1),
                                 linewidth=2, edgecolor=color, facecolor='none',
                                 label=organ_name)
            ax2.add_patch(rect)

        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def visualize_3d_bbox(self, bboxes: Dict[int, Dict[str, np.ndarray]]):
        """
        3D可视化所有器官的bounding box
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        colors = plt.cm.get_cmap('tab10', len(bboxes))

        for i, (label, bbox) in enumerate(bboxes.items()):
            organ_name = self.label_map.get(label, f"器官_{label}")
            color = colors(i)

            # 获取bounding box的尺寸
            x, y, z = bbox['min']
            dx, dy, dz = bbox['size']

            # 创建3D立方体
            xx = [x, x + dx, x + dx, x, x, x + dx, x + dx, x]
            yy = [y, y, y + dy, y + dy, y, y, y + dy, y + dy]
            zz = [z, z, z, z, z + dz, z + dz, z + dz, z + dz]

            # 绘制立方体的边
            for edge in [(0, 1), (1, 2), (2, 3), (3, 0),
                         (4, 5), (5, 6), (6, 7), (7, 4),
                         (0, 4), (1, 5), (2, 6), (3, 7)]:
                ax.plot3D([xx[edge[0]], xx[edge[1]]],
                          [yy[edge[0]], yy[edge[1]]],
                          [zz[edge[0]], zz[edge[1]]],
                          color=color, linewidth=2, label=organ_name if i == 0 else "")

            # 标记中心点
            ax.scatter3D(bbox['center'][0], bbox['center'][1], bbox['center'][2],
                         color=color, s=50, marker='o', edgecolors='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Bounding Boxes of Organs')

        # 处理图例（避免重复）
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

        plt.show()

    def save_bbox_results(self, bboxes: Dict[int, Dict[str, np.ndarray]],
                          output_file: str, use_world_coords: bool = False):
        """
        保存bounding box结果到文本文件
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"器官Bounding Box结果 ({'世界坐标' if use_world_coords else '图像坐标'})\n")
            f.write("=" * 80 + "\n\n")

            for label, bbox in bboxes.items():
                organ_name = self.label_map.get(label, f"器官_{label}")
                f.write(f"器官名称: {organ_name}\n")
                f.write(f"标签值: {label}\n")
                f.write(f"最小坐标: {np.round(bbox['min'], 3)}\n")
                f.write(f"最大坐标: {np.round(bbox['max'], 3)}\n")
                f.write(f"中心点坐标: {np.round(bbox['center'], 3)}\n")
                f.write(f"尺寸: {np.round(bbox['size'], 3)}\n")
                if use_world_coords and 'world_size' in bbox:
                    f.write(f"世界尺寸 (mm): {np.round(bbox['world_size'], 3)}\n")
                f.write("-" * 50 + "\n\n")

        print(f"\n结果已保存到: {output_file}")

    def run(self, ct_file_path: str, mask_file_path: str,
            output_dir: str = "./bbox_results",
            visualize: bool = True,
            save_results: bool = True):
        """
        完整流程：加载数据 -> 提取bounding box -> 可视化 -> 保存结果

        Parameters:
            ct_file_path: CT图像文件路径
            mask_file_path: 标注mask文件路径
            output_dir: 输出目录
            visualize: 是否可视化结果
            save_results: 是否保存结果
        """
        # 创建输出目录
        if save_results and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. 加载数据
            print("正在加载数据...")
            ct_data, ct_affine = self.load_medical_image(ct_file_path)
            mask_data, mask_affine = self.load_medical_image(mask_file_path)

            # 验证CT和mask的形状是否一致
            if ct_data.shape != mask_data.shape:
                raise ValueError(f"CT图像和标注mask形状不一致！CT: {ct_data.shape}, Mask: {mask_data.shape}")

            # 2. 提取bounding box（图像坐标）
            print("\n正在提取Bounding Box...")
            img_bboxes = self.get_organ_bounding_boxes(mask_data)

            # 3. 转换为世界坐标
            world_bboxes = self.convert_to_world_coords(img_bboxes, ct_affine)

            # 4. 打印结果
            self.print_bbox_info(img_bboxes, use_world_coords=False)
            self.print_bbox_info(world_bboxes, use_world_coords=True)

            # 5. 可视化
            if visualize:
                print("\n正在生成可视化...")
                # 可视化中心切片
                self.visualize_bbox_slice(ct_data, mask_data, img_bboxes)
                # 可视化3D bounding box
                self.visualize_3d_bbox(img_bboxes)

            # 6. 保存结果
            if save_results:
                self.save_bbox_results(img_bboxes,
                                       os.path.join(output_dir, "bbox_image_coords.txt"),
                                       use_world_coords=False)
                self.save_bbox_results(world_bboxes,
                                       os.path.join(output_dir, "bbox_world_coords.txt"),
                                       use_world_coords=True)

                # 可选：保存带有bounding box的mask图像（NIfTI格式）
                self.save_bbox_mask(mask_data, img_bboxes,
                                    os.path.join(output_dir, "mask_with_bbox.nii.gz"))

            print("\n处理完成！")
            return img_bboxes, world_bboxes

        except Exception as e:
            print(f"\n处理过程中出错: {str(e)}")
            raise

    def save_bbox_mask(self, mask_data: np.ndarray, bboxes: Dict[int, Dict[str, np.ndarray]],
                       output_path: str):
        """
        保存带有bounding box标记的mask图像
        """
        # 创建一个新的mask，在bounding box边界上标记特殊值
        bbox_mask = mask_data.copy()
        boundary_value = np.max(mask_data) + 1

        for label, bbox in bboxes.items():
            z_min, y_min, x_min = bbox['min'].astype(int)
            z_max, y_max, x_max = bbox['max'].astype(int)

            # 标记bounding box的边界
            # Z方向边界
            bbox_mask[z_min, y_min:y_max + 1, x_min:x_max + 1] = boundary_value
            bbox_mask[z_max, y_min:y_max + 1, x_min:x_max + 1] = boundary_value

            # Y方向边界
            bbox_mask[z_min:z_max + 1, y_min, x_min:x_max + 1] = boundary_value
            bbox_mask[z_min:z_max + 1, y_max, x_min:x_max + 1] = boundary_value

            # X方向边界
            bbox_mask[z_min:z_max + 1, y_min:y_max + 1, x_min] = boundary_value
            bbox_mask[z_min:z_max + 1, y_min:y_max + 1, x_max] = boundary_value

        # 保存为NIfTI文件
        nib_img = nib.Nifti1Image(bbox_mask.astype(np.int16), np.eye(4))
        nib.save(nib_img, output_path)
        print(f"带有Bounding Box的mask已保存到: {output_path}")


# ==============================================================================
# 使用示例
# ==============================================================================
if __name__ == "__main__":
    # 1. 配置参数
    CT_FILE_PATH = "E:/body.nii.gz"  # 替换为你的CT图像路径
    MASK_FILE_PATH = ("E:/gzx_label/liver-right002.nii.gz-liver96-right-label.nii.g_VDGU.nii.gz")  # 替换为你的标注mask路径

    # 定义标签映射（根据你的标注体系修改）
    LABEL_MAP = {
        1: "肝脏",
        2: "脾脏",
        3: "左肾",
        4: "右肾",
        5: "胃",
        6: "胰腺"
        # 添加更多器官...
    }

    # 2. 创建提取器实例
    extractor = OrganBoundingBoxExtractor(
        label_map=LABEL_MAP,
        background_value=0  # 背景像素值
    )

    # 3. 运行完整流程
    img_bboxes, world_bboxes = extractor.run(
        ct_file_path=CT_FILE_PATH,
        mask_file_path=MASK_FILE_PATH,
        output_dir="./bbox_results",
        visualize=True,
        save_results=True
    )

    # 4. 如果你只想获取bounding box坐标，可以这样使用：
    # mask_data, _ = extractor.load_medical_image(MASK_FILE_PATH)
    # bboxes = extractor.get_organ_bounding_boxes(mask_data)