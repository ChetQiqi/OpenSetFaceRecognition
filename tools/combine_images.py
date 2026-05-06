"""
将两张图片左右并排拼接
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def combine_images_side_by_side():
    """把两张图片左右并排拼接，都调整为正方形"""

    output_dir = Path("threshold_analysis")

    # 读取两张图片
    img1_path = output_dir / "normalized_distribution_comparison.png"
    img2_path = output_dir / "far_frr_curves.png"

    print(f"读取图片: {img1_path}")
    print(f"读取图片: {img2_path}")

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    print(f"图1原始尺寸: {img1.size}")
    print(f"图2原始尺寸: {img2.size}")

    # 计算正方形边长（取所有图片宽高中的最小值）
    sizes = [img1.width, img1.height, img2.width, img2.height]
    square_size = min(sizes)

    print(f"\n正方形边长: {square_size} x {square_size}")

    # 调整两张图为正方形（如果原宽度大于正方形边长，则裁剪；如果小于，则填充）
    def fit_to_square(img, size):
        """将图片调整为正方形"""
        # 如果图片比正方形大，则缩小
        if img.width > size or img.height > size:
            img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
        else:
            # 如果图片比正方形小，则填充
            img_resized = Image.new('RGB', (size, size), (255, 255, 255))
            offset = ((size - img.width) // 2, (size - img.height) // 2)
            img_resized.paste(img, offset)

        return img_resized

    img1_square = fit_to_square(img1, square_size)
    img2_square = fit_to_square(img2, square_size)

    # 创建新图片（两个正方形左右拼接）
    total_width = square_size * 2
    total_height = square_size
    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # 粘贴图片
    combined_img.paste(img1_square, (0, 0))
    combined_img.paste(img2_square, (square_size, 0))

    # 保存
    output_path = output_dir / "threshold_analysis_combined.pdf"
    combined_img.save(output_path, dpi=(300, 300))

    print(f"\n✅ 拼接完成！")
    print(f"输出文件: {output_path}")
    print(f"最终尺寸: {combined_img.size} (两个 {square_size}x{square_size} 的正方形左右并排)")

    return output_path


if __name__ == "__main__":
    combine_images_side_by_side()
