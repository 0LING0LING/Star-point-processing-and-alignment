import astroalign as aa
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits
from astropy.nddata import NDData
from PIL import Image
from skimage.transform import resize  # (未使用，但可用于后续图像重采样)

# --------------------------- 定义函数 ---------------------------

def load_image_as_nddata(path):
    """
    将文件加载为 NDData 对象，统一数据格式。
    支持 FITS 与常规图像（PNG/JPG/TIFF）。
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".fits":
        # 读取 FITS 文件
        hdul = fits.open(path)
        data = np.array(hdul[0].data).astype(np.float32)
        hdul.close()
        return NDData(data)

    else:
        # 读取常规图像并转换为灰度
        img = Image.open(path).convert("L")
        data = np.array(img).astype(np.float32)
        return NDData(data)


def normalize_image(img):
    """
    归一化图像到 [0,1] 范围，用于可视化或保存为普通图像。
    - 将 NaN 替换为 0
    - 按最大最小值线性缩放
    """
    img = np.nan_to_num(img, nan=0.0)
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val > 1e-5:
        return (img - min_val) / (max_val - min_val)
    else:
        # 若图像几乎是常数，直接返回原图
        return img


def save_image_with_format(image, ref_path, out_path):
    """
    根据参考路径的扩展名决定保存格式：
    - FITS：保存浮点数据
    - 其他（PNG/JPG/TIFF）：归一化并转换为 8-bit 灰度
    """
    ext = os.path.splitext(ref_path)[-1].lower()

    if ext == ".fits":
        # FITS 保存
        hdu = fits.PrimaryHDU(np.array(image, dtype=np.float32))
        hdu.writeto(out_path, overwrite=True)
    else:
        # 普通图像保存
        img = normalize_image(image)               # 归一化到 [0,1]
        img = (img * 255).astype(np.uint8)         # 转为 0–255
        Image.fromarray(img).save(out_path)


def sigma_clip_stack(image_list, sigma=2.5):
    """
    对齐后图像列表进行 Sigma-Clip 平均堆叠：
    - 计算像素均值与标准差
    - 排除偏离均值超过 sigma * std 的像素
    - 对剩余像素求均值
    """
    stack = np.stack(image_list, axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0)
    mask = np.abs(stack - mean) <= sigma * std
    masked_stack = np.ma.array(stack, mask=~mask)
    clipped_mean = np.ma.mean(masked_stack, axis=0).filled(0)
    return clipped_mean


def evaluate_noise(image):
    """
    使用中位数绝对偏差（MAD）作为图像噪声估计：
    MAD = median(|I - median(I)|)
    """
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    return mad


# --------------------------- 主流程 ---------------------------

def align_and_stack(image_paths, output_folder="results"):
    """
    对输入图像列表进行配准对齐，然后生成多种堆叠结果并保存：
    - mean, median, max, min, std, sigma-clipped
    同时输出每张图像的噪声估计值。
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. 读取第一张图作为参考 ---
    ref_nd = load_image_as_nddata(image_paths[0])
    ref_data = ref_nd.data
    aligned_images = [ref_data]                   # 存放所有对齐后的图像
    noise_estimates = [evaluate_noise(ref_data)]  # 存放噪声估计

    # --- 2. 依次对其他图像配准 ---
    for path in image_paths[1:]:
        nd = load_image_as_nddata(path)
        try:
            # astroalign 配准，返回对齐后图像和变换参数
            aligned, _ = aa.register(nd.data, ref_data)
            aligned_images.append(aligned)
            noise_estimates.append(evaluate_noise(aligned))
        except aa.MaxIterError:
            # 若收敛失败，则跳过该图
            print(f"[Warning] Registration failed for {path}, skipped.")

    # --- 3. 计算各种堆叠结果 ---
    mean_stack   = np.mean(aligned_images, axis=0)
    median_stack = np.median(aligned_images, axis=0)
    max_stack    = np.max(aligned_images, axis=0)
    min_stack    = np.min(aligned_images, axis=0)
    std_stack    = np.std(aligned_images, axis=0)
    sigma_stack  = sigma_clip_stack(aligned_images)

    # --- 4. 按格式保存堆叠结果 ---
    ref_ext = os.path.splitext(image_paths[0])[-1]
    save_image_with_format(mean_stack,   image_paths[0], os.path.join(output_folder, "mean"           + ref_ext))
    save_image_with_format(median_stack, image_paths[0], os.path.join(output_folder, "median"         + ref_ext))
    save_image_with_format(max_stack,    image_paths[0], os.path.join(output_folder, "max"            + ref_ext))
    save_image_with_format(min_stack,    image_paths[0], os.path.join(output_folder, "min"            + ref_ext))
    save_image_with_format(std_stack,    image_paths[0], os.path.join(output_folder, "std"            + ref_ext))
    save_image_with_format(sigma_stack,  image_paths[0], os.path.join(output_folder, "sigma_clipped" + ref_ext))

    # --- 5. 可视化所有堆叠结果 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    titles = ["Mean Stack", "Median Stack", "Max Stack",
              "Min Stack", "Std Dev", "Sigma-Clipped Stack"]
    images = [mean_stack, median_stack, max_stack, min_stack, std_stack, sigma_stack]

    for ax, img, title in zip(axes.ravel(), images, titles):
        ax.imshow(normalize_image(img), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "stack_summary.png"))
    plt.show()

    # --- 6. 打印噪声估计结果 ---
    for path, noise in zip(image_paths, noise_estimates):
        print(f"Noise (MAD) in {os.path.basename(path)}: {noise:.4f}")

    return sigma_stack


# 脚本入口：遍历 ./data 文件夹，处理所有 FITS/PNG/JPG/TIFF
# 当运行完data_process后会得到data_cleaned，需将路径替换为清洗后的数据集
if __name__ == "__main__":
    source_folder = "./data"
    files = sorted([
        os.path.join(source_folder, f)
        for f in os.listdir(source_folder)
        if f.lower().endswith(('.fits', '.png', '.jpg', '.tiff'))
    ])
    result = align_and_stack(files)
