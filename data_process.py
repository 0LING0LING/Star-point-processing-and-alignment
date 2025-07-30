import os
import shutil
import numpy as np
from astropy.io import fits
from PIL import Image
import sep
from scipy.ndimage import gaussian_filter

# --------------------------- 配置参数 ---------------------------
INPUT_DIR        = 'data'            # 原始图像文件夹路径
OUTPUT_DIR       = 'data_cleaned'    # 清洗后图像输出文件夹路径
DARKFIELD_DIR    = 'darkfield'       # 暗场图像文件夹路径

MIN_STAR_COUNT   = 1000     # 通过筛选后最少暗星数量阈值
THRESHOLD_SIGMA  = 10.0     # SEP 提取星点的信噪比阈值（单位：σ）
FWHM0            = 3.0      # 期望的星点 FWHM（全宽半高，单位：像素）
FWHM_MIN         = 0.5 * FWHM0  # 最小允许 FWHM
FWHM_MAX         = 1.5 * FWHM0  # 最大允许 FWHM
ELLIP_MAX        = 0.5      # 最大允许椭圆度（1 - b/a）
SATURATION_LIMIT = 65535    # 饱和 ADU 上限
MINAREA          = 20       # sep.extract 的最小连通像素数
OUTLIER_MAD_THRESH = 3.0    # 使用 MAD 进行离群过滤时的倍数阈值

# 创建输出文件夹（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("当前工作目录:", os.getcwd())
print("输入目录文件列表:", os.listdir(INPUT_DIR))

# 遍历输入目录中的所有文件
for fname in sorted(os.listdir(INPUT_DIR)):
    try:
        print("\n=== 处理文件:", fname)
        path = os.path.join(INPUT_DIR, fname)
        ext  = os.path.splitext(fname)[1].lower()
        print(" 扩展名:", ext)

        # 跳过非图像文件
        if ext not in ('.fits', '.png', '.jpg', '.jpeg', '.tiff'):
            print(" 跳过（不支持的扩展名）")
            continue

        # ---------------------- 1. 读取原图 ----------------------
        if ext == '.fits':
            # FITS 格式读取
            data = fits.getdata(path).astype(np.float32)
        else:
            # 常规图像读取并转为灰度
            data = np.array(Image.open(path).convert('L')).astype(np.float32)
        print("  图像形状:", data.shape)

        # ---------------------- 2. 暗场减除 ----------------------
        darkpath = os.path.join(DARKFIELD_DIR, fname)
        if os.path.exists(darkpath):
            print("  找到暗场，正在减除...")
            # 读取暗场图像
            if darkpath.endswith('.fits'):
                dark = fits.getdata(darkpath).astype(np.float32)
            else:
                dark = np.array(Image.open(darkpath).convert('L')).astype(np.float32)
            # 检查暗场与原图尺寸是否一致
            if dark.shape != data.shape:
                print("  WARNING: 暗场与原图大小不符，跳过")
                continue
            # 执行暗场减除
            data -= dark
        else:
            print("  未找到对应暗场，跳过暗场减除")

        # ------------------ 3. 背景建模与减除 ------------------
        print("  构建背景模型...")
        bkg = sep.Background(data)      # 使用 SEP 建立背景模型
        data_sub = data - bkg.back()    # 减去背景
        data_sub = np.ascontiguousarray(data_sub)
        print("  背景建模完成")

        # ---------- 4. （可选）子图提取测试 -------------
        # 为测试在图像中心区域提取效果，截取 1000×1000 子图
        h, w = data_sub.shape
        sub   = data_sub[h//2-500:h//2+500, w//2-500:w//2+500]
        sub_r = bkg.rms()[h//2-500:h//2+500, w//2-500:w//2+500]
        sub   = np.ascontiguousarray(sub)
        sub_r = np.ascontiguousarray(sub_r)
        print("  → 子图提取测试...")
        objs_sub = sep.extract(sub, THRESHOLD_SIGMA, err=sub_r, minarea=MINAREA)
        print(f"  ← 子图候选: {len(objs_sub)}")

        # ------------- 5. 全图高斯平滑后提取 --------------
        # 对去背景图像进行高斯平滑，减少高频噪声
        data_smooth = gaussian_filter(data_sub, sigma=1.0)
        data_smooth = np.ascontiguousarray(data_smooth)
        rms_full    = np.ascontiguousarray(bkg.rms())
        print("  → 平滑后全图提取...")
        objects = sep.extract(data_smooth, THRESHOLD_SIGMA, err=rms_full, minarea=MINAREA)
        print(f"  ← 全图候选: {len(objects)}")

        # 若无星点，直接跳过
        if len(objects) == 0:
            print("  丢弃（无星点）")
            continue

        # ----- 6. 基于 FWHM 的 MAD 离群过滤 -----
        # 计算每个目标的 FWHM
        a_vals = objects['a']
        b_vals = objects['b']
        sigmas = np.sqrt(a_vals * b_vals)
        fwhms  = 2.355 * sigmas
        # 计算中位数和 MAD
        med    = np.median(fwhms)
        mad    = np.median(np.abs(fwhms - med))
        # 保留在阈值内的目标
        mask   = np.abs(fwhms - med) <= OUTLIER_MAD_THRESH * mad
        filtered = objects[mask]
        print(f"  离群过滤后: {len(filtered)} / {len(objects)}")

        # 若过滤后目标数不足，跳过
        if len(filtered) < MIN_STAR_COUNT:
            print(f"  丢弃（暗星数 {len(filtered)} < {MIN_STAR_COUNT}）")
            continue

        # ------- 7. 形态 & 饱和度筛选 -------
        bad = False
        for obj in filtered:
            fwhm  = 2.355 * np.sqrt(obj['a'] * obj['b'])
            ellip = 1 - (obj['b'] / obj['a'] if obj['a'] > 0 else 0)
            y, x  = int(obj['y']), int(obj['x'])
            peak  = data[y, x]
            # 检查 FWHM 范围、椭圆度和峰值是否符合要求
            if not (FWHM_MIN <= fwhm <= FWHM_MAX and ellip <= ELLIP_MAX and peak < SATURATION_LIMIT):
                bad = True
                break
        if bad:
            print("  丢弃（星点形态或饱和度不合格）")
            continue

        # -------- 8. 保存合格图像 --------
        dst = os.path.join(OUTPUT_DIR, fname)
        shutil.copy2(path, dst)
        print("  保留:", fname)

    except Exception as e:
        print(f"  处理时出错: {e}")
        import traceback; traceback.print_exc()
