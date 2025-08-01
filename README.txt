README：
一个基于astroalign的预处理、清洗并筛选天文图像并将图像星点对齐并叠加的Python项目。该工具可完成背景噪点减除、星点检测、基于大小与形态的筛选，并输出通过质量控制的图像帧。之后可以将图像中的星点进行对齐，并按照多种方式进行叠加。

项目结构：
├── data/              # 原始图像文件夹（FITS, PNG, JPG, TIFF）
├── darkfield/         # 与原始图像匹配的暗场图像
├── data_cleaned/      # 输出清洗后图像的文件夹
├── clean_images.py    # 主处理脚本
├── model.py           # 点对齐并叠加脚本
└── README.md          # 本说明文档

---------------------------------------------------------------------------------------------------

环境依赖：
Python 3.7 或更高版本
安装依赖包：
pip install numpy astroalign astropy pillow sep scipy

---------------------------------------------------------------------------------------------------

该项目中包含数据预处理和星点对齐并叠加的功能，其中：
    1.data_process.py是图像预处理的程序，脚本逐个处理INPUT_DIR中的图像。若为FITS格式，
    使用astropy.io.fits.getdata读取。若为其他图像格式（如PNG、JPG），使用PIL.Image转为灰度图。
    若在 DARKFIELD_DIR 中存在匹配名称的暗场图像，则进行像素级减除。
    使用 sep.Background 构建背景模型。将背景从原图中减去，以突显星点。
    提取图像中心的 1000×1000 区域，用于调试星点检测参数。
    使用 高斯滤波（σ=1.0） 降低高频噪声。
    调用 sep.extract 进行星点检测，参数为阈值和最小连通面积。
    根据 a 和 b 轴计算每个星点的 FWHM。使用中位数与 MAD 统计剔除离群点。
    计算每个星点的椭率和峰值ADU。若有星点不满足 FWHM、椭率或饱和限制，则丢弃整幅图像。
    保存清洗图像，将符合要求的图像复制至 OUTPUT_DIR。

    配置说明:
    参数名称	                    描述说明	               默认值
    INPUT_DIR	         原始图像所在的文件夹路径	       data
    OUTPUT_DIR	           清洗后图像的输出路径	        data_cleaned
    DARKFIELD_DIR	          暗场图像路径	         darkfield
    MIN_STAR_COUNT	    保留图像所需的最小星点数量	       1000
    THRESHOLD_SIGMA	     星点检测阈值（以 σ 表示）	       10.0
    FWHM0	           预期星点的 FWHM 值（像素单位）	    3.0
    ELLIP_MAX	         最大允许的椭率（1 - b/a）	        0.5
    SATURATION_LIMIT	    饱和前的最大ADU值	           65535
    MINAREA	              SEP对象最小连通像素面积	        20
    OUTLIER_MAD_THRESH	用于FWHM异常值剔除的MAD倍数	        3.0

---------------------------------------------------------------------------------------------------

    2.model.py是星点对齐并叠加的程序，里面包含了对齐、堆叠、保存、可视化几个步骤。
    运行前先确保以已经将原始数据集清理，并把main函数中的路径改为清理后得到的数据集
    的路径。其中数据列表的第一个图像为target，所有后续的帧都会配准到这张图的坐标系上。

    图像不同的叠加方式如下：
    Mean Stack - 所有对齐后帧在每个像素位置的算术平均值，噪声随机分布时可显著降低噪声。
    Median Stack - 每个像素位置取所有帧的中位数，对偶发的亮斑或暗斑有更强的抗干扰能力，但细节平滑程度稍差。
    Max Stack - 每个像素位置取所有帧的最大值，用来突出明亮瞬态。
    Min Stack - 每个像素位置取所有帧的最小值，常用于刻画背景或弱光结构，也可检出暗斑。
    Std Dev - 每个像素位置的标准差，用来显示帧间变化／抖动或瞬态噪声在图像上的分布。
    Sigma‑Clipped Stack - 对每个像素先做 sigma‑clipping（剔除偏离平均值超过 2.5σ 的像素），
    然后对剩余值求平均，结合了均值的平滑效果和对极端像素的剔除能力，通常是最干净、最鲁棒的叠加结果。

    叠加后会生成一个所有叠加方式的预览图，并将原图储存在results文件夹中。

MIT License - Use freely in your projects!