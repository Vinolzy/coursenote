[TOC]

# 1 Introduction to CV

- **Image Acquisition**：图像处理的第一步，用于确定数字图像的来源。

- **Image Compression**：图像数据压缩方法的集合，是定义最明确的类别。

- **Image Manipulation**：包括旋转、缩放、调整对比度等图像编辑方法，用于改善图像质量。

- **Image Analysis**：用于检测感兴趣的对象并提取相关参数（如位置和大小）。

- **Image Processing**：源自信号处理，通过分割等方法突出感兴趣的对象，抑制图像其他部分（例如边缘）。

- **Video Processing**：包含大部分图像处理方法，同时利用视频的时间特性。

- **Machine Vision**：指在工业中使用视频处理、图像处理或图像分析的方法，称为机器视觉。

- **Computer Vision**：计算机的视觉系统，包含类似人类的高级算法，如人脸识别，通常涉及多摄像头应用。

| 术语                   | 描述                                                         |
| ---------------------- | ------------------------------------------------------------ |
| **Image Acquisition**  | The first step in image processing that identifies the source of digital images.（图像处理的第一步，用于确定数字图像的来源） |
| **Image Compression**  | Encompasses methods used to reduce the size of image data while maintaining acceptable quality.（包含减少图像数据大小的方法，同时保持可接受的质量） |
| **Image Manipulation** | Involves techniques used to edit images, such as rotating and enhancing quality.（包括编辑图像的技术，如旋转和提高质量） |
| **Image Analysis**     | Focuses on examining images to locate objects of interest and extract relevant parameters.（关注分析图像以找到感兴趣的对象并提取相关参数） |
| **Image Processing**   | Originates from signal processing and involves segmenting objects of interest while enhancing specific features.（源于信号处理，涉及分割感兴趣的对象并增强特定特征） |
| **Video Processing**   | Includes most image processing methods and leverages the temporal aspects of video data.（包含大多数图像处理方法，并利用视频数据的时间特性） |
| **Machine Vision**     | Refers to the application of imaging technologies in industrial settings for tasks like quality control.（指在工业环境中应用图像技术进行质量控制等任务） |
| **Computer Vision**    | Describes a computer's ability to interpret and understand visual information, similar to human vision.（描述计算机解释和理解视觉信息的能力，类似人类视觉） |

其它：

- **Recognition**：识别，指在图像中确定特定对象或模式的身份，如人脸识别或物体识别。

- **Segmentation**：分割，指将图像划分为不同的区域，每个区域对应不同的对象或背景，用于进一步的分析或处理。

- **Classification**：分类，指根据图像的特征将其归类到预定义的类别中，比如把图片分为“猫”或“狗”。

- **Detection**：检测，指在图像中定位和标注目标对象，通常会给出对象的边界框或具体位置。



# 2 IMAGE PROCESSING FUNDAMENTALS

## 2.1 概念

**Sampling(采样)**
Converting a continuous analog image into a digital form by selecting a finite number of discrete points or pixels from the continuous image.(将连续的模拟图像转换为数字形式，通过从连续图像中选择有限数量的离散点或像素)

- Determine the number of pixels and divide the image into a grid.(确定像素数量，将图像划分为网格)

**Quantization(量化)**
Assigning discrete numerical values (usually integers) to the pixel intensity levels in a digital image.(将数字图像中的像素强度分配为离散的数值（通常是整数）)

- Grayscale/Color, bit depth, loss of information.(灰度/颜色，位深度，信息丢失)



## 2.2 图像操作

1. **Arithmetic Operations（算术操作） → Brightness（亮度）**
   通过像素的加减法操作，可以调整图像的亮度。例如，通过加法增加像素值来使图像变亮，或通过减法降低像素值来使图像变暗。
2. **Set & Logical Operations（集合和逻辑操作） → Segmentation（分割）**
   使用集合和逻辑运算（如AND、OR、NOT）可以进行图像分割，帮助识别和分离不同的图像区域。例如，用阈值分割图像，将像素分类为前景和背景。
3. **Spatial Operations（空间操作） → Sharpening（锐化）**
   空间操作包括卷积等操作，用于增强图像边缘和细节，提高图像的清晰度，从而达到锐化效果。
4. **Vector & Matrix Operations（向量和矩阵操作） → Rotation（旋转）**
   使用向量和矩阵运算可以实现图像的旋转变换，将图像在不同角度进行旋转，以满足特定的视角需求。
5. **Image Transforms（图像变换） → Reconstruction（重建）**
   通过傅里叶变换或小波变换等图像变换操作，可以从频域信息中重建图像，或将图像还原到特定的结构形式，用于图像复原和重建。
6. **Probabilistic Methods（概率方法） → Denoising（去噪）**
   使用概率统计方法可以去除图像中的噪声，提高图像质量。这种方法在降低噪声的同时保留图像的细节，常见的方法包括均值滤波和高斯滤波等。



## 2.3 Python图像变换及相关概念

- 代码来源于Tutorial 2

### 2.3.1 基础知识

#### 2.3.1.1 BRG和RGB

**RGB (Red, Green, Blue)**：这是图像处理中最常见的颜色格式，红、绿、蓝三个颜色通道按顺序排列，适用于PIL和Matplotlib等库。

**BGR (Blue, Green, Red)**：这是OpenCV默认使用的颜色格式，颜色通道顺序为蓝、绿、红。这一格式可以在OpenCV中提高处理效率，但在显示前通常需要转换成RGB。

#### 2.3.1.2 灰度图像

将图像转换为灰度图像通常有以下目的：

1. **减少计算量**：灰度图像只有一个通道，数据量比彩色图像少，适合在需要高效处理的场景下使用。
2. **简化分析**：灰度图像没有颜色信息，强调亮度强度，便于进行边缘检测、轮廓提取等任务。
3. **保持关键特征**：在某些图像处理任务中，颜色信息不重要，只需要考虑亮度信息即可。

<font color = blue>**转换原理**</font>

灰度图像是通过将RGB三个颜色通道的值按一定比例混合得到的，常见转换公式是： 

$Gray=0.299×R+0.587×G+0.114×B$

该公式是基于人眼对不同颜色的敏感度来确定权重的，其中绿色权重大，因为人眼对绿色较为敏感。



### 2.3.2 常用库

```
import numpy as np
import pandas as pd
import cv2 as cv
from google.colab.patches import cv2_imshow # for image display
from skimage import io
from PIL import Image
import matplotlib.pylab as plt
```

1. **numpy (np)**：用于数值计算和矩阵操作，特别适合处理大规模数组和矩阵运算。
2. **pandas (pd)**：提供数据分析和数据处理工具，常用于数据清洗和数据操作，特别是表格和时间序列数据。
3. **cv2 (cv)**：OpenCV的Python接口，用于图像处理和计算机视觉任务，支持图像读取、编辑、变换、滤波等操作。
4. **google.colab.patches.cv2_imshow**：在Google Colab中显示图像，因Colab不支持直接使用`cv2.imshow`，需要这个补丁。
5. **skimage.io**：scikit-image库的一部分，支持读取和写入图像文件，专注于科学图像处理。
6. **PIL.Image**：Python Imaging Library的接口，用于基本图像操作，如图像打开、转换、保存、处理等。
7. **matplotlib.pylab (plt)**：matplotlib的一个模块，用于数据可视化，支持创建各种类型的图表和图像展示。

### 2.3.3 代码详解

#### 2.3.3.1 加载URL图片

```
urls = ["https://upload.wikimedia.org/wikipedia/commons/1/14/Gambar_Dewan_Tunku_Canselor%2C_Universiti_Malaya.jpg",
       "https://harissalehanportraiture.wordpress.com/wp-content/uploads/2014/11/img_1123.jpg"]

for url in urls:
  image = io.imread(url)
  image_2 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  final_frame = cv.hconcat((image, image_2))
  cv2_imshow(final_frame)
  print('\n')
```

1. **`io.imread(url)`**
   作用：从给定URL读取图像并加载为NumPy数组，来自`skimage`库，适合读取网络上的图像。
2. **`cv.cvtColor(image, cv.COLOR_BGR2RGB)`**
   作用：将图像的颜色格式从BGR转换为RGB。OpenCV默认使用BGR格式，而常见的显示工具和Matplotlib则使用RGB格式。`cv.COLOR_BGR2RGB` 是转换模式常量。
3. **`cv.hconcat((image, image_2))`**
   作用：水平连接两个图像数组（即`image`和`image_2`），生成一个组合后的图像，方便对比显示。这是OpenCV的一个函数。

#### 2.3.3.2 Arithmetic Operations(算数运算)

算术运算是对图像像素值进行的基础数学操作，包括加法、减法、乘法和除法。它们用于调整亮度、增强对比度以及合并多张图像。例如，将常数值加到所有像素上可以使图像变亮，而减法可以用于对比度调整。

**1、读取图片**

```
img1 = io.imread("https://www.catschool.co/wp-content/uploads/2023/06/orange-tabby-kitten-1024x731.png")
img2 = io.imread("https://harissalehanportraiture.wordpress.com/wp-content/uploads/2014/11/img_1123.jpg")
```

**2、resize, 形状, 数据类型**

```
# Resize images if they are not the same size
if img1.shape != img2.shape:
    img2 = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    
# Check the image matrix data type (could know the bit depth of the image)
print(img1.dtype)
# Check the height of image
print(img1.shape[0])
# Check the width of image
print(img1.shape[1])
# Check the number of channels of the image
print(img1.shape[2])
```

- resize函数：调整图像`img2`的大小，使其宽度和高度与`img1`相同。`img1.shape[1]`和`img1.shape[0]`分别表示`img1`的宽度和高度。

- `img1.shape` 和 `img2.shape`：返回图像的形状，`shape`属性通常表示图像的维度：高度、宽度和通道数。通道数为4一般表示还有一个Alpha透明通道。

- `img1.dtype` 和 `img2.dtype`：检查图像的像素数据类型（如`uint8`或`float32`），即图像的位深度。

**3、灰度图像**

将彩色图像`img1`和`img2`从BGR颜色格式转换为灰度格式。`cv.COLOR_BGR2GRAY`是OpenCV中指定转换的模式常量。

```
# Convert to Grayscale
img1_g = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_g = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
```

**4、加法**

```
added_image = cv.add(img1_g, img2_g)
```

- `cv.add(img1_g, img2_g)`对灰度图像`img1_g`和`img2_g`的像素进行逐元素加法

将两个不相关的灰度图像相加会产生一个新图像，新的像素值是两个原图像对应像素值的和。这种操作在图像处理中的效果具体如下：

1. **亮度叠加**：新图像的亮度通常会变亮，因为两个像素值的叠加往往使得最终值更大。例如，如果一个图像较暗，另一个图像较亮，叠加后会得到更高的亮度效果。
2. **对比度变化**：由于两幅图像不相关，相加后的图像可能会出现不规则的亮度区域和边缘。这可能导致对比度增加，使图像显得更加"杂乱"或“重叠”。
3. **溢出处理**：如果像素值超过灰度图的最大值（通常为255），一些图像处理库会将其截断为255，从而出现“饱和”区域。这些区域会显示为纯白色，失去细节。



**5、减法**

```
# Perform subtraction
sub_image = cv.subtract(img1_g, img2_g)
```

对灰度图像`img1_g`和`img2_g`的像素进行逐元素减法运算，用于增强对比度或突出差异。

将两个不相关的灰度图像相减会产生一个新图像，其像素值是两个原图像对应像素值的差。由于灰度图像的像素值范围通常在0到255之间，减法运算会产生以下效果：

1. **亮度差异**：相减后，结果图像反映出两个图像之间的亮度差异。相似区域的像素差值会较小，而差异较大的区域则会出现更高或更低的像素值。对于完全相同的区域，差值为零，显示为黑色（0），表明无差异。
2. **阴影或轮廓增强**：图像相减的结果可以用来强调两个图像的不同之处。这个技术常用于运动检测、边缘提取或图像对比度分析，突出差异或移动部分。
3. **负值处理**：如果相减后的像素值为负数，一些图像处理库会将其截断为0，导致该区域显示为黑色。在OpenCV中，`cv.subtract()`自动处理负值，将其设置为0，避免出现负数问题。

**6、乘法**

```
multiplied_image = cv.multiply(img1_g, img2_g)
```

将两个不相关的灰度图像相乘会产生一个新的图像，每个像素的值是两个原图像对应像素值的乘积。相乘的效果在图像处理中具体表现如下：

1. **亮度变化**：相乘后的图像亮度会根据两个图像的亮度情况进行放大或缩小。相对较高亮度的像素值相乘后会进一步增大，使该区域变得更亮；而当一个或两个图像的像素值较低时，相乘结果会减小，变得更暗。
2. **对比度增强**：乘法运算会放大高亮和暗区之间的差异，因此可能使图像对比度增强。尤其是当一个图像有显著的黑白区域时，乘法会保留这些区域的差异。
3. **溢出处理**：灰度图像像素值通常在0到255之间，相乘的结果如果超过255会被截断为255，导致图像出现“饱和”区域，显示为纯白。这种情况容易发生在高亮区域的相乘中。

**7、图像显示**

```
plt.figure(figsize=(14, 8))
plt.subplot(131), plt.imshow(img1_g, cmap = 'gray')
plt.title('Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img2_g, cmap = 'gray')
plt.title('Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(added_image, cmap = 'gray')
plt.title('Added Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

- `plt.figure(figsize=(14, 8))`：创建一个新的Matplotlib图形窗口，`figsize`指定窗口的大小（宽度和高度），以英寸为单位。

- `plt.subplot(131), plt.imshow(img1_g, cmap='gray')`：在当前图形中创建一个3行1列的子图布局，并在第1个位置绘制图像`img1_g`。`cmap='gray'`将图像显示为灰度。
  - 同理132表示绘制到3行1列的第二个子图上

- `plt.title('Image 1')`, `plt.xticks([]), plt.yticks([])`：设置图像标题为“Image 1”，并去除X轴和Y轴的刻度，以获得更干净的显示效果。

- `plt.show()`：显示图形窗口中所有已定义的子图。



#### 2.3.3.3 Set and Logical Operations

