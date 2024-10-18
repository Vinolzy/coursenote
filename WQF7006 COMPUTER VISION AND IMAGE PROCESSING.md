
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

#### 2.3.1.3 直方图

- 直方图

   是图像中像素值的统计图，显示每个像素值（通常从 0 到 255）出现的频率。它帮助我们了解图像的亮度分布。

  - **x 轴**：表示像素值的范围（0 表示黑色，255 表示白色）。
  - **y 轴**：表示每个像素值在图像中出现的次数（频率）。

直方图在图像处理中常用于：

1. **对比度分析**：可以通过直方图看到图像的像素值是否集中在某个区域，例如是否过暗或过亮。
2. **图像增强**：基于直方图的分布可以进行对比度拉伸、直方图均衡化等操作，改善图像质量。

#### 2.3.1.4 傅里叶变换

傅里叶变换是一种将**信号从时间（或空间）域转换到频率域**的数学方法。它的基本思想是将复杂的信号分解为不同频率的正弦波（或余弦波）的组合。

**如何理解傅里叶变换**

- **时间（或空间）域**：信号（或图像）在正常情况下是随着时间变化的，比如音频信号，或是空间上的变化，比如图像的亮度值。
- **频率域**：傅里叶变换将这些变化看作是不同频率波形的组合，然后将信号从时间/空间域转换为频率域。这表示信号中每个频率成分的强度。

**在图像中的应用**

- 图像也可以被看作是一个二维信号（X 和 Y 方向的亮度变化）。

- 傅里叶变换

   可以帮助我们分析图像中的频率分布：

  - **高频**：图像中变化剧烈的部分，如边缘、噪声。高频代表快速变化的细节。
  - **低频**：图像中变化平缓的部分，如平滑的区域。低频代表图像的整体结构或背景。

**为什么要用傅里叶变换**

1. **过滤噪声**：通过傅里叶变换，可以识别并去除图像中的高频噪声。
2. **压缩图像**：低频部分通常包含图像的主要信息，而高频部分可以被压缩或去除，从而减少图像的数据量。

**总结**

傅里叶变换可以将图像从空间表示转换为频率表示，帮助我们识别图像中的边缘、细节和噪声等高频部分，以及背景等低频部分。这对于图像处理、去噪和压缩非常有用。

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



#### 2.3.3.3 Set and Logical Operations(集合和逻辑运算)

集合和逻辑操作专注于基于逻辑条件操控像素值集合。常见的操作包括“与（AND）”、“或（OR）”和“非（NOT）”，这些操作有助于二值图像处理、掩膜制作以及图像分割。它们允许根据特定条件选择性地修改像素值，因此在对象检测和特征提取等任务中至关重要。

**1、生成二值图像**

```
# 应用阈值化来创建二值掩膜
_, mask1 = cv.threshold(img1_g, 128, 255, cv.THRESH_BINARY)
_, mask2 = cv.threshold(img2_g, 128, 255, cv.THRESH_BINARY)
```

- `128` 是阈值。像素值大于或等于 128 的位置会被设为 255（白色），否则设为 0（黑色）
- `255` 是最大值，表示阈值化后的像素值为 255（通常表示白色）
- `cv.THRESH_BINARY` 是阈值类型，表示进行二值化操作

**2、交集**

只有当两个图像的对应像素都为白色（255）时，结果图像中的该像素才为白色（255）。如果任意一个图像在某个像素位置为黑色（0），结果图像中的该像素就是黑色（0）。

```
intersection = cv.bitwise_and(mask1, mask2)
```

- 交集操作通常用于检测两个图像中共有的区域。在图像处理中，这常常用来提取两个图像中相同的特征或重叠部分。

**3、并集**

只要其中任意一个图像在某个像素位置为白色（255），结果图像中的该像素就为白色（255）。只有当两个图像在相应位置都为黑色（0）时，结果图像的该像素才为黑色（0）。

```
union = cv.bitwise_or(mask1, mask2)
```

- 并集操作用于检测两个图像中包含的所有信息，它展示了两个图像中所有的白色区域（感兴趣的区域），不论这些区域是否重叠。

**4、差集**

**(1)异或**

当 `mask1` 和 `mask2` 在同一像素位置有不同的值时（即一个为 255，一个为 0），结果图像中的该像素为白色（255）；如果两个图像在同一位置有相同的像素值（即都为 0 或都为 255），结果图像的该像素为黑色（0）。

```
difference1 = cv.bitwise_xor(mask1, mask2)  # mask1 - mask2
```

`bitwise_xor` 用于显示两个图像之间的不同部分，即那些在一个图像中出现而在另一个图像中没有出现的区域。

**(2)减法**

对于每个像素位置，如果 `mask1` 在该位置是 255 而 `mask2` 在该位置是 0，那么结果图像中的像素值为 255（白色）。如果 `mask1` 和 `mask2` 在该位置相同（例如都为 0 或都为 255），结果图像中的像素值为 0（黑色）。

```
difference2 = cv.subtract(mask1, mask2)    # mask2 - mask1
```

这用于展示 `mask1` 中存在但在 `mask2` 中不存在的区域。

**5、按位非**

将输入图像的每个像素值进行取反操作

```
logical_not = cv.bitwise_not(mask)
```



#### 2.3.3.4 Spatial Operations(空间操作)

空间操作是指基于像素的空间排列来处理图像的技术。这包括几何变换，如平移、旋转和缩放，以及空间滤波技术，如卷积和模糊。空间操作对于校正图像失真、增强特征以及为进一步分析准备图像至关重要。

```
# 加载图像
imgN = io.imread("https://img.freepik.com/premium-photo/cute-little-kitten-playing-with-colorful-wool-balls-concept-love-animals-pets-generative-ai_853928-216.jpg")
imgN_g = cv.cvtColor(imgN, cv.COLOR_BGR2GRAY)
```

**(1)模糊**

```
# Example: 5x5 Gaussian kernel for blurring
kernel_blur = np.ones((5, 5), np.float32) / 25
```

- `np.ones((5, 5), np.float32)`：生成一个 5x5 的全 1 数组，这个数组用作卷积核。卷积核是一个用于图像滤波的小矩阵。

- `/ 25`：将卷积核的每个元素都除以 25，生成了一个 5x5 的均值滤波器。25 是 5x5 数组的总元素个数（即 25），这样使得每个元素的值为 `1/25`，即 `0.04`。

- 这个卷积核的作用是对图像中每个 5x5 区域进行平均处理，相当于计算该区域内所有像素的平均值，并用这个均值替代该区域中心的像素值，从而达到模糊效果。

```
blurred_image = cv.filter2D(imgN_g, -1, kernel_blur)
```

`cv.filter2D(imgN_g, -1, kernel_blur)`：使用 OpenCV 的 `filter2D` 函数来对图像 `imgN_g`（灰度图像）进行卷积操作。`filter2D` 函数执行图像卷积，它根据指定的卷积核对图像进行滤波处理。

- `imgN_g`：这是输入的灰度图像。
- `-1`：表示输出图像的深度与输入图像相同（即与 `imgN_g` 的深度相同）。
- `kernel_blur`：这是前面定义的 5x5 模糊卷积核，它会对图像的像素进行均值模糊处理。

**(2)锐化**

```
# Example: 3x3 sharpening kernel
kernel_sharpen = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
sharpened_image = cv.filter2D(imgN_g, -1, kernel_sharpen)
```

这个 3x3 的矩阵 `kernel_sharpen` 是一个 **锐化卷积核**，它用于增强图像的边缘，增加图像的清晰度。

- **中心值为 5**：它放大了中心像素的值，使得图像变得更清晰。
- **周围值为 -1**：它减去周围像素的值，增加了图像的对比度，使得边缘更加突出。
- 整个卷积核通过这种结构增强了图像的边缘和细节，起到了锐化的作用。



#### 2.3.3.5 Vector Operations(向量操作)

向量和矩阵是图像处理中的基本结构，它们以数学形式表示图像数据。向量和矩阵的操作，如加法、乘法和求逆，是执行图像变换、滤波和特征提取等任务的关键。理解这些操作能够有效地处理多维图像数据。

**图像反转**

```
inverted_image = 255 - imgN
```

将图像中的每个像素值进行取反操作，使得图像中的亮区变暗，暗区变亮。具体来说，像素值的范围通常是 0 到 255（灰度图像），0 表示黑色，255 表示白色，中间的值表示灰色。



#### 2.3.3.6 Probabilistic Method & Image Transform(概率方法与图像变换)

图像变换是将图像转换到不同域进行分析的数学技术。常见的变换包括傅里叶变换（用于分析频率成分）和小波变换（用于多分辨率分析）。通过提供图像数据的替代表示，图像变换有助于图像压缩、降噪和特征提取等任务。

```
myImg = io.imread("https://img.freepik.com/premium-photo/cute-little-kitten-playing-with-colorful-wool-balls-concept-love-animals-pets-generative-ai_853928-216.jpg")
```

**(1)直方图**

有时，我们希望增强图像的对比度，或者在特定区域扩大对比度，即使这可能会牺牲那些颜色变化较小或不太重要的细节。直方图是一个很好的工具，能帮助我们发现这些感兴趣的区域。要创建图像数据的直方图，我们可以使用 `matplot.pylab` 的 `hist()` 函数。

```
image = myImg;
plt.hist(image.ravel(),bins = 256, range = [0,256])
plt.show()
```

`plt.hist()`是Matplotlib中用于绘制直方图的函数。

- 直方图用于统计图像中每个像素值的出现频率（即像素值分布），这对图像的对比度、亮度等特性分析很有帮助。

- `ravel()`：将多维数组（例如图像）展平为一维数组。这样做是因为直方图是针对像素值的分布，而图像是一个二维数组，因此需要将图像中的所有像素值转换为一个长的一维列表来进行统计。
- `bins=256`：`bins`表示将像素值分成多少个区间进行统计。对于 8 位灰度图像（像素值在 0 到 255 之间），我们通常选择 256 个区间（每个区间对应一个像素值）。
- `range=[0, 256]`：`range` 指定了直方图统计的范围。对于 8 位图像，像素值的范围是 0 到 255，因此设定为 `[0, 256]`。这个范围确保所有像素值都在直方图中进行统计。

**(2)RGB直方图**

上面的直方图是将RGB三个通道位于某个像素值的个数相加，现在我们单独显示RGB三个通道的直方图。

```
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
```

- `cv.calcHist([image], [i], None, [256], [0, 256])`：是OpenCV 用于计算图像直方图的函数。

  - `[image]`：输入图像，使用列表包裹。

  - `[i]`：指定计算直方图的通道。在循环中，`i` 对应的是图像的第 `i` 个通道，即 `0` 表示蓝色通道，`1` 表示绿色通道，`2` 表示红色通道。

  - `None`：这里不使用掩膜，表示对整个图像计算直方图。

  - `[256]`：直方图的 `bins` 数，即将像素值分成 256 个区间（0 到 255）。

  - `[0, 256]`：像素值的范围，从 0 到 255。

`plt.xlim([0, 256])`：设置 x 轴的范围为 `[0, 256]`，对应像素值的范围（0 到 255）。

**(3)灰度图像直方图**

```
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
plt.hist(gray_image.ravel(),bins = 256, range = [0, 256])
```

**(4)寻找灰度图像的轮廓**

**方法一**：用matplotlib的contour函数

```
plt.contour(gray_image, origin = "image")
```

**方法二：用OpenCV库**

```
# Set threshold for the countour detection
ret, thresh = cv.threshold(gray_image,150,255,0)
```

`cv.threshold()`：二值化图像

- `gray_image`：输入的灰度图像。
- `150`：阈值。大于 150 的像素会被设置为 255，表示前景，小于 150 的像素设置为 0，表示背景。
- `255`：最大像素值，表示白色（前景）。
- `0`：阈值类型，表示二值化操作（固定阈值）。
- `ret`：返回的阈值（此处未使用）。
- `thresh`：二值化后的图像，用于后续轮廓检测。

```
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
```

`cv.findContours()`：这个函数在二值化图像中找到所有轮廓。

- `thresh`：输入的二值化图像。
- `cv.RETR_TREE`：轮廓检索模式，表示提取所有轮廓并建立层级结构（即轮廓嵌套关系）。
- `cv.CHAIN_APPROX_SIMPLE`：轮廓近似方法，压缩水平、垂直和对角线方向的点，只保留拐点，减少轮廓点数量。
- `contours`：这是找到的轮廓列表，每个轮廓是一系列点的集合。
- `hierarchy`：描述轮廓之间层次结构的数组（如果存在嵌套轮廓）。

```
cv.drawContours(image, contours, -1, (0, 255, 0), 3)
plt.imshow(image)
```

`cv.drawContours()`：该函数将检测到的轮廓绘制在原始图像上。

- `image`：原始图像，绘制轮廓后的图像会显示在这里。
- `contours`：找到的轮廓列表。
- `-1`：表示绘制所有轮廓。如果指定其他值，则只绘制特定索引的轮廓。
- `(0, 255, 0)`：轮廓的颜色，(0, 255, 0) 是绿色。
- `3`：轮廓线的粗细。

**(5)灰度变换**
本节提供了一些对灰度图像进行数学变换的示例。

```
im2 = 255 - gray_image
cv2_imshow(im2)
```

这是对灰度图像的**反转操作**，您可以看到明亮的像素变暗，而暗的像素变亮。

```
im3 = (100.0/255)*gray_image + 100
cv2_imshow(im3)
```

另一种图像变换，添加一个常数后，所有像素变得更亮，图像产生类似**雾化**的效果。

- `100.0/255`：这个表达式将灰度图像的像素值按比例缩放。因为像素值的最大范围是 255，将像素值乘以 `100.0/255` 就相当于把像素值缩放到 0 到 100 的范围。这部分操作缩小了原始灰度图像的亮度范围。

```
im4 = 255.0*(gray_image/255.0)**2
cv2_imshow(im4)
```

灰度图像的**亮度**水平在这一步骤之后**降低**。

- `gray_image/255.0`：首先，将灰度图像的像素值从 0 到 255 的范围缩放到 0 到 1 的范围。每个像素值被 255 除后得到的是归一化的灰度值（0 到 1 之间）。

- `(gray_image/255.0)\**2`：对归一化后的像素值进行平方运算。这种平方变换会使较小的值变得更小，而较大的值变化相对较小，结果是图像的亮度会降低。低亮度的像素被压缩得更暗，而高亮度的像素变化不大。

- `255.0\*()`：将平方后的像素值再次扩展回 0 到 255 的范围。这是为了确保结果可以显示为标准的灰度图像。

**(6)直方图均衡化**
本节展示了对一张暗图像进行直方图均衡化的操作。该变换通过平坦化灰度直方图，使所有的亮度值尽可能均匀分布。变换函数是图像中像素值的累积分布函数（cdf），并将像素值范围映射到所需的范围。

```
def histeq(im, nbr_bins = 256):
  """ Histogram equalization of a grayscale image.  """
  # get the image histogram
  imhist, bins = np.histogram(im.flatten(), nbr_bins, [0, 256])
  cdf = imhist.cumsum() # cumulative distribution function
  cdf = imhist.max()*cdf/cdf.max()  #normalize
  cdf_mask = np.ma.masked_equal(cdf, 0)
  cdf_mask = (cdf_mask - cdf_mask.min())*255/(cdf_mask.max()-cdf_mask.min())
  cdf = np.ma.filled(cdf_mask,0).astype('uint8')
  return cdf[im.astype('uint8')]
```

- 使用 `np.histogram()` 计算输入图像的直方图，将图像像素值展平为一维数组，然后统计每个像素值在 [0, 256] 范围内的频率。
- `imhist.cumsum()` 计算累积分布函数（CDF），这会帮助我们重新分配图像的像素值。
  - 例如，如果某个像素值的频率是 10，累加之后，它的 CDF 就表示该像素值及以下的像素总数。
- `cdf = imhist.max() * cdf / cdf.max()`将 CDF 进行归一化，以便调整像素值的分布。
- 使用 `np.ma.masked_equal(cdf, 0)` 忽略 CDF 中的零值（避免除零错误）。
- `cdf_mask = (cdf_mask - cdf_mask.min()) * 255 / (cdf_mask.max() - cdf_mask.min())`：对非零的 CDF 进行归一化，将像素值范围映射到 [0, 255]。
- `cdf = np.ma.filled(cdf_mask, 0).astype('uint8')`：将处理后的 CDF 填充回原来的数组，并转换为 `uint8` 类型。

**(7)灰度图像的傅里叶变换**

傅里叶变换用于找到图像的频域。你可以将图像视为在两个方向上采样的信号。因此，对图像在 X 和 Y 方向上进行傅里叶变换可以得到图像的频率表示。对于正弦信号，如果幅度在短时间内变化得很快，可以称之为高频信号；如果变化缓慢，则称为低频信号。图像中的边缘和噪声是高频内容，因为它们在图像中变化剧烈。

```
# Blur the grayscale image by a Guassian filter with kernel size of 10
imBlur = cv.blur(gray_image,(5,5))
# Transform the image to frequency domain
f = np.fft.fft2(imBlur)
# Bring the zero-frequency component to the center
fshift = np.fft.fftshift(f)
magnitude_spectrum = 30*np.log(np.abs(fshift))
```

- `cv.blur()`：使用一个 5x5 的均值滤波器对灰度图像进行模糊处理，降低图像的细节和噪声。此操作可以平滑图像。
- `np.fft.fft2()`：对图像 `imBlur` 进行 **二维傅里叶变换**，将图像从空间域转换到频率域。

- `np.fft.fftshift()`：将傅里叶变换后的结果进行 **零频率分量移位**，即将低频分量（通常位于图像四角）移动到频谱的中心。
  - 在频率分析中，我们通常更关注图像的低频部分（整体亮度、背景等），而高频部分通常与图像的细节或噪声有关。将低频移动到中心后，我们可以更直观地观察图像中的主要频率成分，便于进一步分析和处理。

- `np.abs(fshift)`：计算傅里叶变换结果的幅度，即频率成分的强度。

- `np.log()`：取对数变换，压缩数值范围，使得显示效果更明显（因为原始幅度差异较大，直接显示会使大部分细节看不见）。

**(8)通过傅里叶高通滤波寻找边缘**
下面代码通过在频率域中进行**高通滤波**，去除图像中的低频成分，保留高频信息，最终得到一张**锐化图像**

```
# 获取中心坐标
rows, cols = imBlur.shape
crow,ccol = round(rows/2) , round(cols/2)
# remove low frequencies with a rectangle size of 10
fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
```

- `crow,ccol`：分别计算图像的中心行和列（即频谱中心点）。由于傅里叶变换后的低频成分位于中心，我们需要确定中心的位置，以便后续操作。

- `fshift[crow-10:crow+10, ccol-10:ccol+10] = 0`：在频谱的中心位置创建一个大小为 10x10 的矩形区域，并将其置为 0，这一步是**移除低频成分**，因为低频成分代表图像的大范围平滑变化。
  - 去除低频部分后，只保留高频成分（即快速变化的细节，如边缘、噪声），从而突出图像的边缘信息。

- `np.fft.ifftshift()`：将频谱重新移回原始位置，因为傅里叶逆变换要求低频成分在频谱的四个角上，而 `fftshift()` 将低频移到了中心，所以在逆变换前需要将其移回。

- `np.fft.ifft2()`：对频谱进行**二维傅里叶逆变换**，将频率域的数据转换回**空间域**，得到经过高通滤波后的图像。
  - 此时，图像会突出高频信息，主要表现为图像中的边缘和细节部分。

- `np.abs()`：由于傅里叶逆变换的结果可能是复数，所以通过取绝对值将其转换为实数图像。

