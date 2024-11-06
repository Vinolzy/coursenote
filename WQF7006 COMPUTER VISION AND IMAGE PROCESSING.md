

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



# 3 IMAGE SEGMENTATION(Color Image Processing)

## 3.1 Color Image Processing (彩色图像处理)

**Electromagnetic Spectrum (电磁波谱)**

- 可见光的波长范围为 **400-700 纳米**，这决定了人眼能够感知的光的颜色。波长越短，光的颜色越接近蓝色，波长越长，颜色越接近红色。
- **Radiance (辐射度)**：光源发出的总能量，用瓦特 (W) 表示。
- **Luminance (亮度)**：观察者接收到的光能量，用流明 (lm) 表示。亮度与视觉系统感知到的亮光有关。
- **Brightness (亮度)**：光的主观强度感知，与物理光强有关。

**Sensitivity of Human Eye Cones (人眼视锥细胞的敏感度)**

- 人眼中有三种视锥细胞，分别对 **红光** (65%)、**绿光** (33%) 和 **蓝光** (2%) 具有敏感性，这就是为什么RGB是常用的颜色模型。

**Color Models (颜色模型)**

1. **RGB Model (RGB 模型)**
   - **RGB** 颜色模型基于三原色：**红** (Red)、**绿** (Green)、**蓝** (Blue)。每个颜色通道的值在0到255之间，组成一个颜色立方体。
   - 主要用于 **显示设备**，如电视和电脑屏幕。
   - **扩展知识**: **颜色深度 (Color Depth)**，RGB模型常用的24位颜色深度意味着可以表示 **16,777,216 种颜色**，即 256^3 种组合。
2. **CMY and CMYK Model (CMY和CMYK模型)**
   - **CMY** 模型是 **减色模型**，使用 **青** (Cyan)、**洋红** (Magenta)、**黄** (Yellow) 来表示颜色。它主要用于 **印刷行业**。
   - **CMYK** 增加了黑色 (K, Key) 分量，因为实际打印过程中混合CMY无法得到纯黑色。
   - **扩展知识**: **减色法与加色法**：减色法用于印刷，颜色越多越接近黑色；加色法用于显示，颜色越多越接近白色。
3. **HSI Model (HSI 模型)**
   - **HSI** 模型是 **色调** (Hue)、**饱和度** (Saturation)、**亮度** (Intensity) 的组合，更符合人类的视觉直观感受。
   - **Hue (色调)**：代表颜色的主波长，是颜色的基础，如红色、黄色等。
   - **Saturation (饱和度)**：代表颜色的纯度，饱和度越高，颜色越鲜艳。
   - **Intensity (亮度)**：代表颜色的亮度或明暗程度。

## 3.2 Pseudocolor and Full-color Image Processing (伪彩色和全彩色图像处理)

**Pseudocolor Image Processing (伪彩色图像处理)**

- 将颜色分配给灰度图像。主要目的是增强人类对不同灰度级的感知。例如，在医学图像中，通过将不同的颜色赋予不同的灰度值，可以更好地区分不同的结构。
- **扩展知识**: **伪彩色 (Pseudocolor)** 是一种视觉增强技术，不一定反映真实颜色，但可以帮助观察者更清楚地辨别不同区域。

**Full-color Image Processing (全彩色图像处理)**

- 处理彩色图像中的真实颜色，通常处理每个颜色分量（如RGB的每个通道）。也可以通过 **向量处理** 来同时处理多个分量。

## 3.3 Image Segmentation (图像分割)

**Segmentation in HSI Color Space (HSI颜色空间中的分割)**

- **HSI空间**：在图像分割中，通常使用 **色调** (Hue) 和 **饱和度** (Saturation) 进行分割，而不常使用 **亮度** (Intensity)。因为色调和饱和度更好地表达颜色信息。
- **扩展知识**: **阈值分割法**：这是最简单的图像分割技术之一，基于颜色分量设定一个阈值，大于阈值的像素归为一个类别，小于阈值的像素归为另一个类别。

**Segmentation in RGB Vector Space (RGB向量空间中的分割)**

- **RGB向量空间**：图像中的每个像素点在RGB空间中都有一个向量坐标，通过 **欧氏距离** 计算各像素与目标颜色的差异，可以用于颜色分割。
- **扩展知识**: **欧氏距离 (Euclidean Distance)** 是计算空间中两点之间距离的标准度量。在RGB颜色空间中，用它来衡量两个颜色之间的差异。



## 3.4 Advanced Color Transformations (高级颜色变换)

**Color Slicing (颜色切片)**

- **颜色切片**：通过选择颜色空间中的一个特定颜色范围（切片），将与该颜色匹配的像素保留，其余部分转换为其他颜色（如灰色）。这种技术常用于高亮显示图像中的某些特定颜色。
- **扩展知识**: **颜色空间 (Color Space)** 是一种表示颜色的数学模型，不同的颜色模型有不同的颜色空间（如RGB、HSI、CIELAB等）。

**Tonal and Contrast Corrections (色调和对比度调整)**

- **色调调整**：主要针对颜色亮度的调整。**对比度调整** 则通过拉伸或压缩亮度值的范围来增强图像中的细节。
- **扩展知识**: **直方图均衡化 (Histogram Equalization)** 是一种常见的对比度增强技术，适用于灰度图像和彩色图像。

**Histogram Equalization (直方图均衡化)**

- 对图像的亮度分布进行调整，常用于增强图像的细节部分。对于彩色图像，通常在HSI模型中只对 **亮度分量** 进行均衡化。

## 3.5 Color Segmentation Examples (颜色分割示例)

**HSI-based Segmentation (基于HSI的分割)**

- 使用HSI颜色空间的 **色调和饱和度分量**，通过阈值操作分割图像中的某些颜色区域。
- **扩展知识**: **二值化 (Binarization)**：一种常用于分割的技术，通过将像素值转换为两类（如黑白），从而突出目标物体。

**RGB-based Segmentation (基于RGB的分割)**

- 使用 **颜色距离** 在RGB空间中进行分割，通常通过 **阈值** 控制，分割出与目标颜色最接近的像素。



## 3.6 HSV,HSI,HSL色彩空间模型区别

1. **HSV（Hue, Saturation, Value）色彩空间**

- **H（色调）**：表示颜色的种类，范围是 0 到 360 度。0 度代表红色，120 度代表绿色，240 度代表蓝色。
- **S（饱和度）**：表示颜色的纯度，范围是 0 到 1。0 代表灰色，1 代表纯色。
- **V（亮度）**：表示颜色的明亮程度，范围是 0 到 1。0 是黑色，1 是最亮的颜色。

**HSV 的应用**：常用于计算机图形设计和图像处理，因为它与人的色彩感知更接近，可以方便地调整颜色的亮度或饱和度。

2. **HSI（Hue, Saturation, Intensity）色彩空间**

- **H（色调）**：与 HSV 中的色调含义相同，表示颜色的种类，范围是 0 到 360 度。
- **S（饱和度）**：表示颜色的纯度，范围是 0 到 1。0 代表灰色，1 代表纯色。
- **I（强度）**：表示颜色的整体亮度，范围是 0 到 1。与 HSV 的亮度不同，HSI 的强度是 RGB 三个分量的平均值，代表颜色的总能量。

**HSI 的应用**：适合用于图像处理中的灰度图像转换以及色彩分析，因为它更好地分离了色调与强度。

3. **HSL（Hue, Saturation, Lightness）色彩空间**

- **H（色调）**：与 HSV 和 HSI 中的色调含义相同，范围是 0 到 360 度。
- **S（饱和度）**：表示颜色的纯度，范围是 0 到 1。
- **L（亮度）**：表示颜色的明暗程度，范围是 0 到 1。0 是黑色，1 是白色。L 是通过颜色中的最亮和最暗分量计算出来的。

**HSL 的应用**：HSL 常用于设计工具，尤其是网页设计，因其可以直观地调节亮度和饱和度。

**区别总结：**

- **HSV**：侧重于颜色的亮度调整，适合在设计中通过调整 V（亮度）来改变颜色的明暗。
- **HSI**：强调颜色的总亮度，适合用于图像处理，特别是在灰度化和颜色信息提取中。
- **HSL**：通过 L（亮度）表现颜色的明暗，L 值比 HSV 的 V 更容易理解，适合用于设计工作中的颜色调配。



## 3.7 作业代码分析

### 3.7.1 HSV-1

```
# to run in google colab
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

if "google.colab" in sys.modules:

    def download_from_web(url):
        import requests

        response = requests.get(url)
        if response.status_code == 200:
            with open(url.split("/")[-1], "wb") as file:
                file.write(response.content)
        else:
            raise Exception(
                f"Failed to download the image. Status code: {response.status_code}"
            )

    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_02a_basic_image_processing/grass.jpg"
    )
    download_from_web(
        "https://github.com/YoniChechik/AI_is_Math/raw/master/c_02a_basic_image_processing/hsv_th.png"
    )

figsize = (10, 10)
```

**检查是否在 Google Colab 中运行**：

- `if "google.colab" in sys.modules` 这行代码检查当前 Python 环境是否在 Google Colab 中。如果是，它会运行内部定义的功能。

**定义下载函数**：

- `download_from_web(url)` 函数用于从网络上下载图片。它使用 `requests` 库从指定的 URL 获取内容，并将其保存为本地文件。
- 如果请求成功 (`status_code == 200`)，它将下载的内容写入文件。文件名由 URL 中的最后一部分确定（即 URL 中最后一个 `/` 之后的部分）。
- 如果下载失败（即状态码不为 200），它会抛出一个异常并显示状态码。

**下载图像**：

- 调用了 `download_from_web` 函数，下载了两张图像文件：`grass.jpg` 和 `hsv_th.png`

**设置图像大小**：

- `figsize = (10, 10)` 这一行设置了绘制图像时的图像大小，单位为英寸（图像的宽度和高度均为 10 英寸）。

----

接下来显示图片：

```
bgr_im = cv2.imread("grass.jpg")
rgb_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2RGB)
plt.figure(figsize=figsize)
plt.imshow(rgb_im)
plt.title("original image")
plt.show()
```

-----------

我们想要将草地和天空分离。我们将通过遮罩掉图像中所有非绿色的像素来实现这一点。首先，找到 HSV 格式的绿色。

```
rgb_green = np.uint8([[[0, 255, 0]]])  # 这是一个三维数组，因为 cvtColor 需要这样的格式…
hsv_green = cv2.cvtColor(rgb_green, cv2.COLOR_RGB2HSV)[0, 0, :]
print(hsv_green)
```

- 代码使用 OpenCV 函数 `cvtColor` 将纯绿色的 RGB 值 `[0, 255, 0]` 转换为 HSV（色调-饱和度-亮度）格式。`cvtColor` 期望输入是一个 3D 数组，因此构建了一个包含 RGB 值的 3D 数组。转换后的 HSV 颜色会输出为 `[hue, saturation, value]`，即绿色的 HSV 值。

--------------

接下来，将图像转换为 HSV 格式并仅针对绿色及其相邻色进行阈值处理。

我们将使用色调 (hue) 阈值范围为 +30 和 -70（因为它远离蓝色-天空）。我们将使用绿色的所有饱和度 (saturation) 和亮度 (value) 变体。在该阈值范围内遮罩所有像素，应该只剩下草地部分。

```
# Convert BGR to HSV
hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)

# define range of hue and intensity
lower_th = hsv_green - np.array([70, 200, 200])
upper_th = hsv_green + np.array([30, 0, 0])

# Threshold the HSV image
mask = cv2.inRange(hsv_im, lower_th, upper_th)

plt.figure(figsize=figsize)
plt.imshow(mask)
plt.title("resulted mask")
plt.show()
```

- 首先将输入图像 `rgb_im` 从 RGB 颜色空间转换为 HSV（色调-饱和度-亮度）颜色空间。这种转换的目的是为了更好地处理颜色，因为在 HSV 空间中，色调 (hue) 直接表示颜色，饱和度 (saturation) 和亮度 (value) 表示颜色的强度和光度，适合颜色过滤和分割。
- `lower_th` 和 `upper_th` 分别定义了 HSV 图像的下限和上限，用于进行绿色的阈值处理。
  - `lower_th = hsv_green - np.array([70, 200, 200])`：减去的值设定了色调、饱和度和亮度的下限。`70` 是色调 (hue) 的范围，允许一定的偏差以捕捉不同强度的绿色；`200` 的饱和度和亮度较高，以确保绿色部分不会太暗。
  - `upper_th = hsv_green + np.array([30, 0, 0])`：加的值设定了绿色的上限，色调的范围是 `+30`，确保捕捉到绿色的变体，同时允许其他参数保持。

- 通过 `cv2.inRange` 函数对图像进行阈值处理。`inRange` 会生成一个二值遮罩图像，在指定的阈值范围内的像素值被标记为白色 (255)，而不在范围内的像素值标记为黑色 (0)。换句话说，所有在绿色范围内的像素会保留（作为白色区域），其他像素将被滤除（黑色区域）。

位与操作，这样最终的结果 `rgb_res` 只保留了绿色区域的像素值，而其他部分则被设置为黑色

```
# Trick: apply 2d mask on 3d image
rgb_res = cv2.bitwise_and(rgb_im, rgb_im, mask=mask)

plt.figure(figsize=figsize)
plt.imshow(rgb_res)
plt.title("output image")
plt.show()
```



### 3.7.2 HSV-2

```
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from skimage.filters import threshold_otsu
```

- `matplotlib.pyplot` 用于图像或数据的可视化；
- `numpy` 用于处理数据数组；
- `skimage.io` 提供图像的输入输出操作；
- `skimage.color` 用于图像的颜色空间转换；
- `skimage.exposure` 用于处理图像的像素值分布；
- `skimage.filters` 提供了图像的过滤和阈值处理工具。

--------------

读取图像，并转换为灰度图像

```
url = "https://thumbs.dreamstime.com/b/yellow-red-balloons-10992473.jpg"
sail = imread(url)
imshow(sail)
sail_gray = rgb2gray(sail)
imshow(sail_gray)
```

----

接下来定义了一个阈值 `th`，其值为 `0.4`。这个阈值用于将灰度图像进行二值化处理。灰度图像中的每个像素值都是介于 `0`（黑色）到 `1`（白色）之间的小数值，因此这里的阈值 `0.4` 表示我们要将亮度小于 `0.4` 的像素作为“黑色”，而亮度大于等于 `0.4` 的像素作为“白色”。

```
th = 0.4
sail_gray_bw = sail_gray<th
imshow(sail_gray_bw)
```

-------------

接下来将一张彩色的 RGB 图像转换为 HSV 颜色空间，然后在一个图形上以灰度图的形式分别显示图像的三个通道（色调、饱和度和亮度）。通过这样分解图像的颜色信息，可以清晰地看到图像中颜色的分布、强度和亮度。

```
sail_hsv = rgb2hsv(sail)
# 使用 plt.subplots 创建一个包含 1 行 3 列的子图 (ax) 布局。宽 12 英寸，高 4 英寸。
fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].imshow(sail_hsv[:,:,0], cmap='gray')
ax[0].set_title('Hue')
ax[1].imshow(sail_hsv[:,:,1], cmap='gray')
ax[1].set_title('Saturation')
ax[2].imshow(sail_hsv[:,:,2], cmap='gray')
ax[2].set_title('Value');
```

- 使用 `rgb2hsv(sail)` 将 RGB 图像 `sail` 转换为 HSV 颜色空间。
- `sail_hsv[:, :, 0]` 选择 HSV 图像的第一个通道，即色调（Hue）。这会显示图像中每个像素的颜色类型（色调）。
  - `cmap='gray'` 将色调以灰度图像的形式显示，以便更直观地查看各个像素的色调分布。

- S(饱和度)和V(亮度)同H(色调)

---

接下来显示色调（Hue）通道并使用 HSV 颜色映射：

```
plt.imshow(sail_hsv[:, :, 0], cmap='hsv')
plt.colorbar()
```

- `sail_hsv[:, :, 0]` 表示 HSV 图像的第一个通道，即 **Hue（色调）**，这是用来表示图像中颜色类型的通道，取值范围一般在 0 到 1 之间。

- `cmap='hsv'` 指定了使用 **HSV 颜色映射**，而不是常规的灰度图。这样可以更直观地表示色调通道中的颜色信息：每个像素的数值将被映射到对应的颜色（根据 HSV 色彩模型）。

- `plt.colorbar()` 添加一个**颜色条**，显示色调通道的数值范围和相应的颜色。这对理解图像中的颜色值非常有帮助。
  - 颜色条会显示 HSV 色彩空间中的颜色如何映射到不同的色调值。色条上，通常从 0（对应红色）到 1（再回到红色），中间包含黄色、绿色、青色、蓝色和紫色等颜色。

------------

下面代码首先创建一个mask，用于筛选 HSV 色彩空间中色调在 **0 到 0.1** 之间的像素（接近红色）。然后通过这个遮罩，只保留了原图中这些色调范围的区域，并将其余部分遮罩为黑色。最终，代码显示了一个仅包含接近红色的区域的图像，其他区域变为黑色。

- 后面的代码选取色调为黄色/黑色的代码类似，不再重复解释

```
lower_mask = sail_hsv[:, :, 0] > 0
upper_mask = sail_hsv[:, :, 0] < 0.1
mask = upper_mask * lower_mask
plt.imshow(mask)
red = sail[:, :, 0] * mask
green = sail[:, :, 1] * mask
blue = sail[:, :, 2] * mask
sail_masked = np.dstack((red, green, blue))
imshow(sail_masked)
```

- **lower_mask**：选取色调大于 0 的像素。

- **upper_mask**：选取色调小于 0.1 的像素。
- **mask**：只有在某个像素同时满足 `upper_mask` 和 `lower_mask` 的条件时（即色调值在 0 到 0.1 之间），该像素的值才为 `True`。否则，值为 `False`。
- **red, green, blue**：应用遮罩到 RGB 通道，非遮罩区域变为 0。
- **sail_masked**：组合 RGB 通道，显示遮罩后的图像，只有接近红色的部分保留，其余为黑色。
  - `np.dstack((red, green, blue))` 使用 `dstack` 将应用遮罩后的 `red`、`green` 和 `blue` 通道沿深度维度组合成一个新的 3D RGB 图像。


# 4 Image Classification(Machine Learning Overview)

- **机器学习**使计算机无需显式编程即可学习，使用大量数据完成如 OCR 和物体检测的任务。

- 它使计算机能够处理难以手动编程的复杂任务，但受限于数据和计算能力。
- 在有足够的数据和计算资源时，机器几乎可以学习任何任务。
- **模型**是机器学习的核心，用于根据已有示例预测新数据。
- 机器人面临动态环境，手动编程不可行，机器学习帮助它们适应环境，类似于人类学习。

<font color = blue>**类型**</font>

**监督学习**：

- 使用带标签的训练数据，目标是正确标记新数据。
- 是最常见的机器学习类型。例如：将电子邮件分类为垃圾邮件或非垃圾邮件，模型通过带标签的数据进行训练以预测新类别。

**强化学习**：

- 使用无标签数据，系统根据动作获得反馈。
- 常用于机器人领域，如教机器人爬楼梯。机器人通过每次成功的动作获得“奖励”，以学习哪些动作有益。

**无监督学习**：

- 使用无标签数据，旨在发现数据中的模式。
- 例如：根据特征将花朵进行分组，而不需要知道它们的种类。

<font color=blue>**应用**</font>

- **手写识别**：将书写的字符转换为数字文本。
- **语言翻译**：翻译口语或书面语言（如 Google Translate）。
- **语音识别**：将语音片段转换为文本（如 Siri、Cortana 和 Alexa）。
- **图像分类**：为图像加上合适的标签（如 Google Photos）。
- **自动驾驶**：让汽车能够自主驾驶。

<font color=blue>**特征**</font>

- **特征是用于形成预测的观测值**。例如，在图像分类中，像素是特征；在语音识别中，声音样本的音高和音量是特征；在自动驾驶中，摄像头、测距传感器和 GPS 提供的数据是特征。
- **提取相关特征对构建模型很重要**。例如，时间对图像分类无关紧要，但在垃圾邮件分类中很重要，因为垃圾邮件通常在晚上出现。
- **机器人中的常见特征类型**包括像素（RGB 数据）、深度数据（声呐、激光测距仪）、运动（编码器值）和方向或加速度（如陀螺仪、加速度计、指南针）。

<font color=blue>**衡量分类的成功**</font>

真正例和真反例被认为是“正确的”，而假正例和假反例被认为是“错误的”。

- **真正例(True Positive)**：正确地识别为相关项。
- **真反例(True Negative)**：正确地识别为不相关项。
- **假正例(False Positive)**：错误地标记为相关项。
- **假反例(False Negative)**：错误地标记为不相关项。

<font color=blue>**Precision, Recall, Accuracy**</font>

**精确率（Precision）**：正确的正类标签的比例。

- 精确率 = (真正例数量) / (真正例数量 + 假正例数量)
- 模型预测的猫中，实际是猫的比例

**召回率（Recall）**：正确标记的正类示例的比例。

- 召回率 = (真正例数量) / (真正例数量 + 假反例数量)
- 所有实际的猫中，有多少被正确预测为猫

**准确率（Accuracy）**：正确标签的比例。

- 准确率 = (真正例数量 + 真反例数量) / (样本总数)
- 在所有图片中，有多少被正确识别

<font color=blue>**训练数据和测试数据**</font>

- **训练数据**：用于训练模型的数据。
- **测试数据**：用于评估模型准确性的数据。
- **过拟合**：模型在训练数据上表现良好，但在测试数据上效果不佳。
  - 过拟合问题可以通过正则化和交叉验证来避免。

<font color=blue>**偏差与方差**</font>

- **偏差(Bias)**：模型预测与真实值的预期差异。
- **方差(Variance)**：模型在不同训练集之间的预测差异。
- **高偏差(High Bias)**：模型在训练数据上的预测不准确。
- **高方差(High Variance)**：模型难以泛化到新数据集。

**关系**

- 偏差和方差通常是反向相关的。
- 一个高偏差低方差的例子是模型总是预测为“非猫”。
- 一个低偏差高方差的例子是模型完美匹配训练集中的“猫”，但对新数据表现不佳。

<font color=blue>**监督学习算法**</font>

- **线性回归(Linear Regression)**：用于预测连续值，例如预测纸飞机的飞行距离。
- **决策树(Decision Trees)**：一种流程图，其中每个节点是一个条件语句，通常是二叉结构。
- **支持向量机(Support Vector Machines)**：用于分类，通过找到分隔数据的超平面。

- **K最近邻算法(K-Nearest Neighbor)**：根据输入值的K个最近邻样本预测输出。
- **神经网络(Neural Networks)**：模拟大脑以学习复杂的非线性关系。

----

PPT中基于Machine Learning的相关原理在7015中有讲解，这里不再赘述。

# 5 OBJECT DETECTION AND RECOGNITION

具体内容略，参考7015的CNN内容。

## 5.1 CNN代码解析

# 3. Convolutional Neural Networks

在上一部分中，我们构建并训练了一个简单的模型来分类ASL（美式手语）图像。虽然模型在训练数据集上表现出了较高的准确率，但在验证数据集上的表现并不理想。这种无法很好地推广到非训练数据的现象称为**过拟合**。在本节中，我们将引入一种特别适合处理图像和分类任务的模型——卷积神经网络（CNN）。

- 针对CNN对数据进行预处理
- 创建一个更复杂的CNN模型，了解更多的模型层
- 训练CNN模型并观察其性能

```python
import torch.nn as nn
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# 检测是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()
```

<font color=blue>**Loading and Preparing the Data**</font>

首先，让我们像前一节一样加载数据框架：

```
from google.colab import drive
drive.mount('/content/drive')

# 读取训练和验证数据
train_df = pd.read_csv("/content/drive/MyDrive/Dataset_Assignment_3/sign_mnist_train.csv")
valid_df = pd.read_csv("/content/drive/MyDrive/Dataset_Assignment_3/sign_mnist_valid.csv")
```

ASL数据已展平成一维格式。

```
sample_df = train_df.head().copy()  # 获取前5行数据
sample_df.pop('label')  # 移除标签列
sample_x = sample_df.values  # 获取图像数据
sample_x
```

在这种格式下，我们无法获得像素之间的邻接关系信息，因此无法应用卷积来检测特征。让我们将数据集重新调整为28x28像素格式，以便卷积层能够关联像素组并检测重要特征。

注意，对于模型的第一卷积层，我们需要提供图像的高度和宽度，还需要颜色通道数。我们的图像是灰度图像，因此通道数为1。

这意味着我们需要将当前形状从 (5, 784) 转换为 (5, 1, 28, 28)。使用NumPy数组时，可以通过传入 -1 来保持某个维度不变。

```
IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1

sample_x = sample_x.reshape(-1, IMG_CHS, IMG_HEIGHT, IMG_WIDTH)
sample_x.shape
```

**Create a Dataset**

接下来，我们将上面的步骤添加到自定义的数据集类 `MyDataset` 中。

```
class MyDataset(Dataset):
    def __init__(self, base_df):
        x_df = base_df.copy()  # 部分操作将直接在原数据上进行
        y_df = x_df.pop('label')
        x_df = x_df.values / 255  # 归一化数据，将值从0到255映射到0到1之间
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)
```

**Create a DataLoader**

接下来，让我们从数据集中创建 `DataLoader`。

```
BATCH_SIZE = 32

train_data = MyDataset(train_df)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_N = len(train_loader.dataset)

valid_data = MyDataset(valid_df)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE)
valid_N = len(valid_loader.dataset)
```

**检验 DataLoader**

让我们从 `DataLoader` 中获取一个批次，以确保它正常工作。

```
batch = next(iter(train_loader))
```

**Creating a Convolutional Model**

如今，许多数据科学家在开始项目时会借鉴类似项目的模型结构。假设问题并不完全独特，那么很有可能已经有人创建了效果不错的模型，并发布在像TensorFlow Hub和NGC Catalog这样的在线资源库中。今天我们将提供一个适合该问题的模型。

- Input -> Convolution -> Max Pooling -> Convolution -> Dropout -> Max Pooling -> Convolution -> Max Pooling -> Dense -> Dense -> Output

我们在课程中介绍了不同类型的层，这里将复习它们，并提供相关的文档链接。有疑问时，请参考官方文档。

```python
n_classes = 24
kernel_size = 3
flattened_img_size = 75 * 3 * 3

model = nn.Sequential(
    # 第一卷积层
    nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),  # 25 x 28 x 28
    nn.BatchNorm2d(25),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 25 x 14 x 14
    # 第二卷积层
    nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 x 14 x 14
    nn.BatchNorm2d(50),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.MaxPool2d(2, stride=2),  # 50 x 7 x 7
    # 第三卷积层
    nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 x 7 x 7
    nn.BatchNorm2d(75),
    nn.ReLU(),
    nn.MaxPool2d(2, stride=2),  # 75 x 3 x 3
    # Flatten至全连接层
    nn.Flatten(),
    nn.Linear(flattened_img_size, 512),
    nn.Dropout(.3),
    nn.ReLU(),
    nn.Linear(512, n_classes)
)
```

**Conv2D**

这些是二维卷积层。小的卷积核会在输入图像上滑动，检测分类所需的重要特征。模型中的早期卷积层会检测简单的特征，如边缘；后续的卷积层则会识别更复杂的特征。

<img src="https://github.com/Vinolzy/coursenote/blob/main/resource/conv2D.png?raw=true" alt="con2d">


我们来看看第一个 `Conv2D` 层的结构：

```
nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1)
```

- `25` 表示要学习的滤波器（过滤器）数量。
- `kernel_size=3`，PyTorch 会自动假设我们想要 `3 x 3` 的滤波器。
- `stride=1` 表示滤波器在图像上滑动时的步长。
- `padding=1` 决定输出图像是否与输入图像的尺寸一致。

**BatchNormalization**

与输入归一化类似，批量归一化对隐藏层中的值进行缩放，以改善训练效果。更多细节可以在文档中查看。关于批量归一化层的最佳位置有不同的观点。

**MaxPool2D**

最大池化将图像“缩小”为较低分辨率。这样做有助于模型在物体发生平移（左右移动）时保持鲁棒性，并且加快模型的处理速度。

- 比如4x4缩放为2x2，取四个里的最高像素

**Dropout**

Dropout 是一种防止过拟合的技术。Dropout 随机选择一部分神经元，将其“关闭”，使它们在当前的前向传播或反向传播中不参与计算。这有助于确保网络的鲁棒性和冗余性，不会过分依赖某个特定区域来得出结论。

**Flatten**

Flatten 将一层的多维输出展平为一维数组。展平后的输出被称为特征向量，将作为输入连接到最终的分类层。

**Linear**

在之前的模型中，我们已经见过线性全连接层。第一个全连接层（512个单元）接收特征向量作为输入，学习哪些特征有助于特定分类。第二个全连接层（24个单元）是最终的分类层，输出模型的预测结果。

<font color=blue>**Summarizing the Model**</font>

虽然信息量很大，但不用担心，现在并不需要完全理解所有细节才能有效地训练卷积模型。最重要的是我们知道卷积神经网络可以帮助从图像中提取有用的信息，并且能够用于分类任务。

```python
model = torch.compile(model.to(device))
```

由于我们要解决的问题（ASL图像分类）没有变化，因此我们将继续使用相同的损失函数和准确率度量方式。

```
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N
```

**Training the Model**

尽管模型的架构有了很大的不同，但训练的方式基本保持不变。

```
def validate():
    loss = 0
    accuracy = 0
	# 启用模型的评估模式，通常在验证或测试模型时调用
    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

def train():
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))
```

**训练循环**

设定训练的轮数为20，每一轮都会运行 `train()` 和 `validate()` 函数，打印出每一轮的损失和准确率。

```
epochs = 20

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train()
    validate()
```

<font color=blue>**Summary**</font>

在本节中，我们使用了多种新的层来实现卷积神经网络（CNN），相较于上一节中使用的简单模型，该模型表现更好。希望通过本节内容，创建和训练一个模型的整体流程会变得更加熟悉。

**Clear the Memory**

在继续进行下一步之前，请执行以下代码单元来关闭 IPython 内核，从而释放 GPU 资源

```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```



