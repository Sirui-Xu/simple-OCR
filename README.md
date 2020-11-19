## Experiments

- lib：文件夹，内含有几个实现的重要的代码工具
  - connect.py：通过连通域算法对文字进行分割。
	
	`python3 connect.py ../images/{image_name}`
  - houghTransform.py：通过霍夫变换对文字进行粗检测。
	
	`python3 houghTransform.py ../images/{image_name}`
  - mser.py：通过最大稳定极值区域算法对文字进行检测。
	
	`python3 mser.py ../images/{image_name}`
- create_data.py：（在mac系统中使用，否则要修改字体设置）随机选择系统中的字体，随机化大小、颜色等设置，生成所有英文字母图像，作为识别的数据集。
	
	`python3 create_data.py`
- create_image.py：（在mac系统中使用，否则要修改字体设置）随机选择系统中的字体，随机化大小、颜色等设置，输入一段文字，生成带文字的图像。
	
	`python3 create_image.py`
- pipeline.py：完整的pipeline代码，运行代码可以框出所有的字母图像，并且在终端输出所有对应的识别出来的字母 
	
	`python3 pipeline.py ./images/{image_name}`
- demo.py：完整的pipeline使用可视化交互，当鼠标移动到图像上可能出现的字母位置，会自动框出该字母，并且在图像上显示识别出的结果。
- results：文件夹，内含利用上述代码生成的检测图片。
- data：文件夹，内含生成的所有字母
- images：文件夹，包括自己生成带一段文字的图片，也包括截图。
- demo.mp4：测试视频。

## Requirements

python(>=3.7)

numpy

opencv

PIL

Matplotlib

sklearn

Skimage








