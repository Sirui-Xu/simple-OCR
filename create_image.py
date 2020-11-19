import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from tqdm import tqdm
from create_data import getColor
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
# mac 存放的字体文件夹 
font_dir = '/System/Library/Fonts/'
fonts = ['SFNSDisplayCondensed-Black.otf', 'SFNSDisplayCondensed-Regular.otf', 'SFNSTextCondensed-Heavy.otf',
         'Geneva.dfont', 'SFCompactDisplay-Medium.otf', 'Palatino.ttc', 'SFCompactText-BoldItalic.otf', 'Keyboard.ttf',
         'SFNSTextCondensed-Bold.otf', 'SFCompactDisplay-Regular.otf', 'Optima.ttc', 'SFNSTextCondensed-Medium.otf',
         'SFCompactRounded-Medium.otf', 'Avenir.ttc', 'PingFang.ttc', 'SFCompactText-LightItalic.otf', 
         'SFCompactRounded-Regular.otf']

width = 500
height = 500
if __name__ == "__main__":
# 输入一句话
    sentence = input('Please input a sentence in English.\n')
    name = input('Please input your image name.\n')
    color = getColor()
    color_inv = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))
    if random.randint(0, 1) == 1:
        color_inv = (255 - color_inv[0], 255 - color_inv[1], 255 - color_inv[2])
# 生成图片的底
    image = Image.new('RGB', (width, height), color_inv)
# 创建Draw对象:
    draw = ImageDraw.Draw(image)
# 确定字体类型，需要加载对应的ttf文件
    font_id = random.randrange(0, len(fonts))
    font_path = os.path.join(font_dir, fonts[font_id])
    # print(fonts[font_id])
    font = ImageFont.truetype(font_path, random.randint(50,80))

# 输出文字 按照不同的起始位置 不同的字体大小 不同的颜色 绘制字母ABC:
    draw.text((random.randrange(0,100), random.randrange(0,100)), sentence, font=font, fill=color)
# 按字母-数字的格式存储文件
    image.save('./images/'+name+'.png')

