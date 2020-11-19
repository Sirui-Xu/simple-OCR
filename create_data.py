import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
from tqdm import tqdm
# mac 存放的字体文件夹 
font_dir = '/System/Library/Fonts/'
fonts = ['SFNSDisplayCondensed-Black.otf', 'SFNSDisplayCondensed-Regular.otf', 'SFNSTextCondensed-Heavy.otf',
         'Geneva.dfont', 'SFCompactDisplay-Medium.otf', 'Palatino.ttc', 'SFCompactText-BoldItalic.otf', 'Keyboard.ttf',
         'SFNSTextCondensed-Bold.otf', 'SFCompactDisplay-Regular.otf', 'Optima.ttc', 'SFNSTextCondensed-Medium.otf',
         'SFCompactRounded-Medium.otf', 'Avenir.ttc', 'PingFang.ttc', 'SFCompactText-LightItalic.otf', 
         'SFCompactRounded-Regular.otf']
def getChar():
    k = random.randint(0, 1)
    chars = [random.randint(65, 90), random.randint(97, 122)]
    return chr(chars[k])


def getColor():
    return (random.randint(32, 200), random.randint(32, 200), random.randint(32, 200))


width = 32
height = 32
# 画所有字母
char_positive = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']
char_data = [chr(i) for i in range(65,91)] + [chr(i) for i in range(97,123)]
os.makedirs('./data', exist_ok=True)
# 生成所有数据集
for char in char_data:
    for i in range(50):
# 生成图片的底
        image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Draw对象:
        draw = ImageDraw.Draw(image)
# 确定字体类型，需要加载对应的ttf文件
        font_id = random.randrange(0, len(fonts))
        font_path = os.path.join(font_dir, fonts[font_id])
        # print(fonts[font_id])
        font = ImageFont.truetype(font_path, random.randint(19,30))

# 输出文字 按照不同的起始位置 不同的字体大小 不同的颜色 绘制字母ABC:
        draw.text((random.randrange(0,2), random.randrange(0,2)), char, font=font, fill=getColor())
# 按字母-数字的格式存储文件
        image.save('./data/'+str(ord(char))+'-'+str(i)+'.png')