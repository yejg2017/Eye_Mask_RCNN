from PIL import Image
import numpy as np
import math
import os

path = 'D:/Eye/train_jpg/try/labelme/'
newpath = 'D:/Eye/train_jpg/try/labelme/'


def toeight():
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for file in filelist:
        if os.path.isdir(file):
            whole_path = os.path.join(path, file)+'/label.png'
            img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
            img = np.array(img)
            # img = Image.fromarray(np.uint8(img / float(math.pow(2, 16) - 1) * 255))
            img = Image.fromarray(np.uint8(img))
            img.save(newpath + file+'/new_label.png')

if __name__=='__main__':
    toeight()
    
    
    
img=Image.open('D:/Eye/train_jpg/try/labelme/')
img = Image.fromarray(np.uint8(img)*20)  