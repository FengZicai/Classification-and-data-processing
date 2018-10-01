'''
Data_Augmentation :图片翻转
'''
from PIL import Image
import os
import os.path

rootdir = r'E:\AI challenger\train_change_simplify' + '\\' + '60'
for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)

        im = Image.open(currentPath)
        out = im.transpose(Image.TRANSPOSE)
        newname = 'E:\\AI challenger\\train_change_simplify\\' + '60' + '\\' + filename + '-TRANSPOSE.jpg'
        out.save(newname)
