# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('深度学习入门/dataset/lena_gray.png') #读入图像
plt.imshow(img)

plt.show()