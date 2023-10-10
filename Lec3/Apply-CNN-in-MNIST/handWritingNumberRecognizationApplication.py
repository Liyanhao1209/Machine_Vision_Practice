import os
import threading
from glob import glob

import cv2
import numpy as np
from keras.models import load_model


def readFiles(f_path):
    return glob(
        os.path.join(f_path + "\\*.png")
    )


def showImages(eles):
    list = []
    for i in range(len(eles)):
        img = cv2.imread(eles[i])
        cv2.imshow('img' + str(i), img)
    cv2.waitKey(0)


def getImages(eles):
    l = []
    for i in range(len(eles)):
        img = cv2.imread(eles[i])
        l.append(img)
    img = l[0]
    ans = np.empty((len(l), img.shape[0], img.shape[1]))
    for i, e in enumerate(l):
        ans[i] = cv2.cvtColor(e, cv2.COLOR_BGR2GRAY)
    return ans


# C:\workplace\Machine_Vision_Practice\Lec3\pic\hwNums
path = input("输入手写数字图片所在文件夹路径:")
files = readFiles(path)
images = getImages(files)
t_show = threading.Thread(target=showImages,args=(files,))
t_show.start()
img_x, img_y = images.shape[1], images.shape[2]
images = images.reshape(images.shape[0], img_x, img_y, 1)
images = images.astype('float32')
images /= 255

model = load_model('cnn.h5')

predictions = model.predict(images)
print('predictions')
print(np.argmax(predictions, axis=1))
