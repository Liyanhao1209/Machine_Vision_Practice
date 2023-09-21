import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((4, 4), np.uint8)

img = cv2.imread('pic/rice.jpg')
cv2.imshow('Original Image', img)
cv2.waitKey(0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
cv2.imshow('GrayImg', gray_img)
cv2.waitKey(0)

# 观察到一件事情，整张图片上半部分颜色浅（亮度高），下半部分颜色深（亮度低）
# 也就是说，如果二值化的话，上半部分应该阈值高一点，下半部分应该阈值低一点

# 定义下半部分二值化图片
ux,uy,uh,uw=gray_img.shape[0],0,gray_img.shape[1],gray_img.shape[0]//3
u_area = gray_img[ux-uw:ux,uy:uy+uh]
ret,uth = cv2.threshold(u_area,110,255,cv2.THRESH_BINARY)
# 整张图片二值化
ret, th1 = cv2.threshold(gray_img,145, 255, cv2.THRESH_BINARY)
# 把一开始的上半部分换进去
th1[ux-uw:ux,uy:uy+uh] = uth
# cv2.imshow('uth',uth)
cv2.imshow('Threshold', th1)
cv2.waitKey(0)

erosion = cv2.erode(th1, kernel, iterations=2)  # 腐蚀
#腐蚀的时候老操作，上面部分要多腐蚀点
ux,uy,uh,uw=0,0,th1.shape[1],4*th1.shape[0]//5
u_area = th1[ux:ux+uw,uy:uy+uh]
kernel = np.ones((5,5),np.uint8)
u_erosion = cv2.erode(u_area,kernel,iterations=1)
kernel = np.ones((2,2),np.uint8)
u_erosion = cv2.erode(u_erosion,kernel,iterations=1)
erosion[ux:ux+uw,uy:uy+uh] = u_erosion
cv2.imshow('uErosion',u_erosion)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
dist_img = cv2.distanceTransform(erosion, cv2.DIST_C, cv2.DIST_MASK_5)  # 距离变换
cv2.imshow('DistanceTransformation', dist_img)
cv2.waitKey(0)
dist_output = cv2.normalize(dist_img, 0, 1.0, cv2.NORM_MINMAX)  # 归一化
cv2.imshow('Normalize', dist_output * 200)
cv2.waitKey(0)
ret, th2 = cv2.threshold(dist_output * 200, 0.3, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold2', th2)
cv2.waitKey(0)
# erosion2=cv2.erode(th2,kernel,iterations=3)
# cv2.imshow('Erosion2', erosion2)
# cv2.waitKey(0)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
opening = np.array(opening, np.uint8)
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓提取
count = 0
for cnt in contours:
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    img = cv2.putText(img,str(count),center,font,0.4,(0,0,255))
    count+=1
img = cv2.putText(img, ('sum=' + str(count)), (50, 50), font, 1, (255, 0, 0))
cv2.imshow('circle_img', img)
cv2.waitKey(0)
# print('大米粒数量：', contours.__len__())
print('大米粒数量：',count)
cv2.destroyAllWindows()

