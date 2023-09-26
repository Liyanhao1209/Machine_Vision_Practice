import cv2 # short for open cv
import numpy as np # numpy is a library for data statistics

font = cv2.FONT_HERSHEY_COMPLEX
kernel = np.ones((7, 7), np.uint8)

img = cv2.imread('pic/corns.jpg')
cv2.imshow('Original Image', img)
cv2.waitKey(0) # blocked until some events happen
# cvt short for convert,convert color to GRAY,BGR short for blue,green and red
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
cv2.imshow('GrayImg', gray_img)
cv2.waitKey(0)
# threshold:阈值 the value between 120 and 255 convert to 1,the others convert to 0
ret, th1 = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold', th1)
cv2.waitKey(0)
erosion = cv2.erode(th1, kernel, iterations=1)  # 腐蚀
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
dist_img = cv2.distanceTransform(erosion, cv2.DIST_L1, cv2.DIST_MASK_3)  # 距离变换
cv2.imshow('DistanceTransformation', dist_img)
cv2.waitKey(0)
dist_output = cv2.normalize(dist_img, 0, 1.0, cv2.NORM_MINMAX)  # 归一化
cv2.imshow('Normalize', dist_output * 80)
cv2.waitKey(0)
ret, th2 = cv2.threshold(dist_output * 80, 0.3, 255, cv2.THRESH_BINARY)
cv2.imshow('ThresholdAfterNormalize', th2)
cv2.waitKey(0)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey(0)
opening = np.array(opening, np.uint8)
contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓提取
count = 0
for cnt in contours:
	(x, y), radius = cv2.minEnclosingCircle(cnt)
	center = (int(x)-15, int(y))
	radius = int(radius)
	circle_img = cv2.circle(opening, center, radius, (255, 255, 255), 1)
	area = cv2.contourArea(cnt)
	area_circle = 3.14 * radius * radius
	# print(area/area_circle)
	if area / area_circle <= 0.5:
		# img = cv2.drawContours(img, cnt, -1, (0,0,255), 5)#差（红色）
		img = cv2.putText(img, 'bad', center, font, 0.5, (0, 0, 255))
	elif area / area_circle >= 0.6:
		# img = cv2.drawContours(img, cnt, -1, (0,255,0), 5)#优（绿色）
		img = cv2.putText(img, 'good', center, font, 0.5, (0, 0, 255))
	else:
		# img = cv2.drawContours(img, cnt, -1, (255,0,0), 5)#良（蓝色）
		img = cv2.putText(img, 'normal', center, font, 0.5, (0, 0, 255))
	count += 1
img = cv2.putText(img, ('sum=' + str(count)), (50, 50), font, 1, (255, 0, 0))
cv2.imshow('circle_img', img)
cv2.waitKey(0)
print('玉米粒数量：', count)
cv2.destroyAllWindows()
