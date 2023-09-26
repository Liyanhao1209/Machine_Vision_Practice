import cv2
#原图
img = cv2 .imread ("pic/faces.png")
cv2.imshow("Original img",img)
#使用预训练摸型创建cascade分类器
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

#识别，将结果存储到faces变量中
faces = faceCascade.detectMultiScale(img, 1.1, 8)
#前后两次相继的扫描中搜索窗口的比例系数，默认为1.1 即每次搜索窗口扩大10%
#构成检测目标的相邻矩形的最小个数

for (x,y,w,h) in faces:
   #将结果绘制到原图中
   cv2.rectangle(img, (x,y),(x+w , y+h),(0,0,255),2)
cv2.imshow("img",img)
cv2.waitKey(0)