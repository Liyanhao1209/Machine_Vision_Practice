import cv2

# 原图
img = cv2.imread("pic/faces2.jpg")
cv2.imshow("Original img", img)
# 使用预训练摸型创建cascade分类器
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray img", gray_img)

# ret,thres = cv2.threshold(gray_img, 140, 255, cv2.THRESH_BINARY)
# thres = cv2.adaptiveThreshold(gray_img,130,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-10)
# cv2.imshow("thres", thres)

# 识别，将结果存储到faces变量中
# ux,uy,uw,uh = 0,0,img.shape[0]//4,img.shape[1];
# u_area = img[ux:ux+uw,uy:uy+uh]
# uf = faceCascade.detectMultiScale(u_area,1.05,8);
# faces = faceCascade.detectMultiScale(img, 1.1, 8)
# faces[0:uf.shape[0],0:uf.shape[1]]=uf
faces = faceCascade.detectMultiScale(img, 1.05, 4)
# 前后两次相继的扫描中搜索窗口的比例系数，默认为1.1 即每次搜索窗口扩大10%
# 构成检测目标的相邻矩形的最小个数

for (x, y, w, h) in faces:
    # 将结果绘制到原图中
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
