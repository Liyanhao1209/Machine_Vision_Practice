import cv2

# 原图
img = cv2.imread("pic/faces2.jpg")
cv2.imshow("Original img", img)
img = cv2.resize(img, (0,0), fx=1.5, fy=1.5)
# 使用预训练摸型创建cascade分类器
# 默认Haar haarcascade_frontalface_default.xml
# 快速Haar haarcascade_frontalface_alt2.xml
# 快速LBP lbpcascade_frontalface.xml
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray img", gray_img)

ux,uy,uw,uh = 0,0,gray_img.shape[0]//5,gray_img.shape[1];
u_area = gray_img[ux:ux+uw,uy:uy+uh]
u_faces = faceCascade.detectMultiScale(u_area,1.01,6);


# 识别，将结果存储到faces变量中
faces = faceCascade.detectMultiScale(gray_img, 1.0185, 8)
# 前后两次相继的扫描中搜索窗口的比例系数，默认为1.1 即每次搜索窗口扩大10%
# 构成检测目标的相邻矩形的最小个数

for (x, y, w, h) in u_faces:
    # 将结果绘制到原图中
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
