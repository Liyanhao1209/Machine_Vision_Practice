import cv2

image0 = cv2.imread('test1.jpg')
image = cv2.resize(image0, (800, 533))

cv2.imshow('original', image)
cv2.waitKey(0)
# '''图片预处理'''
H, W = image.shape[:2]  # 获取尺寸
blod = cv2.dnn.blobFromImage(image, 1, (H, W), (0, 0, 0), swapRB=True, crop=False)

#net = cv2.dnn.readNet('starry_night.t7')
net = cv2.dnn.readNet('mosaic.t7')

net.setInput(blod)
out = net.forward()
out_new = out.reshape(out.shape[1], out.shape[2], out.shape[3])  # 将输出进行加一化处理
cv2.normalize(out_new, out_new, norm_type=cv2.NORM_MINMAX)
# 通道转换
result = out_new.transpose(1, 2, 0)

cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
