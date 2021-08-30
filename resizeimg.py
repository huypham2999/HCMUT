import cv2
import numpy as np
#Resizing

img = cv2.imread("D:/PycharmProjects/yolov5/yolov5/leaf.jpg")
print(img.shape)

imgResize = cv2.resize(img,(227,227))


cv2.imshow("Resize Image",imgResize)

cv2.imwrite('2leaf.jpg', imgResize)
cv2.waitKey(0)