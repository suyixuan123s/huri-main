import cv2
import huri.vision.yolov5x6.detect as yolo
img = cv2.imread("D:\\chen\\huri_shared\\huri\\components\\utils\\20220927165134.jpg")
yolo_img, yolo_results = yolo.detect(source=img, weights="best.pt")
cv2.imshow("YOLO Results", yolo_img)
cv2.waitKey(0)
