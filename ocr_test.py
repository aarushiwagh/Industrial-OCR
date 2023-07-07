from paddleocr import PaddleOCR,draw_ocr
import os
import cv2
ocr = PaddleOCR(use_angle_cls=True)
img_path = r"C:\Users\Aarushi Wagh\Downloads\final_metal\val\11v.jpg"
result = ocr.ocr(img_path)
res = ""
for i in range(len(result[0])):
  res = res+" "+result[0][i][1][0]
print(res)