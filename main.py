from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR,draw_ocr
import io
import cv2
from PIL import Image
import torch
from IPython.display import Image as IPImage
import numpy as np
import shutil

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), use_camera: bool = False):
    if use_camera:
        # Capture frame from the camera
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        image = Image.fromarray(frame)
    else:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

    file_path = r"upload_img.jpg"
    image.save(file_path)

    ocr = PaddleOCR(use_angle_cls=True)  
    result = ocr.ocr(file_path)
    res = ""
    for i in range(len(result[0])):
        res = res+" "+result[0][i][1][0] 

    # Return the prediction results
    response = JSONResponse({"predictions": res})
    response.headers["Cache-Control"] = "no-cache"
    return response
    
