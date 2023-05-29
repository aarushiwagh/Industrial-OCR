from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import cv2
from PIL import Image
import torch
from IPython.display import Image as IPImage
import numpy as np

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile = File(None), use_camera: bool = False):
    if use_camera:
        # Capture frame from the camera
        capture = cv2.VideoCapture(0)
        ret, frame = capture.read()
        image = Image.fromarray(frame)
    else:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'model\metal_best_3.pt')

    # Set the input image size (e.g., 640x640)
    input_size = 640

    # Load and preprocess the image
    image = image.resize((input_size, input_size))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    # Run inference on the image
    results = model(image)
    classes = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E', 'G', 'I', 'K', 'M', 'N', 'P', 'R', 'S', 'T']
    # Extract the bounding box coordinates, class labels, and confidence scores
    boxes = results.xyxy[0][:, :4].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)  # Class labels
    scores = results.xyxy[0][:, 4].cpu().numpy()  # Confidence scores
    sorted_indices = boxes[:, 0].argsort()
    sorted_boxes = boxes[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    class_labels = [classes[i] for i in sorted_labels]
    detected_string = ''.join(class_labels)
    # Return the prediction results
    response = JSONResponse({"predictions": detected_string})
    response.headers["Cache-Control"] = "no-cache"
    return response
    
