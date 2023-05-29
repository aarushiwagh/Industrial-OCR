
from PIL import Image
import torch
from IPython.display import Image as IPImage
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'model\metal_best_3.pt')

# Set the input image size (e.g., 640x640)
input_size = 640

# Load and preprocess the image
image = Image.open('images\IMG_20230406_145411_jpg.rf.828fba72c90f0aa19faa41630306e45e.jpg')
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
print("Detected Letters (Left to Right):", detected_string)

