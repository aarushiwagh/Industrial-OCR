
from PIL import Image
import torch
from IPython.display import Image as IPImage
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'model\metal_best_5.pt')

# Set the input image size (e.g., 640x640)
input_size = 640

# Load and preprocess the image
image = Image.open(r"C:\Users\Aarushi Wagh\Downloads\imgd.jpg")
image = image.resize((input_size, input_size))
image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

# Run inference on the image
results = model(image)
classes = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'E', 'G', 'I', 'K', 'M', 'N', 'P', 'R', 'S', 'T']
confidence_threshold = 0.61
# Extract the bounding box coordinates, class labels, and confidence scores
boxes = results.xyxy[0][:, :4].cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
labels = results.xyxy[0][:, -1].cpu().numpy().astype(int)  # Class labels
scores = results.xyxy[0][:, 4].cpu().numpy()  # Confidence scores
sorted_indices = boxes[:, 0].argsort()
sorted_boxes = boxes[sorted_indices]
sorted_labels = labels[sorted_indices]
sorted_scores = scores[sorted_indices]

filtered_indices = sorted_scores > confidence_threshold
filtered_boxes = sorted_boxes[filtered_indices]
filtered_labels = sorted_labels[filtered_indices]
filtered_scores = sorted_scores[filtered_indices]

class_labels = [classes[i] for i in filtered_labels]
all_labels = [classes[i] for i in sorted_labels]
detected_string = ''.join(class_labels)
if 'S' in all_labels or 'C' in all_labels or 'R' in all_labels and '5' not in all_labels:
    print("Detected Letters (Left to Right): SPICER")
else:
    print("Detected Letters (Left to Right):", detected_string)

print(all_labels)
print(sorted_scores)

