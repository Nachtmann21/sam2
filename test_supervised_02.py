import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Path to the image
image_path = './AFIO/IM000135/IM000135.JPG'

# Array to store click coordinates and labels
click_coordinates = []
point_labels = []

# Callback function to capture mouse click coordinates and labels
def click_event(event, x, y, flags, params):
    global click_coordinates, point_labels
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click for vessel point
        click_coordinates.append((x, y))
        point_labels.append(1)
        print(f"Vessel point added at: ({x}, {y})")
        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow('Image', img)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for background point
        click_coordinates.append((x, y))
        point_labels.append(0)
        print(f"Background point added at: ({x}, {y})")
        cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
        cv2.imshow('Image', img)

# Load the image with OpenCV
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Create a window and set the callback function for mouse click events
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

# Wait until the 'q' key is pressed to close the window
print("Press 'q' to close the window and run segmentation.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Print the list of coordinates and labels in two separate arrays
print("All Click Coordinates and Labels:")
for coord, label in zip(click_coordinates, point_labels):
    print(f"Coordinate: {coord}, Label: {label}")


# Convert click_coordinates and point_labels to numpy arrays for SAM2 model
point_coords = np.array(click_coordinates)
point_labels = np.array(point_labels)

# SAM2 model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Set up model checkpoint and configuration
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "D:\\Desktop\\sam2\\sam2\\configs\\sam2.1\\sam2.1_hiera_l.yaml"

# Initialize SAM2ImagePredictor
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load the image for segmentation
image = Image.open(image_path).convert("RGB")
image = np.array(image)

# Run prediction with SAM2
predictor.set_image(image)
with torch.inference_mode():
    masks, _, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True  # Use False for a single mask output
    )

# Define function to display the mask
def show_mask(mask):
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

# Display original image and segmented mask side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 2, 2)
show_mask(masks[0])  # Display the first mask
plt.title("Segmented Vessel Mask")
plt.show()
