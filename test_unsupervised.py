import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

# Select the device for computation
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

# Create predictor object
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load image
image = Image.open("./AFIO/IM000135/IM000135.JPG").convert("RGB")
image = np.array(image)

# Run prediction
predictor.set_image(image)
with torch.inference_mode():
    masks, _, _ = predictor.predict(
        multimask_output=False  # Use False for a single mask output
    )

# Define function to display the result
def show_mask(mask):
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

# Display original image and mask side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 2, 2)
show_mask(masks[0])  # Display the mask
plt.title("Segmented Vessel Mask")
plt.show()