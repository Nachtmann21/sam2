import cv2

# Path to the image
image_path = './AFIO/IM000135/IM000135.JPG'

# Array to store click coordinates
click_coordinates = []

def click_event(event, x, y, flags, params):
    """
    Callback function to capture mouse click coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add coordinates to the array
        click_coordinates.append((x, y))
        print(f"Clicked at: ({x}, {y})")
        # Draw a point where the click occurred
        cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        # Update the image to show the clicked point
        cv2.imshow('Image', img)

# Load the image
img = cv2.imread(image_path)

# Create a window and set the callback function for mouse click events
cv2.imshow('Image', img)
cv2.setMouseCallback('Image', click_event)

# Wait until the 'q' key is pressed to close the window
print("Press 'q' to close the window.")
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print the list of coordinates
print("All Click Coordinates:", click_coordinates)

# Destroy all the windows
cv2.destroyAllWindows()
