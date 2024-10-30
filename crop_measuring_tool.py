from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = './e2e_data/data1/2024-10-25-14-33-57/2024-10-25-14-45-10-195546.jpg'  # Replace with the path to your image
image = Image.open(image_path)

# Display the image and use plt.ginput() to record points
plt.imshow(image)
plt.title("Click on two points to define the crop area (top-left, bottom-right)")
points = plt.ginput(2)  # Select two points, top-left and bottom-right
plt.close()  # Close the plot window after the second point is selected

# Extract x, y coordinates
(x1, y1), (x2, y2) = points
x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

print(f"Selected crop area:\nimage_crop_x1: {x1}\nimage_crop_y1: {y1}\nimage_crop_x2: {x2}\nimage_crop_y2: {y2}")

# Crop the image based on selected coordinates
cropped_image = image.crop((x1, y1, x2, y2))
cropped_image.show()  # Display cropped image to verify
