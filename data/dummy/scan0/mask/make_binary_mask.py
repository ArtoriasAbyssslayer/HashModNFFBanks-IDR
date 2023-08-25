import cv2
import numpy as np

# Loop through image names from 000.png to 010.png
for i in range(11):
    image_filename = f'{i:03d}.png'
    binary_mask_filename = f'{i:03d}_mask.png'

    # Load the image
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding to create a binary mask
    threshold_value = 128  # Adjust this value based on your images
    _, binary_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the binary mask
    inverted_mask = cv2.bitwise_not(binary_mask)
    # Save the binary mask
    cv2.imwrite(binary_mask_filename, inverted_mask)

    print(f'Generated binary mask for {image_filename}')

print('Binary masks generation complete.')

