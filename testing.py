import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from pathlib import Path

# Load the trained model
model = load_model('saved/cnn.h5')

# Directory containing images
image_folder = Path('testimages')

# Load mapping from file
def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            class_index, ascii_code = line.strip().split()
            mapping[int(class_index)] = int(ascii_code)
    return mapping

# Load the mapping
mapping = load_mapping('data/raw/emnist-balanced-mapping.txt')

def show_image(title, image, cmap='gray'):
    plt.figure()
    plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()

def preprocess_image(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    show_image("Original Image", img)  # Show the original image

    # Thresholding
    _, thresh_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    show_image("Thresholded Image", thresh_img)  # Show the thresholded image

    return thresh_img

def segment_characters(thresh_img):
    # Find contours and sort by x-coordinate to ensure left-to-right order
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        char = thresh_img[y:y+h, x:x+w]
        char_resized = cv2.resize(char, (28, 28)) / 255.0
        char_images.append(np.expand_dims(char_resized, axis=-1))  # Add channel dimension
        bounding_boxes.append((x, y, w, h, char_resized))  # Keep bounding box info for sorting

    # Sort characters from left to right based on the x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0])
    return bounding_boxes  # Return bounding boxes with sorted order

def predict_text(bounding_boxes):
    result = ""
    predictions = []

    for i, (x, y, w, h, char_resized) in enumerate(bounding_boxes):
        char = np.expand_dims(char_resized, axis=0)  # Expand to 4D for model input
        prediction = model.predict(char)
        predicted_class = np.argmax(prediction)
        
        # Map the predicted class to the correct ASCII character using the mapping
        ascii_code = mapping.get(predicted_class, None)
        if ascii_code is not None:
            predicted_char = chr(ascii_code)
        else:
            predicted_char = '?'
        
        predictions.append((predicted_char, char_resized))
        result += predicted_char

    # Display all characters and predictions in a single plot
    fig, axes = plt.subplots(1, len(predictions), figsize=(15, 3))
    for ax, (predicted_char, char_img) in zip(axes, predictions):
        ax.imshow(char_img, cmap='gray')
        ax.set_title(predicted_char)
        ax.axis('off')
    
    plt.suptitle("Predicted Characters")
    plt.show()

    return result

# Loop through each image in the folder
for img_path in image_folder.glob('*.png'):  # Adjust for your image format
    thresh_img = preprocess_image(img_path)
    bounding_boxes = segment_characters(thresh_img)
    text = predict_text(bounding_boxes)
    print(f"Detected text in {img_path.name}: {text}")
