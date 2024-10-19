import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained model (in .h5 format)
model = load_model('my_model.h5')  # replace 'your_model.h5' with your actual model path

# Function to preprocess the image before passing it to the model
def preprocess_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded properly
    if img is None:
        raise ValueError(f"Error: Unable to load image at path: {image_path}")
    
    # Resize to 28x28 pixels (MNIST image size)
    img_resized = cv2.resize(img, (28, 28))
    # Normalize the image to match the input format of the model
    img_normalized = img_resized / 255.0
    # Reshape the image to match the input shape (1, 28, 28, 1) for the model
    img_reshaped = np.reshape(img_normalized, (1, 28, 28, 1))
    
    return img_reshaped

# Function to predict the digit
def predict_digit(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    # Get the prediction from the model
    predictions = model.predict(preprocessed_image)
    predicted_digit = np.argmax(predictions)
    return predicted_digit, 

# Example usage with multiple images
image_paths = [
    'digit.png',  # Replace with your image file paths
    'digit2.png',
    'digit3.png'
]

for image_path in image_paths:
    try:
        predicted_digit = predict_digit(image_path)
        print(f"The predicted digit for {image_path} is: {predicted_digit}")

        # Optionally, display the input image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted Digit: {predicted_digit}")
        plt.show()

    except ValueError as e:
        print(e)
