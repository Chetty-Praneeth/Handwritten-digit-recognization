import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('my_model.h5') 

def preprocess_image(image_path):
    #image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize to 28x28 pixels
    img_resized = cv2.resize(img, (28, 28))
    # Normalize the image
    img_normalized = img_resized / 255.0
    # Reshape the image to match the input shape (1, 28, 28, 1)
    img_reshaped = np.reshape(img_normalized, (1, 28, 28, 1))
    return img_reshaped


def predict_digit(image_path):
  
    preprocessed_image = preprocess_image(image_path)
    
    predictions = model.predict(preprocessed_image)
    predicted_digit = np.argmax(predictions)  
    return predicted_digit


image_path = 'digit.png'  
predicted_digit = predict_digit(image_path)

print(f"The predicted digit is: {predicted_digit}")


img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
