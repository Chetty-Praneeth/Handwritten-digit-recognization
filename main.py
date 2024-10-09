import pygame
import sys
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRY = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMSAVE = False
MODEL = load_model("my_model.h5")
LABELS = {0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR",
           5: "FIVE", 6: "SIX", 7: "SEVEN", 8: "EIGHT", 9: "NINE"}

# Initializing the game
pygame.init()
FONT = pygame.font.Font(None, 18)
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []

imag_cnt = 1
PREDICT = True

while True:
    try:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION and iswriting:
                xcord, ycord = event.pos
                pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord, ycord), 4, 0)
                number_xcord.append(xcord)
                number_ycord.append(ycord)

            if event.type == pygame.MOUSEBUTTONDOWN:
                iswriting = True

            if event.type == pygame.MOUSEBUTTONUP:
                iswriting = False

                # Extract the drawn area
                if number_xcord and number_ycord:
                    rect_min_x = max(min(number_xcord) - BOUNDRY, 0)
                    rect_max_x = min(max(number_xcord) + BOUNDRY, WINDOWSIZEX)
                    rect_min_y = max(min(number_ycord) - BOUNDRY, 0)
                    rect_max_y = min(max(number_ycord) + BOUNDRY, WINDOWSIZEY)

                    img_arr = np.array(pygame.surfarray.pixels3d(DISPLAYSURFACE)).astype(np.float32)
                    img_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y]

                    if img_arr.size == 0:
                        print("No image data found.")
                        continue

                    # Convert to grayscale and invert if needed
                    image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
                    image = cv2.bitwise_not(image)  # If needed, invert colors to match training data

                    # Add padding to ensure the aspect ratio is maintained
                    h, w = image.shape
                    if h > w:
                        padding = (h - w) // 2
                        image = cv2.copyMakeBorder(image, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
                    else:
                        padding = (w - h) // 2
                        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

                    image = cv2.resize(image, (28, 28))  # Resize to 28x28
                    image = image.astype('float32') / 255.0  # Normalize
                    image = image.reshape(1, 28, 28, 1)


                    # Extract the drawn area
                    img_arr = np.array(pygame.surfarray.pixels3d(DISPLAYSURFACE)).astype(np.float32)
                    img_arr = img_arr[rect_min_x:rect_max_x, rect_min_y:rect_max_y]

                    # Print the shape of img_arr for debugging
                    print("Shape of img_arr before processing:", img_arr.shape)

                    if img_arr.size == 0:
                        print("No image data found.")
                        continue

                    if IMSAVE:
                        cv2.imwrite("img.png")
                        imag_cnt += 1

                    

                    if PREDICT:
                        # Convert to grayscale and normalize
                        image = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

                        # Resize to (28, 28) for the model
                        image = cv2.resize(image, (28, 28))

                        # Normalize the image
                        image = image.astype('float32') / 255.0  # Normalize to [0, 1]

                        # Reshape for model input
                        image = image.reshape(1, 28, 28, 1)  # (1, 28, 28, 1) for Keras model
                        print("Reshaped image for model:", image.shape)
                        
                        # Predict and get the label
                        prediction = MODEL.predict(image)
                        label = str(LABELS[np.argmax(prediction)])

                        print("Prediction result:", prediction)  # Output the raw prediction
                        print("Predicted label:", label)

                        textsurface = FONT.render(label, True, RED, WHITE)
                        textrecobj = textsurface.get_rect()
                        textrecobj.left, textrecobj.bottom = rect_min_x, rect_max_y

                        pygame.draw.rect(DISPLAYSURFACE, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)
                        DISPLAYSURFACE.blit(textsurface, textrecobj)

                number_xcord = []  # Reset coordinates after prediction
                number_ycord = []

            if event.type == pygame.KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()

    except Exception as e:
        print("An error occurred:", e)
        pygame.quit()
        sys.exit()
