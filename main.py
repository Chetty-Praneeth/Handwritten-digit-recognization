import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDRY = 5

WHITE= (255,255,255)
BLACK=(0,0,0)
RED= (255,0,0)

IMSAVE=False
MODEL= load_model("my_model.h5")
LABELS={0:"ZERO",1:"ONE",2:"TWO",3:"THREE",4:"FOUR",5:"FIVE",6:"SIX",7:"SEVEN",8:"EIGHT",9:"NINE"}
#initializing the game
pygame.init()

FONT= pygame.font.Font(None, 18)
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit Board")

iswritting = False
number_xcord= []
number_ycord= []

imag_cnt=1

PREDICT = True
while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswritting :
            xcord, ycord =event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord,ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        if event.type == MOUSEBUTTONDOWN:
            iswritting = True

        if event.type == MOUSEBUTTONUP:
            iswritting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0]- BOUNDRY, 0), min(WINDOWSIZEX, number_xcord[-1]+ BOUNDRY)
            rect_min_y, rect_max_y = max(number_ycord[0]- BOUNDRY, 0), min(number_ycord[-1]+ BOUNDRY, WINDOWSIZEY)

            number_xcord = []
            number_ycord = []

            img_arr = np.array(pygame.PixelArray(DISPLAYSURFACE))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)


            if IMSAVE:
                cv2.imwrite("img.png")
                imag_cnt +=1

            if PREDICT:

                
                image = cv2.resize(img_arr,(28, 28))
                image = np.pad(image, (10,10), 'constant', constant_values=0)
                image = cv2.resize(image,(28,28))/255

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])

                textsurface = FONT.render(label, True, RED, WHITE)
                textrecobj = textsurface.get_rect()
                textrecobj.left, textrecobj.bottom = rect_min_x, rect_max_y

                pygame.draw.rect(DISPLAYSURFACE, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

                DISPLAYSURFACE.blit(textsurface, textrecobj)

            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURFACE.fill(BLACK)

        pygame.display.update()