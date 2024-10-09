import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
WINDOWSIZEX = 640
WINDOWSIZEY = 480

WHITE= (255,255,255)
BLACK=(0,0,0)
RED= (255,0,0)

IMGAVE=False
MODEL= load_model("my_model.h5")
LABELS={0:"ZERP",1:"ONE",2:"TWO",3:"THREE",4:"FOUR",5:"FIVE",6:"SIX",7:"SEVEN",8:"EIGHT",9:"NINE"}
#initializing the game
pygame.init()

#FONT= pygame.font.Font("freesansbold.tff", 18)
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digit Board")

iswritting = False
while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswritting :
            xcord, ycord =event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (xcord,ycord), 4, 0)
            