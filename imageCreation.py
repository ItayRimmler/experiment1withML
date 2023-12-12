import numpy as np
import cv2
import tensorflow as tf
import random


def CreateDataForML(leng):

    # We can choose whether to get shapes within a container in a random location (option 1) or to get a focused shape (option 2):
    cir, squ, tri = GetShapes()
    #cir, squ, tri = cv2.imread('assets/ML/Circle.png'), cv2.imread('assets/ML/Square.png'), cv2.imread('assets/ML/Triangle.png')

    # For the 3 shapes, we do the following (in that order):
    # 1. We simplify it by turning it gray
    # 2. We normalize it

    cir = cv2.cvtColor(cir, cv2.COLOR_BGR2GRAY)
    cirX = cir.shape[0]
    cirY = cir.shape[1]
    cir = np.array(cir.reshape(cirX*cirY).astype("float32") / 255.0)
    squ = cv2.imread('assets/ML/Square.png')
    squ = cv2.cvtColor(squ, cv2.COLOR_BGR2GRAY)
    squX = squ.shape[0]
    squY = squ.shape[1]
    squ = np.array(squ.reshape(squX * squY).astype("float32") / 255.0)
    tri = cv2.imread('assets/ML/Triangle.png')
    tri = cv2.cvtColor(tri, cv2.COLOR_BGR2GRAY)
    triX = tri.shape[0]
    triY = tri.shape[1]
    tri = np.array(tri.reshape(triX * triY).astype("float32") / 255.0)

    # We make sure that the images are with the same size:
    minimum = min(cirX*cirY,squX * squY,triX * triY)
    cir = cir[:minimum]
    squ = squ[:minimum]
    tri = tri[:minimum]

    # We create labels:
    labels = np.random.randint(0, 3, size=leng)

    # According to the labels produced, we create an array of shapes:
    shapes = np.empty((0, minimum))
    for i in range(0, leng):
        if labels[i] == 0:
            shapes = np.vstack((shapes, cir))
        if labels[i] == 1:
            shapes = np.vstack((shapes, squ))
        if labels[i] == 2:
            shapes = np.vstack((shapes, tri))

    return shapes, labels

# A function that returns us 3 shapes with random locations within an image:
def GetShapes():
    circles = ['assets/ML/Container1withCircle.png', 'assets/ML/Container2withCircle.png'
               , 'assets/ML/Container3withCircle.png', 'assets/ML/Container3withCircle(2).png'
               , 'assets/ML/Container4withCircle.png', 'assets/ML/Container5withCircle.png']
    squares = ['assets/ML/Container1withSquare.png', 'assets/ML/Container2withSquare.png'
               , 'assets/ML/Container3withSquare.png', 'assets/ML/Container4withSquare.png',
               'assets/ML/Container5withSquare.png', 'assets/ML/Container5withSquare(2).png']
    triangles = ['assets/ML/Container1withTriangle.png', 'assets/ML/Container2withTriangle.png',
                 'assets/ML/Container2withTriangle(2).png', 'assets/ML/Container3withTriangle.png',
                 'assets/ML/Container4withTriangle.png', 'assets/ML/Container5withTriangle.png']
    cir = cv2.imread(random.choice(circles))
    squ = cv2.imread(random.choice(squares))
    tri = cv2.imread(random.choice(triangles))

    return cir, squ, tri