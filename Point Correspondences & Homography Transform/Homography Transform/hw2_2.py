# -*- coding: utf-8 -*-
"""HW2-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ml9CZV8C8KOniJsKypmtI0W73D4mHOWi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from google.colab.patches import cv2_imshow
import math
from os import listdir
import random
from functools import reduce
import math

from google.colab import drive
drive.mount('/content/drive')

img = cv2.imread('/content/drive/MyDrive/HW2/Delta-Building.jpg')
fh = open('/content/drive/MyDrive/HW2/coordinate.txt')
p = []
oriPts, tarPts = [], []

# assign the value of target point's coordinates for source point
dX = 2.6
dY = 0.65
offsetX = 350
offsetY = 130
oriY = 802
for c in fh.readlines():
  if c[-1] == '\n':
    p.append(c[:-1])
  else:
    p.append(c)
    
for i in range(0, len(p)-1, 2):
  x1, y1 = p[i].split(', ')
  x2, y2 = p[i + 1].split(', ')
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  oriPts.append([x1, y1])
  oriPts.append([x2, y2])
  tarPts.append([x1 - offsetX, y1 + offsetY - (oriY - y1) * dY])
  tarPts.append([int(x1+(x2-x1)*dX) - offsetX, y1 + offsetY - (oriY - y1) * dY])

def Homography(oriPts, tarPts, iter):
  n = len(oriPts)
  minSquare = np.inf
  resultH = np.zeros((3,3))
  pointSet = []
  for it in range(iter):
    square = 0
    A = []
    # randomly select 4 points and compute the homography matrix
    randomID = random.sample(range(n), 4)
    for i in randomID:
      A.append([oriPts[i][0], oriPts[i][1], 1, 0, 0, 0, -oriPts[i][0]*tarPts[i][0], -oriPts[i][1]*tarPts[i][0], -tarPts[i][0]])
      A.append([0, 0, 0, oriPts[i][0], oriPts[i][1], 1, -oriPts[i][0]*tarPts[i][1], -oriPts[i][1]*tarPts[i][1], -tarPts[i][1]])
    u, s, v = np.linalg.svd(A)
    H = np.reshape(v[8], (3, 3))
    H = H / H.item(8)

    # compute the distance between transformed point and target point
    for i in range(n):
      x1, y1 = tarPts[i]
      x2, y2, z2 = H @ np.array([oriPts[i][0], oriPts[i][1], 1])
      x2, y2 = x2/z2, y2/z2
      square += (x1 - x2)**2 + (y1 - y2)**2
    # homography matrix with the least distance will be choosed
    if square < minSquare:
      minSquare = square
      resultH = H
      pointSet = randomID 

  print('Backward Homography Matrix:')
  print(resultH)

  return resultH, pointSet

def getImage(img, H):
  h, w = img.shape[:2]
  newImage = np.zeros((h, w, 3))
  for i in range(h):
    for j in range(w):
      # use H to obtain transformed point
      x, y, z = H @ np.array([j, i, 1])
      x, y = x/z, y/z
      x1, y1 = math.floor(x), math.floor(y)
      x2, y2 = math.ceil(x), math.ceil(y)

      if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
        continue

      # bilinear interpolation to estimate pixel value
      a, b = x - x1, y - y1
      p = ((1-a) * (1-b) * img[y1, x1]) + (a * (1-b) * img[y1, x2]) + (b * (1-a) * img[y2, x1]) + (a * b * img[y2, x2])
      
      newImage[i, j] = p

  return newImage

H, points = Homography(tarPts, oriPts, 10000)
newi = getImage(img, H)

selectOri = list(sorted([[int(oriPts[i][0]),int(oriPts[i][1])] for i in points]))
selectTar = list(sorted([[int(tarPts[i][0]),int(tarPts[i][1])] for i in points]))
for i in range(4):
  x1, y1 = selectOri[i]
  x2, y2 = selectTar[i]
  cv2.circle(img, (x1,y1), 2, (0,0,255), 2)
  cv2.circle(newi, (x2,y2), 2, (0,0,255), 2)

# connect the four points used to compute H
cv2.line(img, selectOri[0], selectOri[1], (0,0,255), 1)
cv2.line(img, selectOri[1], selectOri[2], (0,0,255), 1)
cv2.line(img, selectOri[2], selectOri[3], (0,0,255), 1)
cv2.line(img, selectOri[3], selectOri[0], (0,0,255), 1)

cv2.line(newi, selectTar[0], selectTar[1], (0,0,255), 1)
cv2.line(newi, selectTar[1], selectTar[2], (0,0,255), 1)
cv2.line(newi, selectTar[2], selectTar[3], (0,0,255), 1)
cv2.line(newi, selectTar[3], selectTar[0], (0,0,255), 1)

cv2.imwrite('selected_img.jpg', img)
cv2.imwrite('rectified_img.jpg', newi)