# -*- coding: utf-8 -*-
"""HW1_108034022.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hK806G-MPFgcR85S3Twrv7I2vwvhE-lD
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import math
import scipy
from scipy import signal
from os import listdir
from numpy.linalg import eig

from google.colab import drive
drive.mount('/content/drive')

# A.a Gaussian Smooth

def Gaussian(sigma, x, y):
  # return corresponding gaussian value
  mol = math.exp(-(x**2 + y**2)/(2 * sigma**2))
  den = 2 * math.pi * sigma**2
  return mol / den

def getKernel(size, sigma):
  # return a gaussian kernel of size = size
  pad = size // 2
  kernel = np.zeros((size, size))
  for i in range(size):
    for j in range(size):
      kernel[i, j] = Gaussian(sigma, i - pad, j - pad)
  return kernel / kernel.sum()

def gaussianSmoothing(img, size, sigma=5, read=0, imgName='', srcPath='', write=0, tarPath=''):
  # read image from local by setting read = 1
  if read:
    img = cv2.imread(srcPath+'/'+imgName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  kernel = getKernel(size, sigma) # get gaussian kernel of size = size
  result = scipy.signal.convolve2d(img, kernel, mode='valid') # convolve the img with gaussian kernel
  
  # write convolved image to local by setting write = 1
  if write:
    cv2.imwrite('%s/%d_%d_%s' % (tarPath, size, sigma, imgName), result)

  return result

# A.b Intensity Gradient (Sobel edge detection)
Hx = np.array([[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]]) # horizontal sobel operator
Hy = np.array([[1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]]) # vertical sobel operator
def intensityGradient(img, threshold, read=0, imgName='', srcPath='', write=0, tarPath=''):
  if read:
    img = cv2.imread(srcPath+'/'+imgName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
. 
  imgx = scipy.signal.convolve2d(img, Hx,mode='valid') # convolve image with Hx
  imgy = scipy.signal.convolve2d(img, Hy,mode='valid') # convolve image with Hy
  #magnitude
  magnitude = np.sqrt(np.square(imgx) + np.square(imgy)) # get the magnitude with magnitude = sqrt(Ix^2 + Iy^2)
  magnitude = (magnitude > threshold) * magnitude # filter out weak magnitudes
  #direction
  red = np.array([0, 0, 255])
  green = np.array([0, 255, 0])
  blue = np.array([255, 0, 0])
  yellow = np.array([0, 255, 255])
  cyan = np.array([255, 255, 0])
  purple = np.array([255, 0, 255])
  gray = np.array([122, 122, 122])
  orange = np.array([0, 133, 242])

  orientation = cv2.phase(imgy, imgx, angleInDegrees=True) # get direction of each pixel
  direction = np.zeros((imgx.shape[0], imgx.shape[1], 3), dtype = np.uint8)
  direction[(magnitude > 0) & (((0 <= orientation) & (orientation < 22.5)) | (337.5 <= orientation))] = red # 0: red
  direction[(magnitude > 0) & (22.5 <= orientation) & (orientation < 67.5)] = green # 45: green
  direction[(magnitude > 0) & (67.5 <= orientation) & (orientation < 112.5)] = blue # 90: blue
  direction[(magnitude > 0) & (112.5 <= orientation) & (orientation < 157.5)] = yellow # 135: yellow
  direction[(magnitude > 0) & (157.5 <= orientation) & (orientation < 202.5)] = cyan # 180: cyan
  direction[(magnitude > 0) & (202.5 <= orientation) & (orientation < 247.5)] = purple # 225: purple
  direction[(magnitude > 0) & (247.5 <= orientation) & (orientation < 292.5)] = orange # 270: orange
  direction[(magnitude > 0) & (292.5 <= orientation) & (orientation < 337.5)] = gray # 315: gray
  magnitude = magnitude * 200 / np.amax(magnitude) # normalize magnitudes

  if write:
    cv2.imwrite("%s/magnitude_%s" % (tarPath, imgName), magnitude)
    cv2.imwrite("%s/direction_%s" % (tarPath, imgName), direction)
  return imgx, imgy, magnitude, direction

"""**A.c Structure Tensor**"""

Hx = np.array([[-1/8, 0, 1/8], [-2/8, 0, 2/8], [-1/8, 0, 1/8]])
Hy = np.array([[1/8, 2/8, 1/8], [0, 0, 0], [-1/8, -2/8, -1/8]])
def structureTensor(img, size, read=0, imgName='', srcPath='', write=0, tarPath=''):
  pad = size // 2
  if read:
    img = cv2.imread(srcPath+'/'+imgName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  h, w = img.shape

  Ix = scipy.signal.convolve2d(img, Hx,mode='valid')
  Iy = scipy.signal.convolve2d(img, Hy,mode='valid')

  Ix2 = np.square(Ix) # Ix2 = Ix * Ix
  Iy2 = np.square(Iy) # Iy2 = Iy * Iy
  IxIy = Ix * Iy
  IyIx = Iy * Ix
  one = np.full((size, size), 1) # a size * size matrix full of 1

  convIx2 = scipy.signal.convolve2d(Ix2, one,  mode='valid') # convIx2[i, j] = Ix2[i - size//2, j - size//2] + Ix2[i - size//2 + 1, j - size//2 + 1] + ... + Ix2[i + size//2, j + size//2] 
  convIy2 = scipy.signal.convolve2d(Iy2, one,  mode='valid')
  convIxIy = scipy.signal.convolve2d(IxIy, one,  mode='valid')
  convIyIx = scipy.signal.convolve2d(IyIx, one,  mode='valid')
  tr = convIx2 + convIy2 # trace = Ix2 + Iy2
  det = convIx2 * convIy2 - convIxIy * convIyIx # determinant = Ix2 * Iy2 - IxIy * IyIx
  mat = det - 0.04 * tr * tr # corner response = determinant - k * trace^2
  '''
  # compute the structure tensor and smaller eigenvalue of each pixel
  mat = np.zeros((h, w))
  for y in range(pad, h - pad):
    for x in range(pad, w - pad):
      H = np.asarray([[convIx2[y, x], convIxIy[y, x]], [convIyIx[y, x], convIy2[y, x]]]) #np.zeros((2, 2))
      for i in range(y - pad, y + pad + 1):
        for j in range(x - pad, x + pad + 1):
          H[0, 0] += Ix2[i, j]
          H[0, 1] += IxIy[i, j]
          H[1, 0] += IyIx[i, j]
          H[1, 1] += Iy2[i, j]
      val, vec = eig(H)
      mat[y, x] = min(val[0], val[1])
  '''

  if write:
    cv2.imwrite('%s/%d*%d_%s' % (tarPath, size, size, imgName), mat)
  #cv2_imshow(mat)
  return mat

"""**A.d Non-maximal Suppression**"""

def nonMaximalSupression(img, oriImg, threshold, minDis):
  h, w = img.shape
  point = [] # eigenvalue of canditate points
  coord = [] # coordinate of canditate points
  resultC = [] # detected corners

  # traverse img
  for i in range(h):
    for j in range(w):
      if(img[i, j] > threshold): # if eigenvalue of point[i, j] > threshold, add point[i, j] into canditate list
        coord.append([i, j])
        point.append(img[i, j])

  while coord != []: # not stop until canditate list is empty
    maxP = max(point) # select point with the largest eigenvalue among all canditate points
    maxY, maxX = coord[point.index(maxP)] # get the coordinate of selected point
    resultC.append([maxY, maxX]) # add selected point into list of detected corner
    
    # remove selected point from canditate list
    point.remove(maxP) 
    coord.remove([maxY, maxX])
    removeP = []
    removeC = []
    for i in range(len(point)): # traverse the canditate list
      tmpY, tmpX = coord[i]
      if (maxY - tmpY)**2 + (maxX - tmpX)**2 < minDis: # remove point[tmpY, tmpX] if it is too close to the selected point
        removeP.append(point[i])
        removeC.append(coord[i])
    for i in range(len(removeP)):
      point.remove(removeP[i])
      coord.remove(removeC[i])

  for y, x in resultC: # draw all corner points on img
    cv2.circle(oriImg, (x, y), 3, (0, 0, 255), 5)
  
  return oriImg

name = '1a_notredame.jpg'
ori = cv2.imread('/content/drive/MyDrive/CV_HW1/A/original/%s' % name)
tarPath = '/content/drive/MyDrive/CV_HW1/A/result_normal/chessboard'
gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
smooth10 = gaussianSmoothing(gray, 10, write=1, tarPath=tarPath, imgName='smooth_size10_%s' % name)
smooth5 = gaussianSmoothing(gray, 5, write=1, tarPath=tarPath, imgName='smooth_size5_%s' % name)
ix10, iy10, mag10, dir10 = intensityGradient(smooth10, 12, write=1, tarPath=tarPath, imgName='size10_%s' %name)
ix5, iy5, mag5, dir5 = intensityGradient(smooth5, 20, write=1, tarPath=tarPath, imgName='size5_%s' % name)
eig3 = structureTensor(smooth10, 3, write=1, tarPath=tarPath, imgName='eigenvalue_%s' % name)
eig5 = structureTensor(smooth10, 5, write=1, tarPath=tarPath, imgName='eigenvalue_%s' % name)

h, w = ori.shape[:2]
pad = 5
ori = ori[pad:h-pad, pad:w-pad]
result5 = nonMaximalSupression(eig5, rot, 15000, 400)

"""**B. Experiments**"""

name = 'chessboard-hw1.jpg'
ori = cv2.imread('/content/drive/MyDrive/CV_HW1/A/original/%s' % name)
tarPath = '/content/drive/MyDrive/CV_HW1/A/result_rotate/chessboard'
t = 'rotate'

height, width = ori.shape[:2]
center = (width/2, height/2)
rot = cv2.getRotationMatrix2D(center=center, angle=30, scale=1)
rot = cv2.warpAffine(src=ori, M=rot, dsize=(width, height))
gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

smooth10 = gaussianSmoothing(gray, 10, write=1, tarPath=tarPath, imgName='%s_smooth_size10_%s' % (t, name))
smooth5 = gaussianSmoothing(gray, 5, write=1, tarPath=tarPath, imgName='%s_smooth_size5_%s' % (t, name))
ix10, iy10, mag10, dir10 = intensityGradient(smooth10, 12, write=1, tarPath=tarPath, imgName='%s_size10_%s' % (t, name))
ix5, iy5, mag5, dir5 = intensityGradient(smooth5, 20, write=1, tarPath=tarPath, imgName='%s_size5_%s' % (t, name))
eig3 = structureTensor(smooth10, 3, write=1, tarPath=tarPath, imgName='%s_eigenvalue_%s' % (t, name))
eig5 = structureTensor(smooth10, 5, write=1, tarPath=tarPath, imgName='%s_eigenvalue_%s' % (t, name))

ori = cv2.imread('/content/drive/MyDrive/CV_HW1/A/original/%s' % name)
pad = 5
height, width = ori.shape[:2]
center = (width/2, height/2)
rot = cv2.getRotationMatrix2D(center=center, angle=30, scale=1)
rot = cv2.warpAffine(src=ori, M=rot, dsize=(width, height))

h, w = rot.shape[:2]
rot = rot[pad:h-pad, pad:w-pad]

result = nonMaximalSupression(eig5, rot, 15000, 400)

cv2.imwrite('%s/result_%s_5*5_%s' % (tarPath, t, name), result)