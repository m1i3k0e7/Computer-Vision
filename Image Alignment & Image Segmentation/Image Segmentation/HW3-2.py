# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')

def kmeans(img, k, threshold):
  h, w = img.shape[:2]
  points = [[y, x, img[y][x]] for y in range(h) for x in range(w)] # all points
  n = len(points)
  centers = []
  while len(centers) < k: # randomly select k nonrepeat points
    randomCenter = list(points[random.sample(range(n), 1)[0]][2])
    if randomCenter not in centers:
      centers.append(randomCenter)

  convergeClusters = []
  converge = False
  while not converge: # executing until all of k centers converge
    converge = True
    clusters = {id:[] for id in range(k)} # k clusters
    for y in range(h):
      for x in range(w):
        r1, g1, b1 = img[y][x]
        minDistance = np.inf
        c = -1
        for i in range(k):
          r2, g2, b2 = centers[i]
          d = (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2 # euclidean instance in RGB space
          if d < minDistance: # choose the center whose distance to points[y][x] is smallest
            minDistance = d
            c = i
        clusters[c].append([y, x, r1, g1, b1]) # append point[y][x] into one of k clusters

    for i in range(k):
      if len(clusters[i]) > 0:
        rMean, gMean, bMean = np.mean(clusters[i], axis=0)[2:]
        if abs(rMean - centers[i][0]) > threshold or abs(gMean - centers[i][1]) > threshold or abs(bMean - centers[i][2]) > threshold:
          # check whether each center converge
          converge = False
        centers[i] = [rMean, gMean, bMean] # update k centers
    convergeClusters = clusters # update converged clusters

  # processing segmented image
  seg = np.zeros((h, w, 3))
  distance = 0
  for i in range(k):
    r, g, b = centers[i]
    for j in range(len(convergeClusters[i])):
      y, x = convergeClusters[i][j][:2]
      seg[y][x] = centers[i]
      r2, g2, b2 = convergeClusters[i][j][2:]
      distance += (r - r2)**2 + (g - g2)**2 + (b - b2)**2
    
  return distance, seg

def kmeansplusplus(img, k, threshold):
  h, w = img.shape[:2]
  points = [[y, x, img[y][x]] for y in range(h) for x in range(w)]
  n = len(points)
  randomID = random.sample(range(n), k)
  centers = []
  fr, fg, fb = list(points[random.sample(range(n), k=1)[0]][2])
  centers.append([fr, fg, fb])
  weight = [((img[y][x][0]-fr)**2 + (img[y][x][1]-fg)**2 + (img[y][x][2]-fb)**2)**0.5 for y in range(h) for x in range(w)]
  while len(centers) < k:
    # randomly select k centers according to distance from each point to selected centers.
    randomCenter = list(points[random.choices(range(n), weights=weight, k=1)[0]][2])
    if randomCenter not in centers:
      centers.append(randomCenter)
      for i in range(n):
        r1, g1, b1 = points[i][2]
        r2, g2, b2 = randomCenter
        weight[i] += math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)

  # same as kmeans(img, k, threshold)
  convergeClusters = []
  converge = False
  while not converge:
    converge = True
    clusters = {id:[] for id in range(k)}
    for y in range(h):
      for x in range(w):
        r1, g1, b1 = img[y][x]
        minDistance = np.inf
        c = -1
        for i in range(k):
          r2, g2, b2 = centers[i]
          d = (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2
          if d < minDistance:
            minDistance = d
            c = i
        clusters[c].append([y, x, r1, g1, b1])

    for i in range(k):
      if len(clusters[i]) > 0:
        rMean, gMean, bMean = np.mean(clusters[i], axis=0)[2:]
        if abs(rMean - centers[i][0]) > threshold or abs(gMean - centers[i][1]) > threshold or abs(bMean - centers[i][2]) > threshold:
          converge = False
        centers[i] = [rMean, gMean, bMean]
    convergeClusters = clusters

  seg = np.zeros((h, w, 3))
  distance = 0
  for i in range(k):
    r, g, b = centers[i]
    for j in range(len(convergeClusters[i])):
      y, x = convergeClusters[i][j][:2]
      seg[y][x] = centers[i]
      r2, g2, b2 = convergeClusters[i][j][2:]
      distance += (r - r2)**2 + (g - g2)**2 + (b - b2)**2
    
  return distance, seg

def K_means(img, k, threshold, iterations):
  bestSegmentation = []
  minDistance = np.inf
  for i in range(iterations):
    print("iteration %d" % i)
    d, s = kmeans(img, k, threshold)
    print("mean error: %d" % (d / (img.shape[0] * img.shape[1])))
    if d < minDistance:
      # choose best of 50 iterations
      bestSegmentation = s
      minDistance = d
  return s

def K_means_plus_plus(img, k, threshold, iterations):
  bestSegmentation = []
  minDistance = np.inf
  for i in range(iterations):
    print("iteration %d" % i)
    d, s = kmeansplusplus(img, k, threshold)
    print("mean error: %d" % (d / (img.shape[0] * img.shape[1])))
    if d < minDistance: # choose best of 50 iterations
      bestSegmentation = s
      minDistance = d
  return s

def mean_shift(img, threshold, bandwidth):
  h, w = img.shape[:2]
  points = tf.constant([img[y][x] for y in range(h) for x in range(w)], dtype=tf.float64) # RGB information of all points
  img2 = tf.Variable(img, dtype=tf.float64) # segmented image
  i = 1
  while True: # executing until all points converges
    print('iteration %d' % i)
    max_gradient = -np.inf
    for y in range(h):
      for x in range(w):        
        offset = points - img2[y, x] # offset = {[img[y][x][0]-point[0], img[y][x][1]-point[1], img[y][x][2]-point[2]] | point in img}
        distance = tf.norm(offset, axis=1) # distance = {root mean square of offset[i] | i = 1, 2, ... len(offset)}
        index = tf.where(distance < bandwidth) # index = {i | distance[i] < bandwidth}
        in_range = tf.gather_nd(offset, index) # in_range = {offset[i] | i in index}
        gradient = tf.reduce_mean(in_range, 0) # gradient = {mean of offset[i][j] | i = 1, 2, 3; j = 1, 2, ..., len(in_range)}
        img2 = img2[y, x].assign(img2[y, x] + gradient) # shift point to mean of all points in range of bandwidth
        max_gradient = max(tf.norm(gradient), max_gradient)
    if max_gradient < threshold:
      break # converge
    points = tf.constant([img2[y][x] for y in range(h) for x in range(w)], dtype=tf.float64) # update image

  return img2

def mean_shift_spatial(img, color_threshold, spatial_threshold, color_bandwidth, spatial_bandwidth):
  h, w = img.shape[:2]
  points = tf.constant([img[y][x] for y in range(h) for x in range(w)], dtype=tf.float64)
  coords = tf.constant([[y, x] for y in range(h) for x in range(w)], dtype=tf.float64) # spatial information of all points
  p = tf.Variable(img, dtype=tf.float64)
  i = 1
  while True:
    print('iteration %d' % i)
    coords_tmp = []
    max_color_gradient, max_spatial_gradient = -np.inf, -np.inf
    for y in range(h):
      for x in range(w):
        cur_y, cur_x = coords[y * h + x]
        color_offset = points - p[y, x]
        spatial_offset = coords - tf.constant([cur_y, cur_x], dtype=tf.float64)
        color_distance = tf.norm(color_offset, axis=1)
        spatial_distance = tf.norm(spatial_offset, axis=1)
        i1 = tf.where(color_distance < color_bandwidth); i2 = tf.where(spatial_distance < spatial_bandwidth)
        # index = {i | color_distance[i] < color_bandwidth & spatial_distance[i] < spatial_bandwidth}
        index = tf.concat([i1, i2], 0); index, id = tf.raw_ops.UniqueV2(x=index, axis=[0], out_idx=tf.int64) 
        color_in_range = tf.gather_nd(color_offset, index)
        color_gradient = tf.reduce_mean(color_in_range, 0)
        spatial_in_range = tf.gather_nd(spatial_offset, index)
        spatial_gradient = tf.reduce_mean(spatial_in_range, 0)
        p = p[y, x].assign(p[y, x] + color_gradient)
        cur_y, cur_x = float(cur_y + spatial_gradient[0]), float(cur_x + spatial_gradient[1])
        coords_tmp.append([cur_y, cur_x])
        max_color_gradient = max(max_color_gradient, tf.norm(color_gradient))
        max_spatial_gradient = max(max_spatial_gradient, tf.norm(spatial_gradient))

    if max_color_gradient < color_threshold and max_spatial_gradient < spatial_threshold:
      break # converge
    points = tf.constant([p[y][x] for y in range(h) for x in range(w)], dtype=tf.float64)    
    coords = tf.constant(coords_tmp, dtype=tf.float64)

  return p

def pixel_distribution(img1, img2):
  h, w = img1.shape[:2]
  points1 = [img1[y][x] for y in range(h) for x in range(w)]
  points2 = [img2[y][x] for y in range(h) for x in range(w)]
  ax1 = plt.axes(projection='3d')
  ax1.set_xlim(0, 255); ax1.set_ylim(0, 255); ax1.set_zlim(0, 255)
  ax2 = plt.axes(projection='3d')
  ax2.set_xlim(0, 255); ax1.set_ylim(0, 255); ax1.set_zlim(0, 255)
  for p1, p2 in zip(points1, points2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    ax1.scatter(x1, y1, z1, c=[(z1/255, y1/255, x1/255)], marker='.')
    ax2.scatter(x1, y1, z1, c=[(z2/255, y2/255, x2/255)], marker='.')
  return ax1, ax2

path = '/content/drive/MyDrive/HW3/'

# 2-image
img = cv2.imread(path + '2-image.jpg')
segmentation = K_means(img, 3, 5, 50)
cv2.imwrite('3-means_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
segmentation = K_means(img, 5, 5, 50)
cv2.imwrite('5-means_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
segmentation = K_means(img, 7, 5, 50)
cv2.imwrite('7-means_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
segmentation = K_means_plus_plus(img, 3, 5, 50)
cv2.imwrite('3-means++_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
segmentation = K_means_plus_plus(img, 5, 5, 50)
cv2.imwrite('5-means++_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
segmentation = K_means_plus_plus(img, 7, 5, 50)
cv2.imwrite('7-means++_2-image.jpg', segmentation)

img = cv2.imread(path + '2-image.jpg')
meanshift1 = mean_shift(img, 1, 40)
cv2.imwrite('meanshift_h=40_2-image.jpg', meanshift1)
meanshift2 = mean_shift(img, 1, 80)
cv2.imwrite('meanshift_h=80_2-image.jpg', meanshift2)
meanshift3 = mean_shift(img, 1, 100)
cv2.imwrite('meanshift_h=100_2-image.jpg', meanshift3)
meanshift_spatial = mean_shift_spatial(img, 1, 1, 30, 15)
cv2.imwrite('meanshift_spatial_2-image.jpg', meanshift_spatial)

img1 = cv2.imread(path + '2-image_2-image.jpg')
img2 = cv2.imread(path + 'meanshift_h=40_2-image.jpg')
ax1, ax2 = pixel_distribution(img1, img2)


# 2-masterpiece
img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means(img, 3, 5, 50)
cv2.imwrite('3-means_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means(img, 5, 5, 50)
cv2.imwrite('5-means_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means(img, 7, 5, 50)
cv2.imwrite('7-means_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means_plus_plus(img, 3, 5, 50)
cv2.imwrite('3-means++_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means_plus_plus(img, 5, 5, 50)
cv2.imwrite('5-means++_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
segmentation = K_means_plus_plus(img, 7, 5, 50)
cv2.imwrite('7-means++_2-masterpiece.jpg', segmentation)

img = cv2.imread(path + '2-masterpiece.jpg')
meanshift1 = mean_shift(img, 1, 40)
cv2.imwrite('meanshift_h=40_2-masterpiece.jpg', meanshift1)
meanshift2 = mean_shift(img, 1, 80)
cv2.imwrite('meanshift_h=80_2-masterpiece.jpg', meanshift2)
meanshift3 = mean_shift(img, 1, 100)
cv2.imwrite('meanshift_h=100_2-masterpiece.jpg', meanshift3)
meanshift_spatial = mean_shift_spatial(img, 1, 1, 30, 15)
cv2.imwrite('meanshift_spatial_2-image.jpg', meanshift_spatial)

img1 = cv2.imread(path + '2-image_2-masterpiece.jpg')
img2 = cv2.imread(path + 'meanshift_h=40_2-masterpiece.jpg')
ax1, ax2 = pixel_distribution(img1, img2)