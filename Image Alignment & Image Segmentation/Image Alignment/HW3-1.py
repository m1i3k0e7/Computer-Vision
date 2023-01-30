# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import random

from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')

def interest_point_detection(img1, img2):
  gray1, gray2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  # create SIFT detector
  sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=7, contrastThreshold=0.005, edgeThreshold=40, sigma=2.1)
  # detect
  keypoints1, features1 = sift.detectAndCompute(gray1, None)
  keypoints2, features2 = sift.detectAndCompute(gray2, None)
  # draw keypoints
  detect1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints1, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS, color=(0, 0, 255))
  detect2 = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=keypoints2, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS, color=(255, 0, 0))
  # concatenate two images
  h, w = detect2.shape[0], detect1.shape[1]
  black = np.zeros((h, w, 3))
  black[0:detect1.shape[0], 0:w] = detect1
  concat = np.hstack([black, detect2])
  return concat, keypoints1, features1, keypoints2, features2

def twoNN_feature_matching(img1, img2, kps1, kps2, fea1, fea2, threshold, ratio):
  idxMatching = [[np.inf, -1, -1]] * len(fea2) #idxMatching[i] = [difference between kpi and kpj, kpi, kpj]
  for i in range(len(fea1)):
    # two canditates for 2 nearest neighbor algorithm
    minDiff = [-1, np.inf]
    secMinDiff = [-1, np.inf]
    for j in range(len(fea2)):
      diff = cv2.norm(fea1[i], fea2[j], cv2.NORM_L2)
      if diff < minDiff[1]: # update minDiff if keypoints2[j] has shorter distance from keypoints1[i]
        secMinDiff = minDiff
        minDiff = [j, diff]
      elif diff < secMinDiff[1]: # update secMinDiff
        secMinDiff = [j, diff]

    # use 2NN method to check whether kps1[i] and kps2[minDiff] can be matched and # filter out poorly related pairs with threshold
    if minDiff[1] <= secMinDiff[1] * ratio and minDiff[1] < threshold:
      if idxMatching[minDiff[0]][0] > minDiff[1]:
        idxMatching[minDiff[0]] = [minDiff[1], i, minDiff[0]]
  
  # get coordinates of matched interest points and draw them
  pointMatching = []
  for (diff, idA, idB) in idxMatching:
    if diff < np.inf:
      pointA = (int(kps1[idA].pt[0]), int(kps1[idA].pt[1]))
      pointB = (int(kps2[idB].pt[0]), int(kps2[idB].pt[1]))
      pointMatching.append([pointA, pointB])

  print('number of correspondences: %d' % len(pointMatching))
  h, w = img2.shape[0], img1.shape[1]
  black = np.zeros((h, w, 3))
  black[0:img1.shape[0], 0:w] = img1
  concat = np.hstack([black, img2])
  for (pointA, pointB) in pointMatching:
    cv2.line(concat, pointA, (pointB[0] + w, pointB[1]), (0, 255, 0), 1)

  return concat, pointMatching

def RANSAC(img1, img2, points, iterations, threshold, subsetSize):
  import copy
  n = len(points)
  matchings = [(point[0][0], point[0][1], point[1][0], point[1][1]) for point in points] # all correspondences
  bestInliers = []
  for i in range(iterations): # randomly select a subset for estimating homography for some iterations
    randomID = random.sample(range(n), subsetSize)
    inliers = []
    A = []
    # form the selected subset to a matrix A
    for p in randomID:
      x, y, u, v = matchings[p]
      A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
      A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    # compute homography from A
    u, s, v = np.linalg.svd(A)
    H = np.reshape(v[8], (3, 3))
    H = H / H.item(8)

    # count inlier for best estimation of homography
    for pt in matchings:
      x, y, u, v = pt
      p, q, z = H @ np.array([x, y, 1])
      p, q = p/z, q/z
      if (p - u)**2 + (q - v)**2 < threshold:
        inliers.append([x, y, u, v])
    
    # update best homography
    if len(inliers) > len(bestInliers):
      bestInliers = inliers
  
  # use best inliers set to re-estimate homography
  A = []
  for p in bestInliers:
    x, y, u, v = p
    A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
  
  u, s, v = np.linalg.svd(A)
  H = np.reshape(v[8], (3, 3))
  H = H / H.item(8)
  h, w = img2.shape[0], img1.shape[1]
  black = np.zeros((h, w, 3))
  black[0:img1.shape[0], 0:w] = img1
  concat = np.hstack([black, img2])
  mean_length = 0
  tmp2 = copy.deepcopy(img2)

  # draw correspondence matching
  for p in bestInliers:
    x, y = p[:2]
    u, v, z = H @ np.array([x, y, 1])
    u, v = int(u / z), int(v / z)
    cv2.line(concat, (x, y), (w + u, v), (0, 255, 0), 1)
    mean_length += ((u - p[2])**2 + (v - p[3])**2)**0.5
    cv2.arrowedLine(tmp2, (u, v), (p[2], p[3]), (0, 255, 0), 1, tipLength=0.05)
  print('mean length of deviation vectors (inliers): %f' % (mean_length / len(bestInliers)))

  # draw deviation vector
  mean_length = 0
  for pt in matchings:
    x, y, u, v = pt
    p, q, z = H @ np.array([x, y, 1])
    p, q = int(p / z), int(q / z)
    cv2.arrowedLine(img2, (p, q), (u, v), (0, 255, 0), 1, tipLength=0.01)
    mean_length += ((p - u)**2 + (q - v)**2)**0.5
  print('mean length of deviation vectors (all matching points): %f' % (mean_length / len(matchings)))

  return H, concat, bestInliers, img2, tmp2

def P1_A(img1, img2, threshold, ratio):
  import copy
  tmp1 = copy.deepcopy(img1)
  tmp2 = copy.deepcopy(img2)
  detect, kps1, fea1, kps2, fea2 = interest_point_detection(tmp1, tmp2)
  matching, points = twoNN_feature_matching(img1, img2, kps1, kps2, fea1, fea2, threshold, ratio)

  return detect, matching, points

def draw_edges(img1, img2, H, lt, lb, rt, rb):
  cv2.line(img1, tuple(lt[:2]), tuple(lb[:2]), (0,0,255), 2)
  cv2.line(img1, tuple(lb[:2]), tuple(rb[:2]), (0,0,255), 2)
  cv2.line(img1, tuple(rb[:2]), tuple(rt[:2]), (0,0,255), 2)
  cv2.line(img1, tuple(rt[:2]), tuple(lt[:2]), (0,0,255), 2)
  lt = H @ np.array(lt)
  lt /= lt[2]; lt[0] += img2.shape[1]
  lb = H @ (lb)
  lb /= lb[2]; lb[0] += img2.shape[1]
  rt = H @ (rt)
  rt /= rt[2]; rt[0] += img2.shape[1]
  rb = H @ (rb)
  rb /= rb[2]; rb[0] += img2.shape[1]
  cv2.line(img1, (int(lt[0]), int(lt[1])), (int(lb[0]), int(lb[1])), (0,0,255), 2)
  cv2.line(img1, (int(lb[0]), int(lb[1])), (int(rb[0]), int(rb[1])), (0,0,255), 2)
  cv2.line(img1, (int(rb[0]), int(rb[1])), (int(rt[0]), int(rt[1])), (0,0,255), 2)
  cv2.line(img1, (int(rt[0]), int(rt[1])), (int(lt[0]), int(lt[1])), (0,0,255), 2)

path = '/content/drive/MyDrive/HW3/'
''' problem1 '''
threshold = 350
ratio = 0.85
# book1
img1 = cv2.imread(path + '1-book1.jpg')
img2 = cv2.imread(path + '1-image.jpg')
d1, m1, p1 = P1_A(img1, img2, threshold, ratio)
cv2.imwrite('./1-book1-detect.jpg', d1)
cv2.imwrite('./1-book1-matching.jpg', m1)
# book2
img1 = cv2.imread(path + '1-book2.jpg')
img2 = cv2.imread(path + '1-image.jpg')
d2, m2, p2 = P1_A(img1, img2, threshold, ratio)
cv2.imwrite('./1-book2-detect.jpg', d2)
cv2.imwrite('./1-book2-matching.jpg', m2)
# book3
img1 = cv2.imread(path + '1-book3.jpg')
img2 = cv2.imread(path + '1-image.jpg')
d3, m3, p3 = P1_A(img1, img2, threshold, ratio)
cv2.imwrite('./1-book3-detect.jpg', d3)
cv2.imwrite('./1-book3-matching.jpg', m3)

''' problem2 '''
iterations = 1000
threshold = 18
subsetSize = 4
# book1
# estimating inliers
img1 = cv2.imread(path + '1-image.jpg')
img2 = cv2.imread(path + '1-book1.jpg')
H1, r1, i1, d1, t1 = RANSAC(img2, img1, p1, iterations, threshold, subsetSize)
draw_edges(r1, img2, H1, [16, 50, 1], [11, 319, 1], [424, 42, 1], [434, 314, 1])
cv2.imwrite('./1-book1-inliers.jpg', r1)
cv2.imwrite('./1-book1-deviation_vector(all).jpg', d1)
cv2.imwrite('./1-book1-deviation_vector(inlier).jpg', t1)

# book2
img1 = cv2.imread(path + '1-image.jpg')
img2 = cv2.imread(path + '1-book2.jpg')
H2, r2, i2, d2, t2 = RANSAC(img2, img1, p2, iterations, threshold, subsetSize)
draw_edges(r2, img2, H2, [29, 13, 1], [22, 318, 1], [410, 11, 1], [426, 318, 1])
cv2.imwrite('./1-book2-inliers.jpg', r2)
cv2.imwrite('./1-book2-deviation_vector(all).jpg', d2)
cv2.imwrite('./1-book2-deviation_vector(inlier).jpg', t2)

# book3
img1 = cv2.imread(path + '1-image.jpg')
img2 = cv2.imread(path + '1-book3.jpg')
H3, r3, i3, d3, t3 = RANSAC(img2, img1, p3, iterations, threshold, subsetSize)
draw_edges(r3, img2, H3, [24, 29, 1], [24, 301, 1], [418, 30, 1], [424, 295, 1])
cv2.imwrite('./1-book3-inliers.jpg', r3)
cv2.imwrite('./1-book3-deviation_vector(all).jpg', d3)
cv2.imwrite('./1-book3-deviation_vector(inlier).jpg', t3)