# Part 1. Fundamental Matrix Estimation from Point Correspondences
## (a) linear least-squares eight-point algorithm
There are totally three functions in this part: 

(1) **getPts(path):**: Input the path of pt_2D.txt as path, this function will return a 2D array containing the coordinates of all points in pt_2D.txt.
```python
def getPts(path):
    points = []
    fh = open(path)
    p = fh.readlines()
    for point in p:
        coord = point.split()
        points.append([float(coord[0]), float(coord[1]), 1])
    return points
```
(2) **LeastSquare8Point(points, n)**: Input an array points whose entry is equal to [u, v, u', v'] and the length of points, this function will use them to create a nx9 matrix
```python
A = np.zeros((n, 9))
for i in range(n):
    A[i][0] = points[i][0] * points[i][2]
    A[i][1] = points[i][0] * points[i][3]
    A[i][2] = points[i][0]
    A[i][3] = points[i][1] * points[i][2]
    A[i][4] = points[i][1] * points[i][3]
    A[i][5] = points[i][1]
    A[i][6] = points[i][2]
    A[i][7] = points[i][3]
    A[i][8] = 1
```
 Then, return the fundamental matrix computed from all (u, v) and (u', v') stored in points.
```python
u, s, v = np.linalg.svd(A)
f = v[-1].reshape(3, 3)
u, s, v = np.linalg.svd(f)
s[2] = 0
f = u @ np.diag(s) @ v

return f
```
(3) **drawPointsAndLines(img1, img2, points, f)**: Input two image img1 and img2, coordinates of specific points on img1 and img2 and the fundamental matrix generated by  **LeastSquare8Point**, this function will draw the corresponding epipolar lines of points (u, v) on img1 and points (u', v') on img2, respectively, on img2 and img1. To draw a line ax + by + c = 0 on an image, this function will firstly position two points (0, -c1/b1) and (w, -(aw+c)/b), where w is the width of the image, then, it will connect two points to draw a line and measure the distance between line and corresponding point.
```python
distance = 0
h, w = img1.shape[:2]
for x1, y1, x2, y2 in points:
    a1, b1, c1 = f.transpose() @ np.array([x1, y1, 1])
    a2, b2, c2 = f @ np.array([x2, y2, 1])
    distance += abs(a1*x2 + b1*y2 + c1) / ((a1**2 + b1**2)**0.5)
    distance += abs(a2*x1 + b2*y1 + c2) / ((a2**2 + b2**2)**0.5)
    p1, p2 = (0, int(-c1/b1)), (w, int(-(a1*w + c1)/b1))
    p3, p4 = (0, int(-c2/b2)), (w, int(-(a2*w + c2)/b2))
    cv2.line(img1, p3, p4, (0, 0, 255), 1)
    cv2.line(img2, p1, p2, (0, 0, 255), 1)
```
## (b) normalized eight-point algorithm
This part contains one more function than part (a):

(1) **normalize(p)**: Input the same array as points in **LeastSquare8Point(points, n)**, this function will use the means and standard deviations of x and y coordinates to form two 3x3 matrix T1 and T2, using T1, T2 to normalize points on two images, then, returning T1, T2 and all normalized points.
```python
def normalize(p):
    n = len(p)
    p1, p2 = [[], []], [[], []]
    points = []
    for a,b,c,d in p:
        p1[0].append(a)
        p1[1].append(b)
        p2[0].append(c)
        p2[1].append(d)
        
    mean1 = np.mean(p1, axis=1)
    S1 = np.sqrt(2) / np.std(p1[0])
    S2 = np.sqrt(2) / np.std(p1[1])
    T1 = np.array([[S1, 0, -S1 * mean1[0]],
            [0, S2, -S2 * mean1[1]],
            [0, 0, 1]])
        
    mean2 = np.mean(p2, axis=1)
    S1 = np.sqrt(2) / np.std(p2[0])
    S2 = np.sqrt(2) / np.std(p2[1])
    T2 = np.array([[S1, 0, -S1 * mean2[0]],
            [0, S2, -S2 * mean2[1]],
            [0, 0, 1]])
    for a,b,c,d in p:
        x1,y1,z1 = T1 @ np.array([a, b, 1])
        x2,y2,z2 = T2 @ np.array([c, d, 1])
        points.append([x1/z1, y1/z1, x2/z2, y2/z2])

    return points, T1, T2
```
# Part 2. Homography transform
## A. Homography matrix

there is one function in this part:

**Homography(oriPts, tarPts, iter)**: Input the x and y coordinates of source points and target points as **oriPts**, **tarPts** and the number of iterations we want to use to compute the homography, this function will randomly select four pairs of correspondences from oriPts and tarPts to compute the homography matrix for **iter** iterations. 
```python
for it in range(iter):
    A = []
    for i in randomID:
    A.append([oriPts[i][0], oriPts[i][1], 1, 0, 0, 0, -oriPts[i][0]*tarPts[i][0], -oriPts[i][1]*tarPts[i][0], -tarPts[i][0]])
    A.append([0, 0, 0, oriPts[i][0], oriPts[i][1], 1, -oriPts[i][0]*tarPts[i][1], -oriPts[i][1]*tarPts[i][1], -tarPts[i][1]])
    u, s, v = np.linalg.svd(A)
    H = np.reshape(v[8], (3, 3))
    H = H / H.item(8)
```
The homography matrix leading to the least distance between transformed points and target points will be returned.
```python
for it in range(iter):
    square = 0
    randomID = random.sample(range(n), 4)
    for i in range(n):
        x1, y1 = tarPts[i]
        x2, y2, z2 = H @ np.array([oriPts[i][0], oriPts[i][1], 1])
        x2, y2 = x2/z2, y2/z2
        square += (x1 - x2)**2 + (y1 - y2)**2

    if square < minSquare:
        minSquare = square
        resultH = H
        pointSet = randomID
```
## B. Image retification

There are one function in this part: 

**getImage(img, H)**: Input the original image and the homography matrix returned from **Homography**, this function will compute the corresponding coordinate to fetch the pixel value and use bilinear interpolation to compute the appropriate pixel value for each point on the transformed image. More details are illustrated in report.pdf.
```python
for i in range(h):
    for j in range(w):
        x, y, z = H @ np.array([j, i, 1])
        x, y = x/z, y/z
        x1, y1 = math.floor(x), math.floor(y)
        x2, y2 = math.ceil(x), math.ceil(y)

        if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
            continue

        a, b = x - x1, y - y1
        p = ((1-a) * (1-b) * img[y1, x1]) + (a * (1-b) * img[y1, x2]) + (b * (1-a) * img[y2, x1]) + (a * b * img[y2, x2])
            
        newImage[i, j] = p
```