# Dependencies and usage
## 1. For local usage:

(1) Dependencies
- python
- opencv
- numpy
- matplotlib
- tensorflow
- mpl_toolkits

(2) Delete following instructions in HW3-1.py and HW3-2.py:
```python
from google.colab.patches import cv2_imshow
from google.colab import drive
drive.mount('/content/drive')
```
(3) Build a directory and follow the structure shown below.

    HW3/
     ├ 1-book1.jpg
     ├ 1-book2.jpg
     ├ 1-book3.jpg
     ├ 1-image.jpg
     ├ 2-image.jpg
     ├ 2-masterpiece.jpg

(4) Replace the value of path in HW3-1.py and HW3-2.py with the path to HW3

(5) Image alignment with RANSAC
```
cd path\to\HW3_108034022\1
python HW3-1.py
```
(6) K-means and mean-shift
```
cd path\to\HW3_108034022\2
python HW3-2.py
```

## 2. Usage in Google Colab
(1) Copy the codes in HW3-1.py and HW3-2.py to Google Colab and build the same directory as mentioned in step 1.(3) in your Google drive