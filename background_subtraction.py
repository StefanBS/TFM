from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import tarfile
import cv2
from skimage import measure
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

#folder = 'DaimlerBenchmark/Data/TestData/NonPedestrians'
#folder = 'test_images'
#image_files = os.listdir(folder)
#print(image_files[0])
#image_path = os.path.join(folder, image_files[1])
#image = mpimg.imread(image_path)

cv2.namedWindow('mask')
cv2.namedWindow('output')

folder = 'PETS2009/L1-View3'
image_files = sorted(os.listdir(folder))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
# kernel = np.ones((5, 5), np.uint8)

# backsub = cv2.BackgroundSubtractorMOG2(200, 16, 127)
# backsub = cv2.BackgroundSubtractorMOG2(200, 16, 127)
backsub = cv2.BackgroundSubtractorMOG2()
for filename in image_files:
    image_path = os.path.join(folder, filename)
    image = cv2.imread(image_path)

    fgmask = backsub.apply(image, None, 0.01)
    # fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    # fgmask = cv2.erode(fgmask, kernel, iterations=1)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_out = image.copy()

    #cv2.drawContours(image, contours, -1, (255,255,255), -1)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if (h > w):# & (h > 35) & (w > 5):
            cv2.rectangle(image_out, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow('mask', fgmask)
    cv2.imshow('output', image_out)
    cv2.waitKey(1)
#image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
#plt.imshow(image)
#plt.show()