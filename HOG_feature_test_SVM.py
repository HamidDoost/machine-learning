'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 01/11/2020
-- Description:	This code is for HOG feature test (prediction).
-- Status:      In progress
===============================================================================
'''
from sklearn.svm import LinearSVC
from imutils import paths
import argparse

from collections import deque

import numpy as np
import pickle
import cv2
import glob
import os
import sys

from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature

from imutils.paths import list_images
import pickle

index = {}
label = []
data = []
filename = 'HOG.sav'

load_model = pickle.load(open(filename, 'rb'))
data = []
labels = []

for imagePath in paths.list_images(sys.argv[1]):

    image = cv2.imread(imagePath)
    ho, wo = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sample = cv2.resize(gray, (100, 100))
    (H, hogImage) = feature.hog(sample, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    H = (H.reshape(1, -1))
    prediction = load_model.predict(H)
    #      cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
    #         1.0, (0, 0, 255), 3)
    #      cv2.imshow("Image", image)
    #      cv2.waitKey(0)
    print(imagePath, prediction[0])
