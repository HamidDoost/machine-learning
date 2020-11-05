'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 01/11/2020
-- Description:	This code is for Colour Histogram test (prediction).
-- Status:      In progress
===============================================================================
'''
from sklearn.svm import LinearSVC
from imutils import paths
import imutils
import numpy as np
import cv2
import os
import sys
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils.paths import list_images
import pickle

index = {}
label = []
data = []
filename = "GCLM_BR_BS_D.sav"
load_model = pickle.load(open(filename, 'rb'))


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


imagePaths = list(paths.list_images(sys.argv[1]))

data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (200, 200))

    filename = imagePath.split(os.path.sep)[-1]
    label = filename.split("_")[0]
    #label = imagePath.split(os.path.sep)[-2].split("/")[0]
    # print(label)
    hist = extract_color_histogram(image)
    hist = (hist.reshape(1, -1))
    prediction = load_model.predict(hist)
    print(label, prediction[0])
