
'''
===============================================================================
-- Author:      Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 01/11/2020
-- Description:	This code is for HOG feature train.
-- Status:      In progress
===============================================================================
'''
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.calibration import CalibratedClassifierCV

from sklearn.neighbors import KNeighborsClassifier
from imutils import paths

import numpy as np
import pickle
import cv2
import os
import sys
from skimage import feature
from skimage import exposure
from imutils.paths import list_images

modelfile = 'HOG.sav'
data = []
labels = []

for imagePath in paths.list_images(sys.argv[1]):

    image = cv2.imread(imagePath)
    image = cv2.resize(image, (100, 100)).astype("float32")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filename = imagePath.split(os.path.sep)[-1]
    label = (imagePath.split(os.path.sep)[-1]).split("_")[0]
    # k=label.split("_")[0]
    # labels.append(imagePath.split(os.path.sep)[-2])
    k = label
    print(k)
    labels.append(k)
    sample = gray
    (H, hogImage) = feature.hog(sample, orientations=9, pixels_per_cell=(10, 10),
                                cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)

# H=H.flatten()
# H=H.reshape(1,-1)
    data.append(H)
'''knn
model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)
'''
pickle.dump(model, open(modelfile, 'wb'))

'''linearSVM
model = LinearSVC(C=100.0, random_state=42)
model = SVC()
'''

'''calibratedSVM
calibrated=CalibratedClassifierCV(base_estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=4, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False),     cv=4, method='sigmoid')
calibrated.fit(data, labels)
'''
calibrated.save("hog.model")
modelfile = 'hog.sav'
pickle.dump(model, open(modelfile, 'wb'))
