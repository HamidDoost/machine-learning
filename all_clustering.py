'''
===============================================================================
-- Author:		Hamid Doostmohammadi, Azadeh Nazemi
-- Create date: 01/11/2020
-- Description:	This code is for ten clusterig methods.
-- Status:      In progress
===============================================================================
'''
from skimage.metrics import structural_similarity as ssim
from skimage import feature
import cv2
import os
import sys
import imutils
import pickle
import numpy as np
from imutils import paths
from PIL import Image
import imagehash
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from skimage.feature import greycomatrix, greycoprops

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

from matplotlib import pyplot


def c10_GAUSSIAN(X):

    model = GaussianMixture(n_components=10)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c9_SPECTRAL(X):

    model = SpectralClustering(n_clusters=10)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c8_OPTICS(X):
    model = OPTICS(eps=0.8, min_samples=10)
    # fit model and predict clusters
    yhat = model.fit_predict(X)

    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c7_MEANSHIFT(X):
    model = MeanShift()
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c6_MININBATCH(X):
    model = MiniBatchKMeans(n_clusters=10)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c5_KMEANS(X):
    model = KMeans(n_clusters=10)
# fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c4_BRICH(X):
    model = Birch(threshold=0.01, n_clusters=10)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()


def c3_AGGLOMERE(X):

    model = AgglomerativeClustering(n_clusters=10)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show


def c2_AFFINITY(X):
    model = AffinityPropagation(damping=0.9)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    pyplot.show()
# dbscan clusterx


def c1_DBSCAN(X):

    # define the model
    model = DBSCAN(eps=0.30, min_samples=9)
    # fit model and predict clusters
    yhat = model.fit_predict(X)
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])

    # show the plot
    pyplot.show()


cutoff = 5


def GCLM(imageB):
    imageB = cv2.resize(imageB, (200, 200))
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    gcm = greycomatrix(grayB, [1], [0], 256, symmetric=False, normed=True)
    corrolation = np.float(greycoprops(gcm, 'correlation').flatten())*1000
    homogeneity = np.float(greycoprops(gcm, 'homogeneity').flatten())*10000
    contrast = np.float(greycoprops(gcm, 'contrast').flatten())*10
    energy = np.float(greycoprops(gcm, 'energy').flatten())*10000
    data = [contrast, energy, homogeneity, corrolation]
    data = np.array(data)
    data = data/1000

    return data


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):

        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist


class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):

        hist = cv2.calcHist([image], [0, 1, 2],
                            None, self.bins, [0, 256, 0, 256, 0, 256])

        if imutils.is_cv2():
            hist = cv2.normalize(hist)

        else:
            hist = cv2.normalize(hist, hist)

        return hist.flatten()


def histogram(image):
    hists = []

    for chan in cv2.split(image):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        #hist = hist.reshape(1, -1)
        hist = hist.flatten()
  #hist = hist.reshape(1, -1)
        hists.append(hist)

    hists = np.array(hists)

    hists = hists.flatten()
    return np.array(hists)


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    return d


def HOG(imageB):
    imageB = cv2.resize(imageB, (200, 200)).astype("float32")
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    (HB, hogImage) = feature.hog(grayB, orientations=9, pixels_per_cell=(10, 10),
                                 cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", visualize=True)
    return(HB)


def LBP(imageB):
    imageB = cv2.resize(imageB, (200, 200)).astype("float32")
    desc = LocalBinaryPatterns(24, 8)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    histB = desc.describe(grayB)
    return (histB)


def HASH(nameimageB):
    imageB = cv2.imread(nameimageB)
    hashB = imagehash.average_hash(Image.open(nameimageB))
    return(hashB)


def RGB(imageB):
    desc = RGBHistogram([8, 8, 8])
    featuresB = desc.describe(imageB)
    return (featuresB)


def allCLUSTER(X):
    c1_DBSCAN(X)
    c2_AFFINITY(X)
    c3_AGGLOMERE(X)
    c4_BRICH(X)
    c5_KMEANS(X)
    c6_MININBATCH(X)
    c7_MEANSHIFT(X)
    c8_OPTICS(X)
    c9_SPECTRAL(X)
    c10_GAUSSIAN(X)


listed = list(paths.list_images(sys.argv[1]))
imagePaths = sorted(listed, key=lambda e: e)
# imageA = np.zeros((100, 485), dtype='uint8')
# cv2.imwrite('imageA.png', imageA)

# nameimageA = 'imageA.png'
# H, W = imageA.shape[:2]
# print(nameimageA)

# imagePaths[0]
# filenameA=nameimageA.split(os.path.sep)[-1]
# imageA = cv2.imread(nameimageA)
XX = []
YY = []
ZZ = []
GG = []
for imagePath in imagePaths:
    filenameB = imagePath.split(os.path.sep)[-1]
    nameimageB = imagePath
    imageB = cv2.imread(nameimageB)
#    imageB = cv2.resize(imageB, (W, H))
    GM = GCLM(imageB)
    #HB = HOG(imageB)
    #LBPB = LBP(imageB)
  #  featuresB = RGB(imageB)
#    hashB = HASH(nameimageB)
# ##  hashA = HASH(nameimageA)
# ##   featuresA = RGB(imageA)
# ##   LBPA = LBP(imageA)
# ##   HA = HOG(imageA)

#     D1 = chi2_distance(histogram(imageA), histogram(imageB))
#     D2 = chi2_distance(featuresA, featuresB)
#     D3 = chi2_distance(HA, HB)
#     D4 = np.abs(hashA-hashB)

  #  XX.append(HB)
 #   YY.append(LBPB)
    GG.append(GM)
#    ZZ.append(histogram(imageB))
    # cv2.  imwrite("Distance\\"+str(int(D*10))+"_"+filenameB, imageB)
    # err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    # err /= float(imageA.shape[0] * imageA.shape[1])
    # similarity = ssim(imageA, imageB, multichannel=True)
    # cv2.imwrite("Distance\\"+str(int(D*10000))+"_"+filenameB, imageB)
    print(imagePath)
#XX = np.array(XX)
#YY = np.array(YY)
#ZZ = np.array(ZZ)
GG = np.array(GG)
# allCLUSTER(XX)
# allCLUSTER(YY)
# allCLUSTER(ZZ)
allCLUSTER(GG)
