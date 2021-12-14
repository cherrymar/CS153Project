import cv2
import math as m
import numpy as np
from scipy import sparse 
from glob import glob
from sklearn.cluster import KMeans
from xml_parser import getAnnotations
from sklearn.decomposition import LatentDirichletAllocation

'''
input: filepath to image
output: SIFT keypoints and features
'''
def getKpsandFeats(image):
    img = cv2.imread(image)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kps, feats = sift.detectAndCompute(gray,None)

    return kps, feats

'''
input: filepath to images, number of pixels?(Features), num_articles
output: a lil matrix of features for all images
(code from past NLP homework)
'''
def processImages(imgs, num_features, num_articles):
    X = sparse.lil_matrix((num_articles, num_features), dtype='uint8')
    index = 0
    for filename in glob(imgs + '/*.png'):
        kps, feats = getKpsandFeats(filename)
        for j, value in enumerate(feats):
            X[index,j] = value
        index += 1
    return X

'''
input: SIFT features
output: the codewords
'''
def getCodeWords(feats):
    kmeans = KMeans(n_clusters=20, random_state=0).fit(feats) # should be 240 clusters at the end
    codewords = kmeans.cluster_centers_
    return codewords

'''
input: array of numbers
output: square root of the sum of squares of the numbers in the array
'''
def sumOfSquares(arr):
    sum = 0
    answer = []
    for a in arr:
        for i in a:
            sum += i**2
        answer.append(m.sqrt(sum))
    return answer

'''
input: the filepath to folder of images, number of features in each image, number of images
output: (2D array of images and the features corresponding to each image, y=[]) 
we return the empty y list because in some classifiers they would like to have labels, but we didn't need it for this project
'''
def getData(imgs, num_features, num_articles, annotations=None):
    # X is an np array with the codewords for each image
    # y is the class of each image
    X = np.zeros((num_articles, num_features), dtype='O')
    y = []
    index = 0
    for filename in glob(imgs + '/*.jpg'):
        xml_filename = 'annotations\\' + filename.split(".")[0] + '.xml'
        ann = getAnnotations(xml_filename)
        kps, feats = getKpsandFeats(filename)
        codewords = getCodeWords(feats)
        if annotations:
            codewords += annotations ## somehow add the annotations 
        new_array = sumOfSquares(codewords) ## somehow add the annotations 
        while len(new_array) < num_features:
            new_array += [None]
        X[index] = new_array
        index += 1
    return (X, y)


if __name__ == '__main__':
    kps, feats = getKpsandFeats('coins.jpg')
    print(feats.shape)
    x = getCodeWords(feats)
    # print(x)
    print(x.shape)
    # sizeOfImage = 
    # bigfeats = np.zeros((3909, 128))
    # print(bigfeats.shape)
    # bigfeats = getKpsandFeats('coins.jpg')[1]
    # print(bigfeats.shape)
    # bigfeats = getKpsandFeats('penguin.jpg')[1]
    # print(bigfeats.shape)
    # bigfeats = getKpsandFeats('propeller.jpg')[1]
    # print(bigfeats.shape)
    # # print(feats)
    # print(getCodeWords(bigfeats))
    X, y = getData("web_static_street_april-images", 70, 34)

    lda = LatentDirichletAllocation(n_components=6, random_state=0)
    # print(X_train)
    lda.fit(X)

    predict = lda.transform(X)

    print(predict)
