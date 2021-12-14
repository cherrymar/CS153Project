from feature_detection import getData
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np
import csv

fileout = "data2.csv"

'''
input: filepath to folder of images, total number of articles, number of topics (preset to 6)
output: produces a csv file with the probability of each image being in each topic
'''
def do_lda(imgs, num_articles, num_topics=6, annotations=None):
    imgs1, imgs2, imgs3, imgs4, imgs5, imgs6= imgs
    X1, y = getData(imgs1, 20, 40, annotations) # should be 240 maybe at some point i dont know i give up
    X2, y = getData(imgs2, 20, 90, annotations)
    X3, y = getData(imgs3, 20, 6, annotations)
    X4, y = getData(imgs4, 20, 17, annotations)
    X5, y = getData(imgs5, 20, 40, annotations)
    X6, y = getData(imgs6, 20, 129, annotations)

    X = np.concatenate((X1, X2, X3, X4, X5, X6))
    X_list = [X1, X2, X3, X4, X5, X6]

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)

    lda.fit(X)

    with open(fileout, 'w', newline='') as outf:
        fieldnames = ['Class', '1', '2', '3', '4', '5', '6']
        csvwriter = csv.DictWriter(outf, fieldnames)
        csvwriter.writeheader()
    
    for index, im in enumerate(imgs):
        predict = lda.transform(X_list[index])
        with open(fileout, 'a', newline='') as outf:
            fieldnames = ['Class', '1', '2', '3', '4', '5', '6']
            csvwriter = csv.DictWriter(outf, fieldnames)
            for p in predict:
                new_row = {}
                new_row['Class'] = im
                new_row['1'] = p[0]
                new_row['2'] = p[1]
                new_row['3'] = p[2]
                new_row['4'] = p[3]
                new_row['5'] = p[4]
                new_row['6'] = p[5]
                csvwriter.writerow(new_row)

if __name__ == '__main__':

    do_lda(["bottle", "dog", "faces", "freiberg", "motorbike", "office"], 34)