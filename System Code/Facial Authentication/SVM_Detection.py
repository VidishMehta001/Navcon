# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:16:41 2020

@author: vidis
"""

from django.http.response import StreamingHttpResponse
from datetime import datetime, timedelta
import os

from django.shortcuts import render, redirect
from cv2 import cv2
import numpy as np
#import logging
from sklearn.model_selection import train_test_split
from PIL import Image

from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import sklearn.metrics
from sklearn.metrics import roc_curve, auc

import joblib
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

#from Navcon.settings import BASE_DIR


def plot_gallery(images, titles, h, w, n_row=1, n_col=1):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap= plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        

def getImagesWithID(path):
    #counterTest = 0
    h = 150
    w = 150
    # create a list for the path for all the images that is available in the folder
    # from the path(dataset folder) this is listing all the directories and it is fetching the directories from each and every pictures
    # And putting them in 'f' and join method is appending the f(file name) to the path with the '/'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] #concatinate the path with the image name
    #print imagePaths

    # Now, we loop all the images and store that userid and the face with different image list
    faces = []
    Ids = []
    for imagePath in imagePaths:
        #counterTest = counterTest + 1
        # First we have to open the image then we have to convert it into numpy array
        faceToResize = Image.open(imagePath).convert('L') #convert it to grayscale
        faceImg = faceToResize.resize((h,w), Image.ANTIALIAS)
        #faceImg.save('test/'+str(counterTest)+'.jpg')

        # converting the PIL image to numpy array
        # @params takes image and convertion format
        faceNp = np.array(faceImg, 'uint8')
        #Converting 2D array into 1D
        faceNp = faceNp.flatten()

        # Now we need to get the user id, which we can get from the name of the picture
        # for this we have to slit the path() i.e dataset/user.1.7.jpg with path splitter and then get the second part only i.e. user.1.7.jpg
        # Then we split the second part with . splitter
        # Initially in string format so hance have to convert into int format
        ID = int(os.path.split(imagePath)[-1].split('.')[1]) # -1 so that it will count from backwards and slipt the second index of the '.' Hence id
        # Images
        faces.append(faceNp)
        # Label
        Ids.append(ID)
        #print ID
        # cv2.imshow("training", faceNp)
        # cv2.waitKey(10)
    return np.array(Ids), np.array(faces), h, w


def eigenTrain():
    path = 'C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Navcon/svm/dataset'
    print(path)

    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= getImagesWithID(path)
    print('features'+str(faces.shape[1]))
    
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    
    n_classes = y_test.size
    print(y_test)    
    
    target_names = ['1', '2', '3']
    
    n_components = 1
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    # #############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    y_score = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
    


    # #############################################################################
    # Qualitative evaluation of the predictions using matplotlib


    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    # plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    svm_pkl_filename = 'C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Navcon/svm/serializer/svm_classifier.pkl'
    # Open the file to save as pkl file
    svm_model_pkl = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl)
    # Close the pickle instances
    svm_model_pkl.close()

    pca_pkl_filename = 'C:/Users/vidis/OneDrive/Desktop/Navcon/Navcon/Navcon/svm/serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    plt.show()

eigenTrain()