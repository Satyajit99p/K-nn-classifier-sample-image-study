import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import argparse


class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        self.width=width
        self.height=height
        self.inter=inter

#to process all the images into a uniform dimension

    def PreProcess(self,image):
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)

class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        self.preprocessors=preprocessors
        if self.preprocessors == None:
            self.preprocessors=[]

    def load(self,imagePaths,verbose=-1):
        data=[]
        labels=[]
        
#reads each image path and appends it into the data list

        for (i,imagePath) in enumerate(imagePaths):
            image=cv2.imread(imagePath)
            label=imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image=p.PreProcess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 & i > 0 & (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))

            return (np.array(data),np.array(labels))

#Command line interface

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-k","--neighbours",type=int,default=1,help="no.of nearest neighbours for classification")
ap.add_argument("-j","--jobs",type=int,default=-1,help="no of jobs for Knn distance")
args=vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths=list(paths.list_images(args["dataset"]))
sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.reshape((data.shape[0],3072))

print("[INFO] features matrix: {:.1f}MB".format( data.nbytes / (1024 * 1000.0)))

le=LabelEncoder()
labels=le.fit_transform(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.75,random_state=42)

print("[INFO] evaluating k-NN classifier..")
model=KNeighborsClassifier(n_neighbors=args["neighbours"],n_jobs=args["jobs"])
model.fit(trainX,trainY)

print(classification_report(testY,model.predict(testX),target_names=le.classes_))








