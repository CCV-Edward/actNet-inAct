'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os,h5py
import time,json
#import pylab as plt

#######baseDir = "/mnt/sun-alpha/actnet/";
baseDir = "/data/shared/solar-machines/actnet/";
########imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
######## imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
    
def power_normalize(xx, alpha=0.5):

    """Computes a alpha-power normalization for the matrix xx."""

    return np.sign(xx) * np.abs(xx) ** alpha

def readannos():
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    return actionIDs,taxonomy,database

def getnames():
    fname = baseDir+'data/lists/gtnames.list'
    with open(fname,'rb') as f:
        lines = f.readlines()
    names = []
    for name in lines:
        name = name.rstrip('\n')
        names.append(name)
    # print names
    return names

def getpredications(subset,imgtype,weight,vidname):
    predictionfile = '{}predictions/{}-{}-{}/{}.list'.format(baseDir,subset,imgtype,str(weight).zfill(5),vidname)
    with open(predictionfile) as f:
        lines = f.readlines()
    preds = np.zeros((201,len(lines)),dtype = 'float32')
    labels = np.zeros(len(lines))
    lcount = 0;
    for line in lines:
        splitedline = line.split(' ');
        labels[lcount] = int(splitedline[0])
        wcount = 0;
        # print 'line length ', len(splitedline)
        # print splitedline
        for word in splitedline[1:-1]:
            # print word,
            preds[wcount,lcount] = float(word)
            wcount+=1
        lcount +=1
    return labels,preds

def gettopklabel(preds,k,classtopk):
    scores = np.zeros(200)
    topk = min(classtopk,np.shape(preds)[1]);
    for i in range(200):
        values = preds[i,:];
        values = np.sort(values);
        values = values[::-1]
        scores[i] = np.mean(values[:topk])
    # print scores
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]
    return sortedlabel[:k],sortedscores[:k]

def readpkl(filename):
    with open(filename) as f:
        data = pickle.load(f)
    return data

def getdataVal(database,indexs,gtlabels,subset,featType):
    
    if featType == 'MBH':
        filename = baseDir+'data/MBH_Videos_features.hdf5';
        x = np.zeros((18000,65536))
    else:
        filename = baseDir+'data/ImageNetShuffle2016_features.hdf5';
        x = np.zeros((18000,1024))
        
    file = h5py.File(filename,'r')
    features = file['features']
    #print np.shape(features)
    
    count = 0;
    y = np.zeros(18000)
    #features = power_normalize(features)
    for videoId in database.keys():
        videoInfo = database[videoId]
        if not videoInfo['subset'] == 'testing':

            vkey = 'v_'+videoId;
            ind = indexs[vkey]
            label = gtlabels[videoId]
            #feat = features[ind,:]
            x[count,:] = features[ind,:];
            y[count] = label
            count+=1
    file.close()
    return x[:count],y[:count]

def processMBHval():
    for featType in ['MBH']:
        names = getnames()
        gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
        indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
        actionIDs,taxonomy,database = readannos()
        print 'getting training data.... ',
        xtrain,ytrain = getdataVal(database,indexs,gtlabels,'training',featType)
        print 'got it!! and shape is ',np.shape(xtrain)
        #print 'getting validation data.... ',
        #xval,yval = getdata(database,indexs,gtlabels,'validation',featType)
        #print 'got it!! and shape is ',np.shape(xval)
        
    
        if featType == 'IMS':
            jobs = 16
            c = 0.01;
        else:
            jobs = 16
            c = 10;
    
        clf = LinearSVC(C = c)
        clf = clf.fit(xtrain, ytrain)
        
        saveName = '{}data/train-valSVM-{}.pkl'.format(baseDir,featType)
        with open(saveName,'w') as f:
            pickle.dump(clf,f)
            
if __name__=="__main__":
    #processPredictions()
    processMBHval()
    
