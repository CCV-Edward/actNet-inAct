'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''
from sklearn.svm import LinearSVC,SVC
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

def getdata(database,indexs,gtlabels,subset,featType):
    
    if featType == 'MBH':
        filename = baseDir+'data/MBH_Videos_features.hdf5';
        x = np.zeros((12000,65536))
    else:
        filename = baseDir+'data/ImageNetShuffle2016_features.hdf5';
        x = np.zeros((12000,1024))
        
    file = h5py.File(filename,'r')
    features = file['features']
    #print np.shape(features)
    
    count = 0;
    y = np.zeros(12000)
    #features = power_normalize(features)
    for videoId in database.keys():
        videoInfo = database[videoId]
        if videoInfo['subset'] == subset:
            vkey = 'v_'+videoId;
            ind = indexs[vkey]
            label = gtlabels[videoId]
            #feat = features[ind,:]
            x[count,:] = features[ind,:];
            y[count] = label
            count+=1
    file.close()
    return x[:count],y[:count]

def processMBH():
    featType = 'IMS'
    # featType = 'MBH'
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    print 'getting training data.... ',
    xtrain,ytrain = getdata(database,indexs,gtlabels,'training',featType)
    print 'got it!! and shape is ',np.shape(xtrain)
    print 'getting validation data.... ',
    xval,yval = getdata(database,indexs,gtlabels,'validation',featType)
    print 'got it!! and shape is ',np.shape(xval)
    
    numSamples = np.shape(xval)[0]
    bestclf = {};
    bestscore = 0;
    # Cs = [0.01,0.1,1,10,100];    
    C_2d_range = [100,1000]
    gamma_2d_range = [0.001,0.01, 1, 10]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            # clf = LinearSVC(C = cc,gamm)
            clf = clf.fit(xtrain, ytrain)
            preds = clf.predict(xval)
            correctPreds = preds == yval;
            score = 100*float(np.sum(correctPreds))/numSamples
            print 'Overall Accuracy is ',score, '% ', ' C = ',str(C),' features = ',featType
            if score>bestscore:
                bestclf = clf
                bestscore = score
            
    saveName = '{}data/RBF-trainingSVM-{}.pkl'.format(baseDir,featType)
    with open(saveName,'w') as f:
            pickle.dump(bestclf,f)
    
    
def processPredictions():
    weight = 30000;
    imgtype = 'rgb';
    for K in [5,10]:
        for subset in ['validation','testing']:
            if subset == 'testing':
                weight = 40000
            else:
                weight = 30000
            for classtopk in [150,250,300]:
                outfilename = '{}results/classification/{}-{}-{}-K{}-clsk{}.json'.format(baseDir,subset,imgtype,
                                    str(weight).zfill(5),str(K).zfill(3),str(classtopk).zfill(4))
                if not os.path.isfile(outfilename):
                    names = getnames()
                    actionIDs,taxonomy,database = readannos()
                    listname = baseDir+'lists/videolist-'+subset+'.list'
                    fid = open(listname,'wb');
                    vcount = 0;
                    vdata = {};
                    vdata['external_data'] = {'used':True, 'details':"We use darknet's (extraction net) imagent pretrained weights"}
                    vdata['version'] = "VERSION 1.3"
                    results = {}
                    for videoId in database.keys():
                            videoInfo = database[videoId]
                            if videoInfo['subset'] == subset:
                                if vcount >-1:
                                    vidresults = []
                                    vcount+=1
                                    vidname = 'v_'+videoId
                                    print 'processing ', vidname, ' vcount ',vcount ,' classtopk ',classtopk
                                    gtlabels,preds = getpredications(subset,imgtype,weight,vidname)
                                    labels,scores = gettopklabel(preds,K,classtopk)
                                    print labels
                                    print scores
                                    for idx in range(K):
                                        score = scores[idx]
                                        if score>0.05:
                                            label = labels[idx]
                                            name = names[label]
                                            tempdict = {'label':name,'score':score}
                                            vidresults.append(tempdict)
                                    results[videoId] = vidresults
                    vdata['results'] = results
                    # print vdata
                    
                    with open(outfilename,'wb') as f:
                        json.dump(vdata,f)
if __name__=="__main__":
    #processPredictions()
    processMBH()
    
