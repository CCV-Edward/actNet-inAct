'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''

import numpy as np
import pickle
import os
import time,json
import pylab as plt

baseDir = "/mnt/sun-alpha/actnet/";
imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
# imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
    
def readannos():
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    return actionIDs,taxonomy,database
def getnames():
    fname = baseDir+'lists/gtnames.list'
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
    
def processPredictions():
    weight = 30000;
    subset = 'validation'
    imgtype = 'rgb'
    K = 3;
    for classtopk in [50,100,200,500]:
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
                    if vcount <2:
                        vidresults = []
                        vcount+=1
                        vidname = 'v_'+videoId
                        print 'processing ', vidname, ' vcount ',vcount
                        gtlabels,preds = getpredications(subset,imgtype,weight,vidname)
                        labels,scores = gettopklabel(preds,K)
                        print labels
                        print scores
                        for idx in range(K):
                            score = scores[idx]
                            if score>0.015:
                                label = labels[idx]
                                name = names[label]
                                tempdict = {'label':name,'score':score}
                                vidresults.append(tempdict)
                        results[videoId] = vidresults
        vdata['results'] = results
        # print vdata
        outfilename = '{}results/{}-{}-{}-K{}-clsk{}.json'.format(baseDir,subset,imgtype,
                            str(weight).zfill(5),str(K).zfill(3),str(classtopk).zfill(4))
        with open(outfilename,'wb') as f:
            json.dump(vdata,f)
if __name__=="__main__":
    processPredictions()
