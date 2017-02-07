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

def gettopklabel4mp(scores,k):
    scores = scores - np.min(scores);
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]
    ss = sortedscores[:20]
    ss = ss/np.sum(ss)
    ss = ss[:5]
    ss = ss/np.sum(ss)
    return sortedlabel[:k],ss[:k]

def sumfuse(mbh,ims,k):
    mbh = mbh - np.min(mbh)+1.0;
    ims = ims - np.min(ims)+1.0;
    # mbh = mbh/np.sum(mbh)
    # ims = ims/np.sum(ims)
    scores = mbh*ims;
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]

    ss = sortedscores[:5]
    ss = ss/np.sum(ss)
    return sortedlabel[:k],ss[:k]

def wAPfuse(mbh,ims,wmbh,wims,k):
    
    for i in range(200):
         mbh[i] = (1+wmbh[i])*mbh[i]
         ims[i] = (1+wims[i])*ims[i]
    
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    # mbh = mbh/np.sum(mbh)
    # ims = ims/np.sum(ims)
    scores = mbh + ims;
    
    # scores = np.mean(wmbh)*mbh+np.mean(wims)*ims;
    # scores = np.zeros(200)
    # for i in range(200):
    #      scores[i] = (mbh[i]*wmbh[i]+wims[i]*ims[i])/(wmbh[i]+wims[i]+1);
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]

    ss = sortedscores[:5]
    ss = ss/np.sum(ss)
    return sortedlabel[:k],ss[:k]

def fuseThree(mbh,ims,c3d,k):
    
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    #c3d = c3d - np.min(c3d)+1;
    # mbh = mbh/np.sum(mbh)
    # ims = ims/np.sum(ims)
    # print 'we are here in fuse three'
    scores = np.zeros_like(mbh);#*ims*c3d; 
    for i in range(200):
        scores[i] = (mbh[i]*c3d[i]*ims[i])*(mbh[i]+ims[i]+c3d[i])
    
    # scores = mbh*ims;
    # scores = np.mean(wmbh)*mbh+np.mean(wims)*ims;
    # scores = np.zeros(200)
    # for i in range(200):
    #      scores[i] = (mbh[i]*wmbh[i]+wims[i]*ims[i])/(wmbh[i]+wims[i]+1);
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]

    ss = sortedscores[:k]
    ss = ss/np.sum(ss[:5])
    return sortedlabel[:k],ss[:k]

def fuseCLF(clf,mbh,ims,c3d,k):
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    scores1 = mbh*ims*c3d
    scores2 = mbh+ims+c3d
    mbh = mbh/np.mean(mbh)
    ims = ims/np.mean(ims)
    c3d = c3d/np.mean(c3d)
    scores = scores1/np.mean(scores1)
    X = np.zeros((1,800))
    count = 0;
    X[count,:200] =c3d;
    X[count,200:400] = mbh;
    X[count,400:600] = ims;
    X[count,600:] = scores;
    # print np.shape(X)
    clfScore = clf.decision_function(X);
    clfScore = clfScore - np.min(clfScore) +1;
    # print np.shape(clfScore)
    clfScore = scores2*scores1*clfScore[0]
    scores = clfScore/np.sum(clfScore);
    
    sortedlabel = np.argsort(scores)[::-1]
    # print scores
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]
    ss = sortedscores/np.sum(sortedscores[:3])
    
    return sortedlabel[:k],ss[:k]

def fuseCLFnEXT(clf,mbh,ims,c3d,ext,k):
    mbh = mbh - np.min(mbh)+0.9;
    ims = ims - np.min(ims)+1.4;
    scores1 = mbh*ims*c3d
    scores2 = mbh+ims+c3d+ext+1
    mbh = mbh/np.mean(mbh)
    ims = ims/np.mean(ims)
    c3d = c3d/np.mean(c3d)
    scores = scores1/np.mean(scores1)
    X = np.zeros((1,800))
    count = 0;
    X[count,:200] =c3d;
    X[count,200:400] = mbh;
    X[count,400:600] = ims;
    X[count,600:] = scores;
    # print np.shape(X)
    clfScore = clf.decision_function(X);
    clfScore = clfScore - np.min(clfScore) +1;
    # print np.shape(clfScore)
    clfScore = scores2*scores1*clfScore[0]
    scores = clfScore/np.sum(clfScore);
    
    sortedlabel = np.argsort(scores)[::-1]
    # print scores
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]
    ss = sortedscores/np.sum(sortedscores[:3])
    
    return sortedlabel[:k],ss[:k]

def fuseFour(mbh,ims,c3d,ext,k):
    
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    #c3d = c3d - np.min(c3d)+1;
    # mbh = mbh/np.sum(mbh)
    # ims = ims/np.sum(ims)
    scores = mbh*ims*c3d; 
    scores = scores-min(scores);
    
    # scores = np.mean(wmbh)*mbh+np.mean(wims)*ims;
    # scores = np.zeros(200)
    # for i in range(200):
    #      scores[i] = (mbh[i]*wmbh[i]+wims[i]*ims[i])/(wmbh[i]+wims[i]+1);
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    # print sortedlabel
    sortedscores = scores[sortedlabel]
    # print sortedlabel[:k],sortedscores[:k]

    ss = sortedscores
    ss = ss/np.sum(ss[:5])
    return sortedlabel[:k],ss[:k]
      
def getC3dMeanPreds(preds,classtopk=80):
    preds = preds - np.min(preds) + 0.9;
    scores = np.zeros(200)
    topk = min(classtopk,np.shape(preds)[0]);
    
    # for i in range(np.shape(preds)[0]):
        # preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        # preds[i,:] = preds[i,:]/np.sum(preds[i,:]) ;
    
    for i in range(200):
        values = preds[:,i];
        values = np.sort(values);
        values = values[::-1]
        scores[i] = np.mean(values[:topk])
    
    return scores

def getEXTMeanPreds(preds,classtopk=250):
    # preds = preds - np.min(preds) + 1;
    scores = np.zeros(200)
    topk = min(classtopk,np.shape(preds)[0]);
    
    # for i in range(np.shape(preds)[0]):
        # preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        # preds[i,:] = preds[i,:]/np.sum(preds[i,:]) ;
    
    for i in range(200):
        values = preds[:,i];
        values = np.sort(values);
        values = values[::-1]
        scores[i] = np.mean(values[:topk])
    
    return scores

def readpkl(filename):
    with open(filename) as f:
        data = pickle.load(f)
    return data
    
def processOnePredictions():
    
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    
    K = 5;
    subset = 'validation';#,'testing']:
    
    featType = 'MBH'
    savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    with open(savename,'r') as f:
        data = pickle.load(f)
    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3))    
    if True: #not os.path.isfile(outfilename):
        vcount = 0;
        vdata = {};
        vdata['external_data'] = {'used':True, 'details':"We use extraction Net model with its weights pretrained on imageNet dataset and fine tuned on Activitty Net. Plus ImagenetShuffle, MBH features, C3D features privide on challenge page"}
        vdata['version'] = "VERSION 1.3"
        results = {}
        
        for videoId in database.keys():
            videoInfo = database[videoId]
            if videoInfo['subset'] == subset:
                if vcount >-1:
                    vidresults = []
                    vcount+=1
                    vidname = 'v_'+videoId
                    print 'processing ', vidname, ' vcount ',vcount 
                    ind = data['vIndexs'][videoId]
                    preds = data['scores'][ind,:]
                    print 'shape of preds',np.shape(preds)
                    labels,scores = gettopklabel4mp(preds,K)
                    print labels
                    print scores
                    for idx in range(K):
                            score = scores[idx]
                            # if score>0.05:
                            label = labels[idx]
                            name = names[label]
                            tempdict = {'label':name,'score':score}
                            vidresults.append(tempdict)
                    results[videoId] = vidresults
        vdata['results'] = results
        # print vdata
        print 'results saved in ', outfilename
        with open(outfilename,'wb') as f:
            json.dump(vdata,f)

def getDATA(gtlabels,dataIMS,dataMBH,infileC3D,database,subset):
    X = np.zeros((11000,800))
    Y = np.zeros(11000)
    count = 0;
    for videoId in database.keys():
            videoInfo = database[videoId]
            if videoInfo['subset'] == subset:
                # if vcount >-1:
                vidresults = []
                # vcount+=1
                vidname = 'v_'+videoId
                # print 'processing ', vidname, ' vcount ',vcount 
                ind = dataMBH['vIndexs'][videoId]
                predsMBH = dataMBH['scores'][ind,:]
                
                ind = dataIMS['vIndexs'][videoId]
                predsIMS = dataIMS['scores'][ind,:]
                
                preds = infileC3D[videoId]['scores']
                predS3D = getC3dMeanPreds(preds)
                
                predsMBH = predsMBH - np.min(predsMBH)+1;
                predsIMS = predsIMS - np.min(predsIMS)+1;
                scores = predS3D*predsMBH*predsIMS
                predS3D = predS3D/np.mean(predS3D)
                predsMBH = predsMBH/np.mean(predsMBH)
                predsIMS = predsIMS/np.mean(predsIMS)
                scores = scores/np.mean(scores)
                
                X[count,:200] =predS3D;
                X[count,200:400] = predsMBH;
                X[count,400:600] = predsIMS;
                X[count,600:] = scores;
                Y[count] = gtlabels[videoId];
                count+=1
                #labels,scores = fuseThree(predsMBH,predsIMS,predS3D,K)
    return X[:count],Y[:count]
                
def trainPreds():
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    
    K = 5;
    subset = 'validation';#,'testing']:
    
    featType = 'IMS'
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        dataIMS = pickle.load(f)
    
    featType = 'MBH'
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        dataMBH = pickle.load(f)
        
    featType = 'C3D'
    savename = '{}data/ALLpredictions-SVM-{}.hdf5'.format(baseDir,featType)
    infileC3D = h5py.File(savename,'r');
    
    xtrain,ytrain = getDATA(gtlabels,dataIMS,dataMBH,infileC3D,database,'training')
    print 'got training and shape is ',np.shape(xtrain)
    xval,yval = getDATA(gtlabels,dataIMS,dataMBH,infileC3D,database,'validation')
    print 'got validation and shape is ',np.shape(xval)
    
    numSamples = np.shape(xval)[0]
    bestclf = {};
    bestscore = 0;
    Cs = [0.001,0.01,0.1,1,10,100];    
    for cc in Cs:
        clf = LinearSVC(C = cc)#,probability=True)
        clf = clf.fit(xtrain, ytrain)
        preds = clf.predict(xval)
        correctPreds = preds == yval;
        score = 100*float(np.sum(correctPreds))/numSamples
        print 'Overall Accuracy is ',score, '% ', ' C = ',str(cc),' features = ',featType
        if score>bestscore:
            bestclf = clf
            bestscore = score
            
    saveName = '{}data/LinearfusiontrainingSVM-{}.pkl'.format(baseDir,featType)
    with open(saveName,'w') as f:
            pickle.dump(bestclf,f)
            
def processThreePredictions():
    
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    
    K = 196;
    subset = 'testing' #'validation';#,'testing']:
    
    featType = 'IMS'
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        dataIMS = pickle.load(f)
    
    featType = 'MBH'
    savename = '{}data/ALLpredictions-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        dataMBH = pickle.load(f)
        
    featType = 'C3D'
    savename = '{}data/ALLpredictions-SVM-{}.hdf5'.format(baseDir,featType)
    infileC3D = h5py.File(savename,'r');
    
    featType = 'EXT'
    savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
    infileEXT = h5py.File(savename,'r');
    
    featType = 'C3D'
    savename = '{}data/LinearfusiontrainingSVM-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        clf = pickle.load(f)
        
    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,'IMS-MBH-C3D-SUBMIT-OLD',str(K).zfill(3))    
    if True: #not os.path.isfile(outfilename):
        vcount = 0;
        vdata = {};
        vdata['external_data'] = {'used':True, 'details':"We use ImagenetShuffle  features, MBH features  and C3D features provided on challenge page."}
        vdata['version'] = "VERSION 1.3"
        results = {}
        for videoId in database.keys():
            videoInfo = database[videoId]
            if videoInfo['subset'] == subset:
                # if vcount >-1:
                   
                    vidresults = []
                    vcount+=1
                    vidname = 'v_'+videoId
                    print 'processing ', vidname, ' vcount ',vcount 
                    ind = dataMBH['vIndexs'][videoId]
                    predsMBH = dataMBH['scores'][ind,:]
                    
                    ind = dataIMS['vIndexs'][videoId]
                    predsIMS = dataIMS['scores'][ind,:]
                    
                    preds = infileC3D[videoId]['scores']
                    predS3D = getC3dMeanPreds(preds,10)
                    
                    preds = np.transpose(infileEXT[videoId]['scores'])
                    predEXT = getEXTMeanPreds(preds,20)
                    
                    #print 'shape of preds',np.shape(preds)
                    # labels,scores = fuseThree(predsMBH,predsIMS,predS3D,K)
                    # labels,scores = fuseFour(predsMBH,predsIMS,predS3D,predEXT,K)
                    # labels,scores = fuseCLF(clf,predsMBH,predsIMS,predS3D,K)
                    labels,scores = fuseCLFnEXT(clf,predsMBH,predsIMS,predS3D,predEXT,K)
                    print labels,scores
                    
                    for idx in range(K):
                            score = scores[idx]
                            # if score>0.05:
                            label = labels[idx]
                            name = names[label]
                            tempdict = {'label':name,'score':score}
                            vidresults.append(tempdict)
                    results[videoId] = vidresults
        vdata['results'] = results
        # print vdata
        print 'result saved in ',outfilename
        print 'process three result saved in ',outfilename
        with open(outfilename,'wb') as f:
            json.dump(vdata,f)
            
def fuse2withAP():
    
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    
    K = 5;
    subset = 'validation';#,'testing']:
    
    featType = 'IMS'
    savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    with open(savename,'r') as f:
        dataIMS = pickle.load(f)
    
    savename = '{}data/weightAP-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        wIMS = pickle.load(f)
        
    featType = 'MBH'
    savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    with open(savename,'r') as f:
        dataMBH = pickle.load(f)
    
    savename = '{}data/weightAP-{}.pkl'.format(baseDir,featType)
    with open(savename,'r') as f:
        wMBH = pickle.load(f)
    
    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,'ap-fused-IMS-MBH',str(K).zfill(3))
    
    if True: #not os.path.isfile(outfilename):
        vcount = 0;
        vdata = {};
        vdata['external_data'] = {'used':True, 'details':"We use extraction Net model with its weights pretrained on imageNet dataset and fine tuned on Activitty Net. Plus ImagenetShuffle, MBH features, C3D features privide on challenge page"}
        vdata['version'] = "VERSION 1.3"
        results = {}
        
        for videoId in database.keys():
            videoInfo = database[videoId]
            if videoInfo['subset'] == subset:
                if vcount >-1:
                    vidresults = []
                    vcount+=1
                    vidname = 'v_'+videoId
                    print 'processing ', vidname, ' vcount ',vcount 
                    ind = dataMBH['vIndexs'][videoId]
                    predsMBH = dataMBH['scores'][ind,:]
                    
                    ind = dataIMS['vIndexs'][videoId]
                    predsIMS = dataIMS['scores'][ind,:]
                    #print 'shape of preds',np.shape(preds)
                    # labels,scores = sumfuse(predsMBH[:201],predsIMS[:201],K)
                    labels,scores = wAPfuse(predsMBH,predsIMS,wMBH,wIMS,K)
                    print labels
                    print scores
                    for idx in range(K):
                            score = scores[idx]
                            # if score>0.05:
                            label = labels[idx]
                            name = names[label]
                            tempdict = {'label':name,'score':score}
                            vidresults.append(tempdict)
                    results[videoId] = vidresults
        vdata['results'] = results
        # print vdata
        print 'Result saved in ',outfilename
        with open(outfilename,'wb') as f:
            json.dump(vdata,f)

if __name__=="__main__":
    # processOnePredictions()
    # processTwoPredictions()
    # fuse2withAP()
    processThreePredictions()
    # trainPreds()
    
