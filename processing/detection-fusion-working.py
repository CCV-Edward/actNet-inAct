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
import scipy.io as sio
import copy
#import pylab as plt

#######baseDir = "/mnt/sun-alpha/actnet/";
baseDir = "/data/shared/solar-machines/actnet/";
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

def getsegmentswithcls(preds,alpha=5):
    labels,scores = gettopklabel(preds,10,200)
    labels,scores = refineCalssification(labels,scores)
    #(p,D) = dpEM(preds,alpha)
    #labels,starts,ends = getLabels(p)
    starts,ends = getfullLength(labels,np.shape(preds)[1])
    # print 'Number of segments generated are ',np.shape(labels)
    #scores = getscores(D,starts,ends,labels)
    #labels,scores,starts,ends = removeBackground(labels,scores,starts,ends)
    return labels,scores,starts,ends

def getfullLength(labels,length):
    starts=[];ends=[]
    offset = int(length*0.15)
    for i in range(len(labels)):
        starts.append(offset);
        ends.append(length-offset)
    return np.asarray(starts),np.asarray(ends)

def fuseThree(mbh,ims,c3d,numf,k):    
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    scores = mbh*ims*c3d; 
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    sortedscores = scores[sortedlabel]
    ss = sortedscores
    ss = ss/np.sum(ss[:5])
    labels,scores = sortedlabel[:k],ss[:k]
    starts,ends = getfullLength(labels,numf)
    return labels,scores,starts,ends


def dpEM(M,alpha):
    (r,c) = np.shape(M);
    D = np.zeros((r, c+1)) # add an extra column
    D[:,0] = 1# % put the maximum cost
    D[:, 1:(c+1)] = M;
    
#    v = np.ones(r)*alpha;
    phi = np.zeros((r,c))
#    pdb.set_trace()
    for j in xrange(1,c):
        for i in xrange(r):
            
#            values.index(min(values))
            v1 = np.ones(r)*alpha
            v1[i] = 0;
            values= D[:, j-1]+v1
            tb = np.argmin(values)
            dmax = min(values)
            D[i,j] = D[i,j]+dmax;
            phi[i,j] = tb;
#    pdb.set_trace()
    q = c-1;
    values= D[:, c-1]
    p = np.argmin(values)
    i = p

    j = q 
    ps = np.zeros(c)
    ps[q] = p
    while j>0:
        tb = phi[i,j];
        j = int(j-1);
        q = j;
        ps[q] = tb;
        i = int(tb);
    
    D = D[:,1:];
    return (ps,D)

def getLabels(p):
    starts = np.zeros(500);
    ends = np.zeros(500);
    labels = np.zeros(500,dtype='int32');
    fl = 0
    i=0
    starts[i]=0
    fl = p[0]
    labels[i] =  p[0]
#    print p[0]
#    pdb.set_trace()
    for ii in range(len(p)):
        if abs(p[ii] -fl)>0:
            ends[i]=ii-1
            fl = p[ii]
            i+=1
            starts[i]=ii
            labels[i] = fl
    ends[i] = len(p)-1
#    print i, starts[:i+1],ends[:i+1],labels[:i+1]
    return labels[:i+1],starts[:i+1],ends[:i+1]

def getSegment4mAlphaEXT(topklabels,topscors,clscores,predEXT,C3Dfeat,fps,numf,alpha):  
    
    labels = [];    scores = [];    starts = [];     ends = [];
    clscores = clscores/np.sum(clscores);
    norms = np.sum(topscors[:5])
    for label in topklabels[:1]:
        clScore = clscores[label]/norms;
        colScore = predEXT[:,label];
        colScore = colScore/np.max(colScore)
        M = np.zeros((2,len(colScore)))
        M[1,:] = 1-colScore
        M[0,:] = colScore
        # print M
        
        scs = [.5,.6,.7,.8,.9,.1]
        offsetA = 0;
        while len(scs)>2:
            (p,D) = dpEM(M,alpha+offsetA)
            # print p
            ls,ss,eds = getLabels(p)
            scs,ss,eds = refinelabels(ls,ss,eds,colScore)
            offsetA+=1
            
        if len(scs)>0:
            for ind in range(len(scs)):
                labels.append(label)
                scores.append(scs[ind])
                starts.append(ss[ind])
                ends.append(eds[ind])
        else:
            labels.append(label)
            # scores.append(clScore)
            scols = sorted(colScore)
            scols = scols[::-1]
            scores.append(np.mean(scols[:min(len(colScore),200)]))
            starts.append(int(len(colScore)*0.10))
            ends.append(int(len(colScore) - len(colScore)*0.10))

        
    return labels,scores,starts,ends

def getSegmentBinaryC3D(topklabels,topscors,clscores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha):  
    # topklabels,topscors,scores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha
    indexs = np.asarray(C3Dfeat['indexs']);
    frameLabels = np.asarray(C3Dfeat['labels']);
    preds = np.asarray(C3Dfeat['scores']);
    c3numf  = np.shape(preds)[0];
    preds = preds - np.min(preds);
    
    # predsEXT = predEXT['scores']
    # sio.savemat('data.mat',mdict= {'indexs':indexs,'topklabels':topklabels,'topscors':topscors,'clscores':clscores,'preds':preds,'numf':numf,'fps':fps,'frameLabels':frameLabels,'predEXT':predEXT})
    
    for i in range(c3numf):
        preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    # preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    t2f = (c3numf*fps)/numf;
    labels = [];    scores = [];    starts = [];     ends = [];
    clscores = clscores/np.sum(clscores);
    norms = np.sum(topscors[:2])
    topscors = topscors/norms;
    lcount = 0;
    binSVM = smoothit(np.asarray(C3DfeatbinSVM['scores']))
    binRFcopy = np.asarray(C3DfeatbinRF['scores'])
    
    for label in topklabels[:15]:
        binRF  = copy.deepcopy(binRFcopy[:,1])
        clScore = topscors[lcount];
        colScore = preds[:,label]
        lcount +=1
        
        # colScoreSmoothed = smoothColScores(colScore,10)
        # binRF =   colScoreSmoothed;
        # binRF = smoothit(binRF);
        #binRF = binRF-np.mean(binRF);
        # sortedScores = sorted(binRF)
        # offset = int(c3numf*0.06);
        # minS = np.mean(sortedScores[:offset])
        # sortedScores = sortedScores[:-1];
        # print sortedScores
        # maxS  = np.mean(sortedScores[c3numf-int(1.5*offset):])
        # print minS,maxS
        # binRF = (binRF-minS)/(maxS-minS)
        # extColScore = predEXT[indexs,label]
        # binRF[extColScore>0.8] = 0.95;
        # binRF[binSVM>0.65] = 0.85;
        # binRF[binSVM>0.8] = 0.99;
        # binRF[binSVM<-0.4] = 0.1;
        # binRF[binSVM<-0.85] = 0.00;
        # binRF[colScoreSmoothed>0.6] = 0.9;
        # binRF[colScoreSmoothed<0.1] = 0.0;
        # else:
        #     binRF = binRF-minS
        # binRF =   newScores;
        # print 'saving it'
        # sio.savemat('colScoreSmoothed.mat',mdict = {'binSVM':binSVM,'binRF':binRF,'frameLabels':frameLabels});
        # sio.savemat('colScoreSmoothed.mat',mdict = {'binRF':binRF,'frameLabels':frameLabels});
        # colScoreSmoothed = binRF[:,1]
        # M = np.transpose(binRF);
        # print M
        
        M = np.zeros((2,c3numf))
        M[0,:] = 1-binRF
        M[1,:] = binRF
        # # print M
        
        ls = [1,2,3,4,5,6,7,8,9,10,11]
        # talpha = alpha;
        # while len(ls)>7:
        (p,D) = dpEM(M,alpha)
        ls,ss,eds = getLabels(p)
        # talpha += 0.2
             
        scs,ss,eds = refinelabels(ls,ss,eds,binRF)
        
        # print scs,ss,eds
         
        if len(scs)>0:
             for ind in range(len(scs)):
                 labels.append(label)
                 scores.append(clScore)
                 starts.append(ss[ind])
                 ends.append(eds[ind])
        else:
            # error('we have problem')
        # else:
            labels.append(label)
            # scols = sorted(binRF)
            # scols = scols[::-1]
            # seglen = min(int(len(scols)*0.6),30)
            # scores.append(np.mean(scols[:seglen])*clScore)
            scores.append(clScore)
            # scores.append(clScore)
            starts.append(int(c3numf*0.12))
            ends.append(int(c3numf - c3numf*0.12))

        # st = int(segInit[sInd]*t2f)
        # et = int(segEnd[sInd]*t2f)
        # bscore = clScore*np.mean(colScore[st:et])*pscores[sInd]
        # # print st,et,bscore,np.mean(colScore[st:et]),pscores[sInd],clScore
        # 
        # if bscore>0.01:
        #     labels.append(label); scores.append(bscore);
        #     starts.append(segInit[sInd]*fps);
        #     ends.append(segEnd[sInd]*fps);
        
    return labels,scores,starts,ends,c3numf

def getSegment4mAlphaC3D(topklabels,topscors,clscores,predEXT,C3Dfeat,fps,numf,alpha):  
    
    indexs = np.asarray(C3Dfeat['indexs']);
    frameLabels = np.asarray(C3Dfeat['labels']);
    preds = np.asarray(C3Dfeat['scores']);
    c3numf  = np.shape(preds)[0];
    preds = preds - np.min(preds);
    sio.savemat('data.mat',mdict= {'indexs':indexs,'topklabels':topklabels,'topscors':topscors,'clscores':clscores,'preds':preds,'numf':numf,'fps':fps,'frameLabels':frameLabels,'predEXT':predEXT})
    
    for i in range(c3numf):
        preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
        # preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    t2f = (c3numf*fps)/numf;
    labels = [];    scores = [];    starts = [];     ends = [];
    clscores = clscores/np.sum(clscores);
    norms = np.sum(topscors[:5])
    topscors = topscors/norms;
    lcount = 0;
    for label in topklabels[:1]:
        clScore = topscors[lcount];
        lcount +=1
        colScore = preds[:,label]/norms;
        colScoreSmoothed = smoothColScores(colScore,10)
        sio.savemat('colScoreSmoothed.mat',mdict = {'colScoreSmoothed':colScoreSmoothed,'colScore':colScore});
        
        M = np.zeros((2,len(colScoreSmoothed)))
        M[1,:] = 1-colScoreSmoothed
        M[0,:] = colScoreSmoothed
        # print M
        (p,D) = dpEM(M,alpha)
        # print p
        ls,ss,eds = getLabels(p)
        # print p
        # print ls,ss,eds
        scs,ss,eds = refinelabels(ls,ss,eds,colScoreSmoothed)
        # print scs,ss,eds
        if len(scs)>0:
            for ind in range(len(scs)):
                labels.append(label)
                scores.append(scs[ind]*clScore)
                starts.append(ss[ind])
                ends.append(eds[ind])
        # else:
        labels.append(label)
        scols = sorted(colScoreSmoothed)
        scols = scols[::-1]
        seglen = min(int(len(scols)*0.5),30)
        scores.append(np.mean(scols[:seglen])*clScore)
        # scores.append(clScore)
        starts.append(int(c3numf*0.10))
        ends.append(int(c3numf - c3numf*0.10))
        
        
        # st = int(segInit[sInd]*t2f)
        # et = int(segEnd[sInd]*t2f)
        # bscore = clScore*np.mean(colScore[st:et])*pscores[sInd]
        # # print st,et,bscore,np.mean(colScore[st:et]),pscores[sInd],clScore
        # 
        # if bscore>0.01:
        #     labels.append(label); scores.append(bscore);
        #     starts.append(segInit[sInd]*fps);
        #     ends.append(segEnd[sInd]*fps);
        
    return labels,scores,starts,ends,c3numf


def getBinaryAccuracy(topklabels,topscors,clscores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha):  
    # topklabels,topscors,scores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha
    indexs = np.asarray(C3Dfeat['indexs']);
    frameLabels = np.asarray(C3DfeatbinRF['labels']);
    preds = np.asarray(C3Dfeat['scores']);
    c3numf  = np.shape(preds)[0];
    preds = preds - np.min(preds);
    
    # predsEXT = predEXT['scores']
    # sio.savemat('data.mat',mdict= {'indexs':indexs,'topklabels':topklabels,'topscors':topscors,'clscores':clscores,'preds':preds,'numf':numf,'fps':fps,'frameLabels':frameLabels,'predEXT':predEXT})
    
    for i in range(c3numf):
        preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    # preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    t2f = (c3numf*fps)/numf;
    # labels = [];    scores = [];    starts = [];     ends = [];
    clscores = clscores/np.sum(clscores);
    norms = np.sum(topscors[:2])
    topscors = topscors/norms;
    lcount = 0;
    binSVM = smoothit(np.asarray(C3DfeatbinSVM['scores']),5)
    binRF = np.asarray(C3DfeatbinRF['scores'])
    # print ' shapes ', np.shape(binRF),np.shape(binSVM)
    binRF = smoothit(binRF[:,1],5)
    
    # print ' shapes ', np.shape(binRF)
    label =topklabels[0]
    clScore = topscors[lcount];
    colScore = preds[:,label]
    lcount +=1
    colScoreSmoothed = smoothColScores(colScore,6)
    extColScore = predEXT[indexs,label]
    
    accEXT = 1.0; accbinSVM = 1.0; accbinRF = 1.0; accC3D = 1.0;
    extth = 0.8; binSVMth = 0.8; binRFth = 0.8; c3dth = 0.8;
    countrf = 1.0; countext = 1.0;
    countc3 = 1.0; countBsvm = 1.0;
    # binRF[binSVM<-0.1] = 0.05;
    # binRF[extColScore>0.6] = 0.85;
    for i in range(c3numf):
        if extColScore[i]>=extth:
            if frameLabels[i]<200:
                accEXT+=1
            countext = countext+1
        if binRF[i]>=binRFth:
            if frameLabels[i]<200:
                accbinRF+=1
            countrf = countrf+1
        if binSVM[i]>=binSVMth:
            if frameLabels[i]<200:
                accbinSVM+=1
            countBsvm = countBsvm+1
        if colScoreSmoothed[i]>=c3dth:
            if frameLabels[i]<200:
                accC3D+=1
            countc3 = countc3+1
        # if (colScoreSmoothed[i]<c3dth and frameLabels[i]==200):
        #     accC3D+=1
        # if (binRF[i]<binRFth and frameLabels[i]==200):
        #     accbinRF+=1
        # if binSVM[i]<binSVMth and frameLabels[i]==200:
        #     accbinSVM+=1;
        # if frameLabels[i]==200:
        #     count+=1
    
    # for i in range(c3numf):
    #     if (extColScore[i]>=extth and frameLabels[i]<200) or (extColScore[i]<extth and frameLabels[i]==200):
    #         accEXT+=1
    #     if (colScoreSmoothed[i]>=c3dth and frameLabels[i]<200) or (colScoreSmoothed[i]<c3dth and frameLabels[i]==200):
    #         accC3D+=1
    #     if (binRF[i]>=binRFth and frameLabels[i]<200) or (binRF[i]<binRFth and frameLabels[i]==200):
    #         accbinRF+=1
    #     if (binSVM[i]>=binSVMth and frameLabels[i]<200) or (binSVM[i]<binSVMth and frameLabels[i]==200):
    #         accbinSVM+=1;
    #     if frameLabels[i]<201:
    #         count+=1
    print np.asarray([accEXT/countext,accbinRF/countrf,accbinSVM/countBsvm,accC3D/countc3])
    
    return accEXT/countext,accbinRF/countrf,accbinSVM/countBsvm,accC3D/countc3,1

def smoothit(colScore,hws=5):
    if len(colScore)<hws:
        colScore = colScore/np.max(colScore)
        return colScore             #hws = int(len(colScore)/2);
    newScores = np.zeros_like(colScore)
    
    numelm = len(colScore)
    for i in range(numelm):
        ts = 0;count = 0;
        for k in np.arange(max(i-hws,0),min(numelm,i+hws),1):
            count += 1
            ts += colScore[k]
        if count>0:
            newScores[i]  = float(ts)/count
        else:
            newScores[i]  = colScore[i]
    return newScores

def smoothColScores(colScore,hws=5):
    if len(colScore)<hws:
        colScore = colScore/np.max(colScore)
        return colScore             #hws = int(len(colScore)/2);
    newScores = np.zeros_like(colScore)
    
    numelm = len(colScore)
    for i in range(numelm):
        ts = 0;count = 0;
        for k in np.arange(max(i-hws,0),min(numelm,i+hws),1):
            count += 1
            ts += colScore[k]
        if count>0:
            newScores[i]  = float(ts)/count
        else:
            newScores[i]  = colScore[i]
            
    sortedScores = sorted(newScores)[::-1]
    minS = np.mean(sortedScores[-5:-2])
    # sortedScores = sortedScores
    # print sortedScores
    maxS  = np.mean(sortedScores[1:5])
    if maxS>0:
        newScores = (newScores-minS)/(maxS-minS)
    else:
        newScores = newScores-minS
    newScores[newScores<0] = 0
    
    # for i in range(len(newScores)):
    #     if newScores[i]>0.4 and newScores[i]<0.5:
    #         newScores[i]*=1.5
    # newScores[newScores>0.8] = 1.0
    return newScores

    
def refinelabels(inlabels,instarts,inends,colScore):
    scores = [];    starts = [];     ends = [];
    offset = len(colScore)*0.15;
    for ind in range(len(inlabels)):
        segIndexs = np.asarray(np.arange(instarts[ind],inends[ind],dtype=int))
        if inlabels[ind] == 0 and len(segIndexs)>offset:
            starts.append(max(offset,instarts[ind]-5));
            ends.append(min(inends[ind]+5,len(colScore)-offset))
            scols = sorted(colScore[segIndexs])
            seglen = min(int(len(scols)*0.8),190)
            scols = scols[::-1]
            sc = np.mean(scols[:seglen])
            scores.append(len(segIndexs))
    if len(scores)>0:
        ind = np.argmax(scores)
        return [scores[ind]],[starts[ind]],[ends[ind]]
    else:
        return scores,starts,ends
        
    
def getSegment4mProp(topklabels,topscors,clscores,C3Dfeat,props,fps,numf):
    pscores = props['score'];
    segInit = props['segment-init'];
    segEnd = props['segment-end'];
    
    indexs = C3Dfeat['indexs'];
    frameLabels = C3Dfeat['labels'];
    preds = C3Dfeat['scores'];
    preds = preds - np.min(preds) + 1;
    c3numf  = np.shape(preds)[0];
    for i in range(c3numf):
        preds[i,:] = preds[i,:] - np.min(preds[i,:])+1;
        preds[i,:] = preds[i,:]/np.sum(preds[i,:]);
    
    t2f = (c3numf*fps)/numf;
    labels = [];    scores = [];    starts = [];     ends = [];
    clscores = clscores/np.sum(clscores);
    norms = np.sum(topscors)
    for label in topklabels[:1]:
        clScore = clscores[label]/norms;
        colScore = preds[:,label]/norms;
        print 'number of props',len(pscores)
        for sInd in range(min(len(pscores),5)):
            # if pscores[sInd]>0.3:
                st = int(segInit[sInd]*t2f)
                et = int(segEnd[sInd]*t2f)
                bscore = clScore*np.mean(colScore[st:et])*pscores[sInd]
                # print st,et,bscore,np.mean(colScore[st:et]),pscores[sInd],clScore
                
                if bscore>0.01:
                    labels.append(label); scores.append(bscore);
                    starts.append(segInit[sInd]*fps);
                    ends.append(segEnd[sInd]*fps);
                
    return labels,scores,starts,ends

def getTOPclasses(mbh,ims,c3d,k):    
    mbh = mbh - np.min(mbh)+1;
    ims = ims - np.min(ims)+1;
    scores = mbh*ims*c3d; 
    scores = scores/np.sum(scores);
    sortedlabel = np.argsort(scores)[::-1]
    sortedscores = scores[sortedlabel]
    return sortedlabel[:k],sortedscores[:k],scores

def getC3dMeanPreds(preds,classtopk=80):
    preds = preds - np.min(preds) + 1;
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


def processThreePredictions():
    
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    for alpha in [3,]:
        
        K = 15;
        subset = 'testing';#,
        featType = 'IMS'
        savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
        with open(savename,'r') as f:
            dataIMS = pickle.load(f)
        
        featType = 'MBH'
        savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
        with open(savename,'r') as f:
            dataMBH = pickle.load(f)
            
        featType = 'C3D'
        savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3D = h5py.File(savename,'r');
        
        featType = 'EXT'
        savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileEXT = h5py.File(savename,'r');
        
        
        featType = 'C3D'
        savename = '{}data/predictions-BWRF-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3DbinRF = h5py.File(savename,'r');
        
        featType = 'C3D'
        savename = '{}data/predictions-BWSVM-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3DbinSVM = h5py.File(savename,'r');
        
        
        # savename = '{}data/activitynet_v1-3_proposals.hdf5'.format(baseDir)
        # infileProp = h5py.File(savename,'r');
        # 
        outfilename = '{}results/detection/{}-{}-K-{}-alpha-{}.json'.format(baseDir,subset,'C3D-BIN-BOOST-LONG',str(K).zfill(3),str(int(alpha*10)).zfill(3))
        
        if True: #not os.path.isfile(outfilename):
            vcount = 0;
            vdata = {};
            vdata['external_data'] = {'used':True, 'details':"We use ImagenetShuffle  features, MBH features  and C3D features provided on challenge page."}
            vdata['version'] = "VERSION 1.3"
            results = {}
            
            for videoId in database.keys():
                videoInfo = database[videoId]
                
                numf = videoInfo['numf'];
                duration = videoInfo['duration']
                #fps = videoInfo['fps'];
                fps = numf/duration;
                if videoInfo['subset'] == subset:
                    
                    vcount +=1
                    if vcount > -1:
                        vidresults = []
                        # print videoInfo
                        # vcount+=1
                        vidname = 'v_'+videoId
                        # print 'processing ', vidname, ' vcount ',vcount 
                        ind = dataMBH['vIndexs'][videoId]
                        predsMBH = dataMBH['scores'][ind,:]
                        
                        ind = dataIMS['vIndexs'][videoId]
                        predsIMS = dataIMS['scores'][ind,:]
                        
                        C3Dfeat = infileC3D[videoId]
                        C3Dscores = C3Dfeat['scores']
                        predS3D = getC3dMeanPreds(C3Dscores)
                        
                        # props = infileProp[vidname]
                        predEXT = np.transpose(infileEXT[videoId]['scores'])
                        # predEXT = getC3dMeanPreds(preds,220)
                        C3DfeatbinRF = infileC3DbinRF[videoId]
                        C3DfeatbinSVM = infileC3DbinSVM[videoId]
                        #print 'shape of preds',np.shape(preds)
                        
                        print 'processing ', vidname, ' vcount ',vcount,' fps ',fps, ' numf ',numf,' alpha ',alpha,
                        
                        # labels,scores,starts,ends = fuseThree(predsMBH,predsIMS,predS3D,numf,K)
                        topklabels,topscors,scores= getTOPclasses(predsMBH,predsIMS,predS3D,K)
                        # labels,scores,starts,ends = getSegment4mProp(topklabels,topscors,scores,C3Dfeat,props,fps,numf)
                        # labels,scores,starts,ends = getSegment4mAlphaEXT(topklabels,topscors,scores,predEXT,C3Dfeat,fps,numf,alpha)
                        labels,scores,starts,ends,c3numf = getSegmentBinaryC3D(topklabels,topscors,scores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha)
                        print ' Number of detection are ',len(labels)
                        # print labels,scores
                        fps = c3numf/duration;
                        for idx in range(len(labels)):
                                score = scores[idx]
                                label = labels[idx]
                                name = names[label]
                                st = float(starts[idx])/fps
                                et = float(ends[idx])/fps
                                segment = [];
                                segment.append(st);segment.append(et)
                                # print label,score,segment,starts[idx],ends[idx]
                                tempdict = {'label':name,'score':float(score),'segment':segment}
                                vidresults.append(tempdict)
                        results[videoId] = vidresults
            
            
            vdata['results'] = results
            print 'result-saved-in ',outfilename
            with open(outfilename,'wb') as f:
                json.dump(vdata,f)

def getaccuracy():
    
    #########################################
    #########################################
    names = getnames()
    gtlabels = readpkl('{}data/labels.pkl'.format(baseDir))
    indexs = readpkl('{}data/indexs.pkl'.format(baseDir))
    actionIDs,taxonomy,database = readannos()
    ########################################
    ########################################
    for alpha in [0.3,]:
        K = 5;
        subset = 'validation';#,'testing']:
        
        featType = 'IMS'
        savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
        with open(savename,'r') as f:
            dataIMS = pickle.load(f)
        
        featType = 'MBH'
        savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
        with open(savename,'r') as f:
            dataMBH = pickle.load(f)
            
        featType = 'C3D'
        savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3D = h5py.File(savename,'r');
        
        featType = 'EXT'
        savename = '{}data/predictions-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileEXT = h5py.File(savename,'r');
        
        
        featType = 'C3D'
        savename = '{}data/predictions-BWRF-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3DbinRF = h5py.File(savename,'r');
        
        featType = 'C3D'
        savename = '{}data/predictions-BWSVM-{}-{}.hdf5'.format(baseDir,subset,featType)
        infileC3DbinSVM = h5py.File(savename,'r');
        
        
        # savename = '{}data/activitynet_v1-3_proposals.hdf5'.format(baseDir)
        # infileProp = h5py.File(savename,'r');
        # 
        outfilename = '{}results/detection/{}-{}-K-{}-alpha-{}.json'.format(baseDir,subset,'C3D-BIN',str(K).zfill(3),str(int(alpha*10)).zfill(3))
        
        accEXT = 0.0; accbinSVM = 0.0; accbinRF = 0.0; accC3D = 0.0;
        count = 0;
        if True: #not os.path.isfile(outfilename):
            vcount = 0;
            vdata = {};
            vdata['external_data'] = {'used':True, 'details':"We use extraction Net model with its weights pretrained on imageNet dataset and fine tuned on Activitty Net. Plus ImagenetShuffle, MBH features, C3D features privide on challenge page"}
            vdata['version'] = "VERSION 1.3"
            results = {}
            
            for videoId in database.keys():
                videoInfo = database[videoId]
                
                numf = videoInfo['numf'];
                duration = videoInfo['duration']
                #fps = videoInfo['fps'];
                fps = numf/duration;
                if videoInfo['subset'] == subset:
                    
                    vcount +=1
                    if vcount <2000:
                        vidresults = []
                        # print videoInfo
                        # vcount+=1
                        vidname = 'v_'+videoId
                        # print 'processing ', vidname, ' vcount ',vcount 
                        ind = dataMBH['vIndexs'][videoId]
                        predsMBH = dataMBH['scores'][ind,:]
                        
                        ind = dataIMS['vIndexs'][videoId]
                        predsIMS = dataIMS['scores'][ind,:]
                        
                        C3Dfeat = infileC3D[videoId]
                        C3Dscores = C3Dfeat['scores']
                        predS3D = getC3dMeanPreds(C3Dscores)
                        
                        # props = infileProp[vidname]
                        predEXT = np.transpose(infileEXT[videoId]['scores'])
                        # predEXT = getC3dMeanPreds(preds,220)
                        C3DfeatbinRF = infileC3DbinRF[videoId]
                        C3DfeatbinSVM = infileC3DbinSVM[videoId]
                        #print 'shape of preds',np.shape(preds)
                        
                        print 'processing ', vidname, ' vcount ',vcount,' fps ',fps, ' numf ',numf,' alpha ',alpha,
                        
                        # labels,scores,starts,ends = fuseThree(predsMBH,predsIMS,predS3D,numf,K)
                        topklabels,topscors,scores= getTOPclasses(predsMBH,predsIMS,predS3D,K)
                        # labels,scores,starts,ends = getSegment4mProp(topklabels,topscors,scores,C3Dfeat,props,fps,numf)
                        # labels,scores,starts,ends = getSegment4mAlphaEXT(topklabels,topscors,scores,predEXT,C3Dfeat,fps,numf,alpha)
                        aEXT,aSVM,aRF,aC3D,cnf= getBinaryAccuracy(topklabels,topscors,scores,predEXT,C3DfeatbinRF,C3DfeatbinSVM,C3Dfeat,fps,numf,alpha)
                        accEXT += aEXT; accbinSVM += aSVM; accbinRF += aRF; accC3D += aC3D;
                        count +=cnf
            
            print 'Avergae Accuracy is ', np.asarray([accEXT,accbinRF,accbinSVM,accC3D])/count
            

if __name__=="__main__":
    # processOnePredictions()
    # processTwoPredictions()
    # fuse2withAP()
    processThreePredictions()
    # getaccuracy()

    
