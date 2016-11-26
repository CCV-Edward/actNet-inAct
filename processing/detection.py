'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''

import numpy as np
import pickle
import os
import time,json
#import pylab as plt

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
    ## Read frame level predications
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
        for word in splitedline[1:-1]:
            # print word,
            preds[wcount,lcount] = float(word)
            wcount+=1
        lcount +=1
    return labels,preds

def getscores(D,starts,ends,labels):
    scores = np.zeros_like(labels,dtype = 'float32')
    # print 'shape of scores ',np.shape(scores)
    idx = 0;
    for label in labels:
        scores[idx] = (D[label,ends[idx]] - D[label,starts[idx]])/(ends[idx]-starts[idx])
        idx+=1
    return scores

def removeBackground(labels,scores,starts,ends):
    newlabels = np.zeros_like(labels)
    newscores = np.zeros_like(scores)
    newstarts = np.zeros_like(starts)
    newends= np.zeros_like(ends)
    count = 0;idx = 0
    for label in labels:
        if label<200:
            newscores[count] = scores[idx]
            newlabels[count] = labels[idx]
            newstarts[count] = starts[idx]
            newends[count] = ends[idx]
            count+=1
        idx+=1
    return newlabels[:count],newscores[:count],newstarts[:count],newends[:count]

def getsegments(preds,alpha=5):
    (p,D) = dpEM(preds,alpha)
    labels,starts,ends = getLabels(p)
    # print 'Number of segments generated are ',np.shape(labels)
    scores = getscores(D,starts,ends,labels)
    labels,scores,starts,ends = removeBackground(labels,scores,starts,ends)
    return labels,scores,starts,ends

def refineCalssification(labels,scores):
    newLabels = [labels[0]]
    newScores = [scores[0]]
    for ind in [1,2,3,4]:
        if scores[ind]>0.05:
            newLabels.append(labels[ind])
            newScores.append(scores[ind])
    return newLabels,newScores
def getfullLength(labels,length):
    starts=[];ends=[]
    for i in range(len(labels)):
        starts.append(0);
        ends.append(length)
    return np.asarray(starts),np.asarray(ends)

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
    

def dpEMmax(M,alpha=3):
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
            values= D[:, j-1] - v1
            tb = np.argmax(values)
            dmax = max(values)
            D[i,j] = D[i,j]+dmax;
            phi[i,j] = tb;
#    pdb.set_trace()
    q = c-1;
    values= D[:, c-1]
    p = np.argmax(values)
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

def processPredictions():
    weight = 35000;
    subset = 'testing'
    imgtype = 'rgb'
    for alpha in [100]:
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
                fps = videoInfo['fps'];
                numf = videoInfo['numf'];
                if videoInfo['subset'] == subset:
                    if vcount >-1:
                        vidresults = []
                        vcount+=1
                        vidname = 'v_'+videoId
                        print 'processing ', vidname, ' vcount ',vcount,' fps ',fps, ' numf ',numf,'alpha',alpha,
                        gtlabels,preds = getpredications(subset,imgtype,weight,vidname)
                        labels,scores,starts,ends = getsegmentswithcls(preds,alpha)
                        print ' Number of detection are ',len(labels)
                        print labels,scores
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
                                # tempdict = json.JSONEncoder.default(tempdict)
                                vidresults.append(tempdict)
                        results[videoId] = vidresults
        vdata['results'] = results
        # print vdata
        outfilename = '{}results/Detection/{}-{}-{}-alpha{}.json'.format(baseDir,subset,imgtype,
                                                                  str(weight).zfill(5),str(alpha).zfill(3))
        print 'result are saved in ',outfilename
        # vdata = json.JSONEncoder.default(vdata)
        with open(outfilename,'wb') as f:
            json.dump(vdata,f)
if __name__=="__main__":
    processPredictions()
