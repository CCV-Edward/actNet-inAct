'''
Autor: Gurkirt Singh
Start data: 15th May 2016
purpose: of this file is read frame level predictions and process them to produce a label per video

'''
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import os
import time,json
import pylab as plt
from eval_classification import ANETclassification
import scipy.io as sio

#######baseDir = "/mnt/sun-alpha/actnet/";
baseDir = "/data/shared/solar-machines/actnet/";
#baseDir = "/mnt/solar-machines/actnet/";
########imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
######## imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
  
def getscore(ground_truth_filename, prediction_filename,
         subset='validation', verbose=True, check_status=True,top_k=1):
    anet_classification = ANETclassification(ground_truth_filename,
                                             prediction_filename,
                                             subset=subset, verbose=verbose,
                                             check_status=True,top_k=top_k)
    anet_classification.evaluate()
    return anet_classification.ap,anet_classification.hit_at_k,anet_classification.avg_hit_at_k

   
def processOnePredictions():
        
    K = 10;
    subset = 'validation';#,'testing']:
    gtfiile = 'data/activity_net.v1-3.min.json'
    featType = 'C3D'
    result = []; count = 0;
    cltopks = np.asarray([20,40,60,80,100,120,140,180,240,300,350])
    if featType == 'EXT':
        cltopks = np.asarray([200,130,160,180,200,220,240,280,300,350,400,500,600,800])
    
    for cltopk in reversed(cltopks):
        outfilename = '{}results/classification/{}-{}-{}-clk{}.json'.format(baseDir,subset,featType,str(K).zfill(3),str(cltopk).zfill(4))
        
        ap,hit1,avghit1 = getscore(gtfiile,outfilename,top_k=1)
        ap,hit3,avghit3 = getscore(gtfiile,outfilename,top_k=3)
        result.append([cltopk,np.mean(ap),hit1,hit3])
        print featType ,' cltopk ', cltopk,result[count]
        count+=1
    
    # result = np.aaarray(result)
    sio.savemat('result-{}.mat'.format(featType),mdict={'result':result})
    
def saveAPs():
    K = 5;
    subset = 'validation';#,'testing']:
    featType = 'IMS-MBH'
    # savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    # with open(savename,'r') as f:
    #     data = pickle.load(f)
    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3))
    gtfiile = 'data/activity_net.v1-3.min.json'
    ap,hit1,avghit1 = getscore(gtfiile,outfilename,top_k=1)
    print ap
    print np.mean(ap),hit1
    savename = '{}data/weightAP-{}.pkl'.format(baseDir,featType)
    print 'Results saved in ',savename
    with open(savename,'w') as f:
        pickle.dump(ap,f)
        
def plotmAPs():
    K = 5;
    subset = 'validation';#,'testing']:
    aps = [];
    count = 0;
    colors = ['red','green','blue']
    for featType in ['IMS-MBH','IMS','MBH']:
        savename = '{}data/weightAP-{}.pkl'.format(baseDir,featType)
        print 'Results saved in ',savename
        with open(savename,'r') as f:
            ap = pickle.load(f)
        ind = np.arange(count,600+count,3)
        plt.bar(ind,ap,width=0.5,color=colors[count])
        count += 1
    plt.show()
    
def evalALL():
    K = 5;
    subset = 'validation';#,'testing']:
    gtfiile = 'data/activity_net.v1-3.min.json'
    featType = 'IMS-MBH-C3D-SUBMIT122'
    result = []; 

    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3))
    print 'Evaluating results from ',outfilename
    for tk in [1,3]:
        ap,hit1,avghit1 = getscore(gtfiile,outfilename,top_k=tk)
        ap,hit3,avghit3 = getscore(gtfiile,outfilename,top_k=3)
        result.append([cltopk,np.mean(ap),hit1,hit3])
        
    result = np.aaarray(result)
    sio.savemat('result-{}.mat'.format(featType),mdict={'ap':ap})
        
if __name__=="__main__":
    # processOnePredictions()
    # saveAPs()
    # plotmAPs()
    evalALL()
    
