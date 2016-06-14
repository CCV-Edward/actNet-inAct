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
from eval_detection import ANETdetection
import scipy.io as sio

#######baseDir = "/mnt/sun-alpha/actnet/";
baseDir = "/data/shared/solar-machines/actnet/";
#baseDir = "/mnt/solar-machines/actnet/";
########imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
######## imgDir = "/mnt/DATADISK2/ss-workspace/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"

def getscore(ground_truth_filename, prediction_filename,
          tiou_thr=0.5,subset='validation', verbose=True, check_status=True):
    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thr=tiou_thr,
                                   verbose=verbose, check_status=True)
    ap = anet_detection.evaluate()
    return ap
    
def saveAPs():
    K = 5;
    subset = 'validation';#,'testing']:
    featType = 'IMS-MBH'
    # savename = '{}data/predictions-{}-{}.pkl'.format(baseDir,subset,featType)
    # with open(savename,'r') as f:
    #     data = pickle.load(f)
    outfilename = '{}results/classification/{}-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3))
    gtfiile = 'data/activity_net.v1-3.min.json'
    ap = getscore(gtfiile,outfilename,top_k=1)
    print ap
    print np.mean(ap)
    savename = '{}data/weightAP-{}.pkl'.format(baseDir,featType)
    print 'Results saved in ',savename
    with open(savename,'w') as f:
        pickle.dump(ap,f)
        
def plotAPs():
    K = 1;
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
    
def evalAll():
    K = 10;
    subset = 'validation';#,'testing']:
    gtfiile = 'data/activity_net.v1-3.min.json'
    
    result = []; count = 0;
    featType = 'C3D-BIN-BOOST-LONG'
    # outfilename = '{}results/detection/{}-{}-K-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3),'alpha-001')
    for alpha in [1,3,5,]:
        outfilename = '{}results/detection/{}-{}-K-{}-{}.json'.format(baseDir,subset,featType,str(K).zfill(3),'alpha-{}'.format(str(int(alpha*10)).zfill(3)))
        print 'Evaluating results from ',outfilename
        for tioth in [0.5,0.4,0.3,0.2,0.1]:
            ap = getscore(gtfiile,outfilename,tiou_thr=tioth)
            result.append([alpha,tioth,np.mean(ap)])
            
    result = np.aaarray(result)
    sio.savemat('result-{}.mat'.format(featType),mdict={'ap':ap})
        
if __name__=="__main__":
    #processOnePredictions()
    # saveAps()
    # plotmAPs()
    evalALL()
