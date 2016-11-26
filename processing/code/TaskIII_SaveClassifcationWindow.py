import os.path
from Mytools import GestureSample
import Mytools as mytools
import scipy.io as sio
import numpy as np
import pickle as pickle
        
def saveClfOut(QuadDir,OutDir,halfwins,Numberofframe,svmclf,nbc):
              
    for halfwin in halfwins:
            outFile = '{}Classfication_nbc_{}_halfwin{}.mat'.format(OutDir,str(nbc),str(halfwin))
#        if not os.path.isfile(outFile):
            FVsFile = "{}FVS/FVsnbc_{}_halfwin_{}.mat".format(QuadDir,str(nbc),str(halfwin))
            fvs = sio.loadmat(FVsFile)['fvs']
            vecAllfvs = np.zeros((Numberofframe,nbc*13))
            isFrame_labeled = np.zeros(Numberofframe)
            i = 0;
            for fnum in xrange(Numberofframe):
                fvsum = np.sum(fvs[fnum])
                if abs(fvsum)>0:
                    vecAllfvs[i,:] = fvs[i,:]
                    isFrame_labeled[fnum] = 1
                    i+=1          

            vecAllfvs = vecAllfvs[:i,:]
            vecAllfvs = mytools.power_normalize(vecAllfvs,0.2)
            frame_probs = svmclf.predict_proba(vecAllfvs)
            frame_label = svmclf.predict(vecAllfvs)
            frame_probstemp  = np.zeros((Numberofframe,20))
            frame_probstemp[isFrame_labeled>0,:] = frame_probs
            frame_labelstemp  = np.zeros(Numberofframe)
            frame_labelstemp[isFrame_labeled>0]=frame_label
            print 'saving to ' , outFile
            sio.savemat(outFile,mdict={'frame_probs':frame_probstemp, 'frame_label':frame_labelstemp, 
            'isFrame_labeled':isFrame_labeled})

if __name__=='__main__':
    """ Main script. Perform only the traing part, including feature computation steps """

    datapath = '../TestData/' 
    Single =True;
    modelname = '../Models/model.pkl'
    with open(modelname,'r')  as f:
        svmclf = pickle.load(f)
    print 'Done Loading Model'

    fileList = os.listdir(datapath)
    nbc =128
    samplelist=[files for files in fileList if files.startswith("Sample")]

    halfwins = [12,15,17,18,19,22,25]
    halfwins = [8,10,30]
    for sample in samplelist:
        if Single:
            QuadDir = '{}{}/SpecificQuadDescriptors/'.format(datapath,sample)
        else:
            QuadDir = '{}{}/AllQuadDescriptors/'.format(datapath,sample)
        print 'saving classifction number for',sample
        smp=GestureSample(datapath,sample,training = False);
        Numberofframe = smp.data['numFrames'];
        OutDir = '{}ClassifierOutputs/'.format(QuadDir)
        if not os.path.isdir(OutDir):
            os.mkdir(OutDir)
        saveClfOut(QuadDir,OutDir,halfwins,Numberofframe,svmclf,nbc)
