import os
from Mytools import GestureSample
import Mytools as mytools
import scipy.io as sio
import numpy as np
import pickle as pickle
        
def saveClfOut(sample,QuadDir,OutDir,halfwin,Numberofframe,svmclf,nbc):

        outFile = '{}binary_halfwin_{}.mat'.format(OutDir,str(halfwin))
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
        vecAllfvs = mytools.power_normalize(vecAllfvs,0.4)

        frame_label = svmclf.predict(vecAllfvs)
        frame_labelstemp  = np.zeros((Numberofframe))
        frame_labelstemp[isFrame_labeled>0]=frame_label
        print 'saving to ' , outFile
        sio.savemat(outFile,mdict={'frame_label':frame_labelstemp,'isFrame_labeled':isFrame_labeled})

if __name__=='__main__':

    datapath = '../TestData/'

    Single =True;
    nbc =128

    models = [{},{},{}];m=0
    halfwins = [2,3,4]
    for halfwin in halfwins:
        clfsavename = '../Models/Binary_{}.pkl'.format(str(halfwin))
        print 'loading Model '
        with open(clfsavename,'r')  as f:
            binarclf = pickle.load(f)
        print 'Done Loading Model'

        fileList = os.listdir(datapath)

        samplelist=[files for files in fileList if files.startswith("Sample")]  
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
            saveClfOut(sample,QuadDir,OutDir,halfwin,Numberofframe,binarclf,nbc)
