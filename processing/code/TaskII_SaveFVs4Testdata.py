import os.path
from Mytools import GestureSample
import Mytools as mytools
import scipy.io as sio
import numpy as np
import pickle as pickle
        
def saveFVs(QuadDir,NumberofFrame,gmm,nbc,halfWindows):
    savefilename = "{}FVS/FVsnbc_{}_halfwin_{}.mat".format(QuadDir,str(nbc),str(halfWindows[-1])) 
#    if not os.path.isfile(savefilename):
    XX = []
    print "Saving Framwise FVS for ", QuadDir;
    for numFrame in range(1,NumberofFrame+1):
        filename = '{}desc{}.mat'.format(QuadDir,str(numFrame).zfill(5))
        Quads  = sio.loadmat(filename)['QuadDescriptors']
        XX.append(Quads)
    num = np.shape(XX)[0]
#    fvs = np.zeros((NumberofFrame,nbc*13))
    Allfvs = [np.zeros((NumberofFrame,nbc*13)) for k in range(len(halfWindows)) ]
#    del fvs
    print np.shape(Allfvs),' one ',np.shape(Allfvs[0])
    for numFrame in xrange(1,NumberofFrame+1):
        wincount = -1
        for halfwin in halfWindows:
            wincount+=1
            XXtemp = []
            for fnum in np.arange(max(0,numFrame-halfwin-1),min(numFrame+halfwin,NumberofFrame),1):
                Quads = XX[fnum]
                if np.shape(Quads)[0]>1:
                    XXtemp.extend(Quads)
            num = np.shape(XXtemp)[0]
            if num>0:
                Allfvs[wincount][numFrame-1,:] = mytools.fisher_vector(XXtemp, gmm)
    
    wincount = -1    
    for halfwin in halfWindows:
        wincount+=1
        savefilename = "{}FVS/FVsnbc_{}_halfwin_{}.mat".format(QuadDir,str(nbc),str(halfwin))
        fvs = Allfvs[wincount]
        sio.savemat(savefilename,mdict = {'fvs':fvs})
        
if __name__=='__main__':
    datapath = '../TestData/'
    isAll = False 
    Single =True;
    nbc = 128
    halfWindows = [1,2,3,4,8,10,12,15,17,18,19,22,25,30]
    
    vocabName = '../Models/Vocab_nbc_{}.pkl'.format(str(nbc))
    with open(vocabName,'r')  as f:
        gmm = pickle.load(f)
    fileList = os.listdir(datapath)
        
    samplelist=[files for files in fileList if files.startswith("Sample")]
    
    for sample in samplelist:
        if Single:
            QuadDir = '{}{}/SpecificQuadDescriptors/'.format(datapath,sample)
        else:
            QuadDir = '{}{}/AllQuadDescriptors/'.format(datapath,sample)
        print("Computing fisher vectors for  " + sample);
        if not os.path.isdir(QuadDir+'FVS/'):
            os.mkdir(QuadDir+'FVS/')
        smp=GestureSample(datapath,sample,training = False);
        Numberofframe = smp.data['numFrames'];
        saveFVs(QuadDir,Numberofframe,gmm,nbc,halfWindows)

