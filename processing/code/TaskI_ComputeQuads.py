import os.path
from Mytools import GestureSample
import Mytools as mytools
import scipy.io as sio
import numpy as np

def storeQuads4Sample((datapath,saveDir,isAll,sample,Single)):   
    if Single:
        combs = sio.loadmat('../Models/lesscombs.mat')['combs'];
    else:
        combs = sio.loadmat('../Models/combnk.mat')['combs'];
    JointsToUse = ['HipCenter','Spine','ShoulderCenter','Head','ShoulderLeft', \
                    'ShoulderLeft','ElbowLeft','WristLeft','HandLeft',\
                    'ShoulderRight','ElbowRight','WristRight','HandRight']
    #print 'shape of comb is', np.shape(combs)
    smp=GestureSample(datapath,sample,skel=True);

    print 'Computing quads for ', sample 
    Numberofframe = smp.data['numFrames'];
    
    for numFrame in range(1,Numberofframe+1):
        skel=smp.getSkeleton(numFrame);
        JointsData = skel.getWorldCoordinates();
        Joints3D = np.zeros((12,3))
        i=0;
        for joint in JointsData:
            if joint in JointsToUse:
                Joints3D[i,:] = JointsData[joint];
                i = i+1
        saveQuads(saveDir,isAll,numFrame,Joints3D,Single,combs)
                              
def saveQuads(saveDir,isAll,numFrame,Joints3D,Single,combs):
    savefilename = '{}desc{}.mat'.format(saveDir,str(numFrame).zfill(5))  
    QuadDescriptors = []
#    AllQuadDescriptors = []
    if np.sum(Joints3D[0,:])>0.05:     
        for combination in combs:
            quadrupleJoints = Joints3D[combination-1]
            QuadDescriptor = mytools.ComputeQuadDescriptor(quadrupleJoints,Single,isAll)
            if isAll:
                QuadDescriptors.extend(QuadDescriptor)    
            else:
                QuadDescriptors.append(QuadDescriptor)
    QuadDescriptors = checkDescs4NAN(QuadDescriptors)
    sio.savemat(savefilename,mdict={'QuadDescriptors':QuadDescriptors})
                
def checkDescs4NAN(des):
    NANs = np.isnan(des)
    newdes = des;
    if NANs.any()>0:
        print 'There is NAN case', np.shape(des)
        newdes = des[not NANs];
        print 'sahpe after', np.shape(newdes)
    return newdes
if __name__ == '__main__':
#    main()
    # Path which contains sample files in .zip format
    
    datapaths = ['../TestData/']
    # Keep an copy the the sample files We delete what is not required
    # _depth.mp4 _video.mp4 _user.mp4  and sampleXXXXX.zip files will be deteted
    mytools.UnzipAllfiles(datapaths[0])
    kk = 0;
    isAll = False 
    Single =True;
    for datapath in datapaths:
        # Get the list of training samples
        fileList = os.listdir(datapath)
        print datapath
        #     Filter input files (only ZIP files)

        samplelist=[files for files in fileList if files.startswith("Sample")]

        for sample in samplelist:
            #print("\t Processing file " + sample)
            if Single:
                QuadDir = '{}{}/SpecificQuadDescriptors/'.format(datapath,sample)
            else:
                QuadDir = '{}{}/AllQuadDescriptors/'.format(datapath,sample)
            if not os.path.isdir(QuadDir):
                os.mkdir(QuadDir)
            storeQuads4Sample((datapath,QuadDir,isAll,sample,Single))


