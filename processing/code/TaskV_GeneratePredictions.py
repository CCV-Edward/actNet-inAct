import os,pdb
import csv,re,math,shutil
import numpy as np
import scipy.io as sio
from Mytools import GestureSample
import Mytools as mytools

def main():
        
    datapath = '../TestData/'
    predictionDir = '../Predictions/'
    if not os.path.isdir(predictionDir):
        os.mkdir(predictionDir)
    Single =True;
    nbc =128
    binhalfwins = [2,3,4];
    nbc =128
    lenthresh = 10
    halfwins = [8,17,25];
    alpha = 3
    shiftstart = -4
    shiftend = -1
    fileList = os.listdir(datapath)
    siz = 1;thresh = 0;
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

        allBinaryLabels = np.zeros((1,Numberofframe))
        ss1=0
        for binhalfwin in binhalfwins:
            BinClfFile = BinClfFile = '{}binary_halfwin_{}.mat'.format(OutDir,str(binhalfwin))
            BinaryLabels = sio.loadmat(BinClfFile)['frame_label']                        
            allBinaryLabels += BinaryLabels
            ss1+=1
        allprobs = np.zeros((20,Numberofframe))
        ss =0 
        for halfwin in halfwins: 
            ss+=1    
            inFile = '{}Classfication_nbc_{}_halfwin{}.mat'.format(OutDir,str(nbc),str(halfwin))
            probabilties = 1-np.transpose(sio.loadmat(inFile)['frame_probs'])
            allprobs += probabilties
        (r,c) = np.shape(probabilties);
        M = np.zeros((r+1,c));
        M[:20,:] = allprobs/ss
#                        M[20,:] = BinaryLabels
        BinaryLabels = mytools.myfilter(allBinaryLabels/ss1,siz,thresh)
#                        M[20,:] = binprob
        M[20,:] = BinaryLabels;
        (p,D) = dpEM(M,alpha)
        (labels,starts,ends) = getLabels(p)
        FinalLabels = refineLabes(labels,starts,ends,M,lenthresh,Numberofframe,shiftstart,shiftend)

        tempfilename  = '../Predictions/{}_prediction.csv'.format(sample)
        mytools.exportPredictions(FinalLabels,tempfilename)
        #GTcsvFile = '../../validationLabels/{}_labels.csv'.format(sample)
        #(overlap,tp,fp,intr,uni,preds,ggt) = mytools.gesture_overlap_csv(GTcsvFile,tempfilename,Numberofframe)
        #print sample,overlap,tp,fp,intr,uni,preds,ggt

def refineLabes(labels,starts,ends,D,lenthresh,Numberofframe,shiftstart,shiftend):
    length = len(labels)
    score = np.zeros(length)
    newLabels = np.zeros(length);
    newStarts = np.zeros(length);
    newEnds = np.zeros(length);
    occurances = np.zeros(21);
    k =int(0);
    for ii in range(length):
        if (ends[ii]-starts[ii])>lenthresh:
            mid = int(starts[ii]+(ends[ii]-starts[ii])/2)
            score[k] = D[int(labels[ii]),mid];
            newLabels[k] = int(labels[ii]);
            occurances[int(labels[ii])]+=1
            newStarts[k] = int(starts[ii]);
            newEnds[k] = int(ends[ii]);
            k+=1
    score = score[:k];
    newLabels = newLabels[:k];
    newStarts = newStarts[:k];
    newEnds = newEnds[:k];
    FinalLabels = np.zeros((k,3))
    fk=0
#    print np.shape(occurances)
    for ii in xrange(k):
#        print newLabels[ii]
        if occurances[int(newLabels[ii])]==1 and newLabels[ii]<20:      
#        if newLabels[ii]<20:  
                FinalLabels[fk,:] = [newLabels[ii],max(newStarts[ii]+shiftstart,0),min(newEnds[ii]+shiftend,Numberofframe)];
                fk+=1
    for ii in xrange(20):
        if occurances[ii]>1:        
            repeatedgesture = ii;
            mylabel = np.zeros((int(occurances[ii]),3))
            m=0;
            for kk in xrange(k):
                if newLabels[kk] ==  repeatedgesture:
#                    print repeatedgesture
                    mylabel[m,:] = [score[kk],newStarts[kk],newEnds[kk]];
                    m+=1
            idx = np.argmin(mylabel[:,0])
            FinalLabels[fk,:] = [repeatedgesture,max(mylabel[idx,1]+shiftstart,0),min(mylabel[idx,2]+shiftend,Numberofframe)];
            fk+=1
    return FinalLabels[:fk,:]+1    
        
def getLabels(p):
    starts = np.zeros(300);
    ends = np.zeros(300);
    labels = np.zeros(300);
    fl = 0
    i=0
    starts[i]=0
    fl = p[0]
    labels[i] =  p[0]
#    print p[0]
#    pdb.set_trace()
    for ii in range(len(p)):
        if abs(p[ii] -fl)>0:
            ends[i]=ii
            i+=1
            fl = p[ii]
            starts[i]=ii+1
            labels[i] = fl
    ends[i] = len(p)
#    print i, starts[:i+1],ends[:i+1],labels[:i+1]
    return (labels,starts,ends)
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
    
if __name__=='__main__':
    main()
