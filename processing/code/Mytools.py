import os, os.path,numpy,zipfile
import csv,re,math,shutil
import numpy as np
import scipy as sp
import scipy
import scipy.io as sio

from PIL import Image, ImageDraw


def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20
#    pdb.set_trace()
    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathgt, 'rb') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            gtGestures.append(int(row[0]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathpred, 'rb') as csvfilepred:
        csvpred = csv.reader(csvfilepred)
        for row in csvpred:
            binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            predGestures.append(int(row[0]))

    # Get the list of gestures without repetitions for ground truth and predicton
    gtGestures = numpy.unique(gtGestures)
    predGestures = numpy.unique(predGestures)

    # Find false positives
    falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    intr = 0;uni = 0;ggt = 0;preds=0;
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)
        ggt+=sum(binvec_gt[idx-1])
        intr+=intersec;
        preds+=sum(binvec_pred[idx-1]);
        uni+= union
    # Use real gestures and false positive gestures to calculate the final score
    return (sum(overlaps)/(len(overlaps)+len(falsePos)),len(overlaps),len(falsePos),intr,uni,preds,ggt)

def evalGesture(prediction_dir,truth_dir):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
    	predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

        # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        score+=gesture_overlap_csv(labelsFile, predFile, numFrames)

    return score/numSamples


def exportPredictions(prediction,output_filename):
    """ Export the given prediction to the correct file in the given predictions path """
    output_file = open(output_filename, 'wb')
    for row in prediction:
        output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
    output_file.close()

def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
 
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
 
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
 
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
 
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]
 
    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK
 
    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N
 
    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covars_
        + 2 * Q_xx * gmm.means_)
 
    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

def gestFinalWeights(wf,ws):
    w = wf*ws
    w = w/sum(w)
    w = w/max(w)
    w = np.round(w*8)
    return w
def getSkeltonWeights(smp,startFrame,endFrame,numFramesinGesture,scale=0.06):
    JointsToUse = ['HipCenter','Spine','ShoulderCenter','Head', \
                    'ShoulderLeft','ElbowLeft','WristLeft','HandLeft',\
                    'ShoulderRight','ElbowRight','WristRight','HandRight']
    Joints3D = np.zeros((numFramesinGesture,36))
    centralFrame = int(numFramesinGesture/2)
#    pdb.set_trace()
    for numFrame in range(startFrame,endFrame):
        idx = numFrame - startFrame
        skel=smp.getSkeleton(numFrame);
        JointsData = skel.getWorldCoordinates();
        data = []
        for joint in JointsData:
#            print joint
            if joint in JointsToUse:
#                print joint
                data.extend(JointsData[joint])
        
        Joints3D[idx,:] = data;
    sio.savemat('data.mat',mdict = {'J':Joints3D})
    weights =  np.zeros(numFramesinGesture)
    for numFrame in range(startFrame,endFrame):
        idx = numFrame - startFrame
        diff = Joints3D[idx,:]-Joints3D[centralFrame-1,:]
        weights[idx]=np.exp(-1*sum((diff)**2)/scale)
    return weights
def myfilter(Input,siz,thresh):
    h = np.ones(siz)
#    print np.shape(np.asarray(Input[0]))
    out = scipy.convolve(Input[0,:],h,mode='same')
    return out>thresh
def getweights(numFramesinGesture,scale = 110.0):
    weights =  np.zeros(numFramesinGesture)
    centralFrame = int(numFramesinGesture/2)
#    print centralFrame
    for i in xrange(numFramesinGesture):
        weights[i] = -1.0*((centralFrame-i-1)**2.0)/scale
    weights = np.exp(weights)
#    print weights
    
#    weights = -1*weights
#    weights = np.exp(weights)
    return weights
def power_normalize(xx, alpha=0.5):
    """Computes a alpha-power normalization for the matrix xx."""
    return np.sign(xx) * np.abs(xx) ** alpha
    
def L2_normlize_bacths(xx,nbc,D):
    nE = nbc*(2*D+1)
    for idx in range(10):
        xx[:,idx*nE:(idx+1)*nE] = L2_normalize(xx[:,idx*nE:(idx+1)*nE])
    return xx
def L2_normalize(xx):
    """L2-normalizes each row of the data xx."""
    Zx = np.sum(xx * xx, 1)
    xx_norm = xx / np.sqrt(Zx[:, np.newaxis])
    xx_norm[np.isnan(xx_norm)] = 0
    return xx_norm
    
def UnzipAllfiles(datapath):
    
    fileList = os.listdir(datapath)

#     Filter input files (only ZIP files)

    samplelist=[file for file in fileList if file.endswith(".zip")]

#    print samplelist
    for sample in samplelist:
        seqID=os.path.splitext(sample)[0]
        dst_folder = os.path.join(datapath,seqID)
        if not os.path.isdir(dst_folder):
            os.makedirs(dst_folder)
        src_file = os.path.join(datapath,sample);
        datafilename = '{}{}'.format(seqID,'_data.csv')
        skelfilename = '{}{}'.format(seqID,'_skeleton.csv')
        print datafilename
        with zipfile.ZipFile(src_file,"r") as z:
            for name in [datafilename,skelfilename]:
                z.extract(name,dst_folder)
        print 'done unZiping',seqID
        os.remove(src_file)
        
        
def ComputeQuadDescriptor(quadrupleJoints, Single = False,isAll=False):
  
    # If order of point is given return one quad descriptor
    # else put points in order and return one descriptor
    
    if Single:
        Quads0 = SimilarityNormTransform(quadrupleJoints,Single)        
        if not isAll:
            return Quads0
        else:        
            Quads1 =  np.concatenate((Quads0[3:], Quads0[:3]),axis=None)
            Neworderpoints = quadrupleJoints[[1,0,2,3],:]
            Quads2 =  SimilarityNormTransform(Neworderpoints,Single)
            Quads3 = np.concatenate((Quads2[3:], Quads1[:3]),axis=None)
            return np.array([Quads0,Quads1,Quads2,Quads3])
    ## Else compute four combination of Joitnst point Two by swaping
    # most widely sperated points(mean firt two of orderedpoints) and
    # Two by swaping last two points of orderedpoints  
    else:
        orderedpoints = getoderedPoints(quadrupleJoints,FixOrigin=True)
        Quads0 =  SimilarityNormTransform(orderedpoints)
        Quads1 =  np.concatenate((Quads0[3:], Quads0[:3]),axis=None)
        Neworderpoints = orderedpoints[[1,0,2,3],:]
        Quads2 =  SimilarityNormTransform(Neworderpoints)
        Quads3 = np.concatenate((Quads2[3:], Quads1[:3]),axis=None)
        
        return np.array([Quads0,Quads1,Quads2,Quads3])
#         print 'Quads',np.shape(QuadDescriptors)
        

def getoderedPoints(mat, FixOrigin = True):
    indT = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
#     print 'mat',mat
    dismat = np.zeros(6)
    dismat[0] = np.sum(pow(sp.diff(mat[[0, 1],:],axis = 0),2),axis=1)
    dismat[1] = np.sum(pow(sp.diff(mat[[0, 2],:],axis = 0),2),axis=1)
    dismat[2] = np.sum(pow(sp.diff(mat[[0, 3],:],axis = 0),2),axis=1)
    dismat[3] = np.sum(pow(sp.diff(mat[[1, 2],:],axis = 0),2),axis=1)
    dismat[4] = np.sum(pow(sp.diff(mat[[1, 3],:],axis = 0),2),axis=1)
    dismat[5] = np.sum(pow(sp.diff(mat[[2, 3],:],axis = 0),2),axis=1)
#     print 'dis', dismat
    index =  np.argmax(dismat)
    i = indT[index,0];
    j = indT[index,1];
    
    for ind in indT:
#         print ind;
        if (ind[0]!=i and ind[1]!=i and ind[0]!=j and ind[1]!=j):
            otherindexs = ind
#             print 'we are here'
#     print 'i', i, 'j', j, 'other' , otherindexs
    Origin  = i; Unity = j;
    if FixOrigin:
#         print np.sum(pow(mat[i,:],2)), np.sum(pow(mat[j,:],2))
        if np.sum(pow(mat[i,:],2)) > np.sum(pow(mat[j,:],2)):
            Origin = j
            Unity = i
#     print Origin,Unity,otherindexs[0],otherindexs[1]
    orderIndexs = np.array([Origin,Unity,otherindexs[0],otherindexs[1]])
    orderedPoints = mat[orderIndexs,:]
    return orderedPoints
    
    
def SimilarityNormTransform(Pin,Single=False):
    #  similarity normalization transform
    #  Pin = [p1 p2 p3 p4]; 4x3 matrix with four 3D points
    #  Pout is the similarity normalization transform of Pin with respect
    #  to p1,p2, so that p1 goes to (0,0,0) and p2 goes to (1,1,1).
    # 
    #  The order of p1,p2,p3,p4 assumes that p1,p2 is the most widely separated
    #  pair of points and p1 is the closest to the camera. Also, between p3 and
    #  p4, p3 is the nearest to p1.

    P = Pin
    T = -1*P[[0,0,0],:]
    p2 = np.subtract(P[1,:], P[0,:])
    theta1 = math.atan2(p2[1],p2[0])
    phiXY1 = -theta1+sp.pi/4;
    c1 = math.cos(phiXY1);
    s1 = math.sin(phiXY1);
    Rz1 = np.array(([[c1, -s1, 0], [s1, c1, 0], [ 0, 0, 1]]));
    p2_1 = c1*p2[0]-s1*p2[1];
    p2[1] = s1*p2[0]+c1*p2[1];
    p2[0] = p2_1
    no2 = np.sum(p2[:2]*p2[:2])
    no = 2*(no2+p2[2]*p2[2]);
    pp = sum(p2[:2])/math.sqrt(no);
    if pp>1:
#        print 'we have cos angle greater than 1.0 and it is ', pp
        pp = 1.0
    phi = math.acos(pp);
    if p2[2]>0:
        phi = -phi+0.615479708670387;
    else:
        phi = phi+0.615479708670387;
    
    r = np.array([p2[1], -p2[0], 0]);
    r = r/math.sqrt(no2)
    
    C = math.cos(phi)
    S = math.sin(phi)
    F = 1-C
    
    RR = np.array(([[F*pow(r[0],2)+C, F*r[0]*r[1], S*r[1]],
                   [F*r[0]*r[1], F*pow(r[1],2)+C, -S*r[0]],
                   [-S*r[1], S*r[0], C]]));
    P = P[1:,:];
    P = np.transpose(P+T);
    P = RR.dot(Rz1.dot(P))
    P = P/P[0,0]
    if Single:
        QaudDescritor = np.array([P[0,1],P[1,1],P[2,1],P[0,2],P[1,2],P[2,2]])
    else:
        N1 = math.sqrt(np.sum(P[:,1]*P[:,1]));
        N2 = math.sqrt(np.sum(P[:,2]*P[:,2]));
    #     print 'norm1',N1,'norm2',N2
        if N1<=N2:
            QaudDescritor = np.array([P[0,1],P[1,1],P[2,1],P[0,2],P[1,2],P[2,2]])
        else:
            QaudDescritor = np.array([P[0,2],P[1,2],P[2,2],P[0,1],P[1,1],P[2,1]])
    return QaudDescritor
        
        
class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['HipCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Spine']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Head']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][2]
        return skel
    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getPixelCoordinates().keys():
            p=self.getPixelCoordinates()[node]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,datapath, seqID, openRGB = False, openDepth=False, skel=False, training = False):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.isdir(datapath):
            raise Exception("Sample data path does not exist: ")

        # Prepare sample information
        self.RGBopend = openRGB
        self.Depthopened = openDepth
        self.dataPath = datapath
        self.seqID=seqID
        self.samplePath=os.path.join(datapath,seqID)

        # Open video access for RGB information
        if openRGB:
            rgbVideoPath=self.samplePath + os.path.sep +  self.seqID + '_color.mp4'
            if not os.path.exists(rgbVideoPath):
                raise Exception("Invalid sample file. RGB data is not available")
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            while not self.rgb.isOpened():
                self.rgb = cv2.VideoCapture(rgbVideoPath)
                print "Video is already opend"
                cv2.waitKey(500)
        
        # Open video access for Depth information
        if openDepth:
            depthVideoPath=self.samplePath + os.path.sep + self.seqID + '_depth.mp4'
            if not os.path.exists(depthVideoPath):
                raise Exception("Invalid sample file. Depth data is not available")
            self.depth = cv2.VideoCapture(depthVideoPath)
            while not self.depth.isOpened():
                self.depth = cv2.VideoCapture(depthVideoPath)
                cv2.waitKey(500)
                # Open video access for User segmentation information
            userVideoPath=self.samplePath + os.path.sep + self.seqID + '_user.mp4'
            if not os.path.exists(userVideoPath):
                raise Exception("Invalid sample file. User segmentation data is not available")
            self.user = cv2.VideoCapture(userVideoPath)
            while not self.user.isOpened():
                self.user = cv2.VideoCapture(userVideoPath)
                cv2.waitKey(500)
        
        ## Read skeleton data
        if skel:
            skeletonPath=self.samplePath + os.path.sep + self.seqID + '_skeleton.csv'
            if not os.path.exists(skeletonPath):
                raise Exception("Invalid sample file. Skeleton data is not available")
            self.skeletons=[]
            with open(skeletonPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.skeletons.append(Skeleton(row))
                del filereader
            
        # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader
        ## Read labels data
        if training:
            labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
            if not os.path.exists(labelsPath):
                warnings.warn("Labels are not available", Warning)
            self.labels=[]
            with open(labelsPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(map(int,row))
                del filereader
    def clean(self):
        """ Clean temporal unziped data """
        del self.rgb;
        del self.depth;
        del self.user;
        shutil.rmtree(self.samplePath)
    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame
    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)
    def getDepth(self, frameNum):
        """ Get the depth image for the given frame """
        #get Depth frame
        depthData=self.getFrame(self.depth,frameNum)
        # Convert to grayscale
        depthGray=cv2.cvtColor(depthData,cv2.cv.CV_RGB2GRAY)
        # Convert to float point
        depth=depthGray.astype(numpy.float32)
        # Convert to depth values
        depth=depth/255.0*float(self.data['maxDepth'])
        depth=depth.round()
        depth=depth.astype(numpy.uint16)
        return depth
    def getUser(self, frameNum):
        """ Get user segmentation image for the given frame """
        #get user segmentation frame
        return self.getFrame(self.user,frameNum)
    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]
    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,480,(255,255,255))

    def getNumFrames(self):
        """ Get the number of frames for this sample """
        return self.data['numFrames']

    def getComposedFrame(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        skel=self.getSkeletonImage(frameNum)

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize1=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        compSize2=(max(user.shape[0],skel.shape[0]),user.shape[1]+skel.shape[1])
        comp = numpy.zeros((compSize1[0]+ compSize2[0],max(compSize1[1],compSize2[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]=depth
        comp[compSize1[0]:compSize1[0]+user.shape[0],:user.shape[1],:]=user
        comp[compSize1[0]:compSize1[0]+skel.shape[0],user.shape[1]:user.shape[1]+skel.shape[1],:]=skel

        return comp
    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels
    def getGestureName(self,gestureID):
        """ Get the gesture label from a given gesture ID """
        names=('vattene','vieniqui','perfetto','furbo','cheduepalle','chevuoi','daccordo','seipazzo', \
               'combinato','freganiente','ok','cosatifarei','basta','prendere','noncenepiu','fame','tantotempo', \
               'buonissimo','messidaccordo','sonostufo')
        # Check the given file
        if gestureID<1 or gestureID>20:
            raise Exception("Invalid gesture ID <" + str(gestureID) + ">. Valid IDs are values between 1 and 20")
        return names[gestureID-1]

    def exportPredictions(self, prediction,predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'wb')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "\n")
        output_file.close()

    def evaluate(self,csvpathpred):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=11
        seqLength=self.getNumFrames()

        # Get the list of gestures from the ground truth and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqLength))
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqLength))
        with open(csvpathpred, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_pred[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        for row in self.getActions():
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = []
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps.append(intersec/union)

        # Use real gestures and false positive gestures to calculate the final score
        return sum(overlaps)/(len(overlaps)+len(falsePos))
