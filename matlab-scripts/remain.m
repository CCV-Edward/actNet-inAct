%% Write something here
function remain()
baseDir = '/mnt/earth-beta/Datasets/actnet/';
vidDir = [baseDir,'videos/'];

load('vidlist.mat');
load('newlistlessthan8.mat')
fprintf('files to be done are %d\n',length(newlist))

if 1
    for i = 1:length(newlist)
        vid = newlist(i);
        processone(vid,vidlist,vidDir,baseDir,i)
    end
end

function processone(vid,vidlist,vidDir,baseDir,ii)
videoName = vidlist{vid};
vidPath = [vidDir,videoName];
imgPath = [baseDir,'images/',videoName(1:end-4)];
fprintf('We are Doing %s %d %d\n',vidPath,vid,ii);
try
    vidobj = VideoReader(vidPath);    
    width = vidobj.Width;
    height = vidobj.Height;
    
    if width>height;
        ressizes =  [256, NaN];
    else
        ressizes =  [NaN, 256];
    end
    framecount = 0;
    if ~exist(imgPath, 'dir')
        mkdir(imgPath)
    end
    while hasFrame(vidobj)
        %	    fprintf('We aredoing frmae  %d\n',framecount);
        image = readFrame(vidobj);
        imagname = sprintf('%s/%05d.jpg',imgPath,framecount);
        resizedimage = imresize(image,ressizes);
        imwrite(resizedimage,imagname)
        framecount = framecount + 1;
    end
catch
    fprintf('We have problem')
    fid = fopen(sprintf('vids-rem/open-%s.txt',videoName(1:end-4)),'w');
    fclose(fid);
end