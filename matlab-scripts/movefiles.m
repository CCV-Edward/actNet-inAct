

function convertVid2Images()
baseDir = '/mnt/earth-beta/Datasets/actnet/';
vidDir = [baseDir,'videos/'];
%dirlist = dir([vidDir,'*.mp4']);
%vidlist = sort({dirlist.name});
% fid = fopen('temp.txt','w');
%save('dirlist.mat','vidlist');
load('dirlist.mat');
newlist = [];
if 0
for i=1:length(vidlist)
  videoName = vidlist{i};
  imgPath = [baseDir,'images/',videoName(1:end-4)];
  if exist(imgPath, 'dir')
	newlist = [newlist;i];
  end
end
end
%mkdir('/mnt/sun-alpha/actnet/videos/');
load('newlistRemainmorethan8.mat');
fprintf('files to be done are %d\n',length(newlist))
if 1
for i =  10701:length(newlist)
    vid = newlist(i);
    processone(vid,vidlist,vidDir,baseDir)
end
end

function processone(vid,vidlist,vidDir,baseDir)

videoName = vidlist{vid};
vidPath = [vidDir,videoName];

newvidpath = ['/mnt/sun-alpha/actnet/videos/',videoName];

fprintf('Moving %s to %s %d\n',vidPath,newvidpath,vid);
try
movefile(vidPath,newvidpath);
catch
fprintf('Can not move above file\n');
end
if 0
%if exist(imgPath, 'dir')
    fprintf('We are ')
    %         fprintf(fid,'%s\n',imgPath);
    fprintf('Doing %s\n',vidPath);
    
    vidobj = VideoReader(vidPath);
    lastFrame = read(vidobj, inf);
    
    NumberOfFrames = vidobj.NumberOfFrames;
    imagname = sprintf('%s/%05d.jpg',imgPath,NumberOfFrames-1);
    fprintf('Imagename %s %d %d %d\n',imagname,exist(imagname,'file'),NumberOfFrames,vid);

    if ~exist(imagname,'file')
        width = vidobj.Width;
        height = vidobj.Height;
        
        if width>height;
            ressizes =  [256, NaN];
        else
            ressizes =  [NaN, 256];
        end
        framecount = 0;
        mkdir(imgPath)
        for f = 1:NumberOfFrames
	   % fprintf('We aredoing frmae  %d\n',f)
            image = read(vidobj,f);
            imagname = sprintf('%s/%05d.jpg',imgPath,framecount);
            resizedimage = imresize(image,ressizes);
            imwrite(resizedimage,imagname)
            framecount = framecount + 1;
        end
    else
    fprintf('This video is alll comlete\n');
    end

end
