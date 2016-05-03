function convertVid2Images()

baseDirVid = '/mnt/sun-alpha/actnet/';
vidDir = [baseDirVid,'videos/'];
baseDirImg = '/mnt/earth-beta/Datasets/actnet/';
imageDir = [baseDirImg,'images/'];
dirlist = dir([vidDir,'*.mp4']);
vidlist = sort({dirlist.name});
fprintf('files to be done are %d\n',length(vidlist))

parfor vid = 1001:10000
    processone(vid,vidlist,vidDir,imageDir)
end

function processone(vid,vidlist,vidDir,imageDir)

videoName = vidlist{vid};
vidPath = [vidDir,videoName];
imgPath = [imageDir,videoName(1:end-4)];

fprintf('We are ')
fprintf('Doing %s ',vidPath);

try
    vidobj = VideoReader(vidPath);
    lastFrame = read(vidobj, inf);    
    NumberOfFrames = vidobj.NumberOfFrames;
    imagname = sprintf('%s/%05d.jpg',imgPath,NumberOfFrames-1);
    fprintf('\nImagename %s %d %d %d \n',imagname,exist(imagname,'file'),NumberOfFrames,vid);
    
    if ~exist(imagname,'file') && NumberOfFrames>0
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
        fprintf('This video is alll comlete and has %d frames\n',NumberOfFrames);
    end
    
catch
    fprintf(' !!!problem!!!!!!\n')
    movefile(vidPath,['/mnt/earth-beta/Datasets/actnet/videos/',videoName]);
    fid = fopen(sprintf('vids/%s.txt',videoName(1:end-4)),'w');
    fclose(fid);
end

