from argparse import ArgumentParser
import glob
import json
import os

def crosscheck_videos(video_path, ann_file):
    # Get existing videos
    existing_vids1 = glob.glob("%s/*.mp4" % video_path)
    existing_vids2 =  glob.glob("/mnt/sun-alpha/actnet/videos/*.mp4");
    print len(existing_vids1),"second one",len(existing_vids2)
    count = 0;
    existing_vids = [' ' for a in range(len(existing_vids1)+len(existing_vids2))]
    for idx, vid in enumerate(existing_vids1):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[count] = basename[2:]
            count+=1
        elif len(basename) == 11:
            existing_vids[count] = basename
            count+=1
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    for idx, vid in enumerate(existing_vids2):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[count] = basename[2:]
            count+=1
        elif len(basename) == 11:
            existing_vids[count] = basename
            count+=1
        else:
            raise RuntimeError("Unknown filename format: %s", vid)

    # Read an get video IDs from annotation file
    print "existing videos are ",len(existing_vids)
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    all_vids = anet_v_1_0["database"].keys()
    non_existing_videos = []
#    vidc = 0;
    for vid in reversed(all_vids):
#	vid = all_vids[vidc];
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)
    return non_existing_videos

def main(video_path, ann_file, output_filename):
    non_existing_videos = crosscheck_videos(video_path, ann_file)
    print "Number of videos remaing are ",len(non_existing_videos)
    filename = os.path.join(video_path, "v_%s.mp4")
    cmd_base = "youtube-dl -f best -f mp4 "
    cmd_base += '"https://www.youtube.com/watch?v=%s" '
    cmd_base += '-o "%s"' % filename
    with open(output_filename, "w") as fobj:
        for vid in non_existing_videos:
            cmd = cmd_base % (vid, vid)
            fobj.write("%s\n" % cmd)

if __name__ == "__main__":
    parser = ArgumentParser(description="Script to double check video content.")
    parser.add_argument("video_path", help="Where are located the videos? (Full path)")
    parser.add_argument("ann_file", help="Where is the annotation file?")
    parser.add_argument("output_filename", help="Output script location.")
    args = vars(parser.parse_args())
    main(**args)
