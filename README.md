# Introduction
This repository was intended to host work in progress that I did in ActivityNet2016 
We recived `2nd prize` in `detection task` of the challege. Also, this secured `1st position` on average mAP.
Also, it was psotioned at 10th place in classification task.
 
This work described in tenchnical report available at [arxiv](https://arxiv.org/abs/1607.01979).

Please refer to Evalation page of the [challenge] (http://activity-net.org/challenges/2016/evaluation.html) to check the standing.

## Repository Structure

This repository is built upon the fork of [activitynet/ActivityNet](https://github.com/activitynet/ActivityNet) repo.

`Crawler` is modified for personal use, you can use it from original [repo] as well.

`python-script` contains the script to convert `.mp4` video int `.jpg` frames produce the level labels. 
Originally temporal labelling is done in from start time and end time in 'seconds'. We resave these annotation in as the label of frames.

There are two scripts to extract frames.
matlab script implement the functionality of reading the video and converting it into frames
python script has same ability and much faster in doing that

Then, there is python script to re-save the annotation according to frame level annotations 
and a action class ID; action class IDs are from 1-200 continuous number, unlike nodeId that is provided. This converstion reversable as well.

`processing` contains python script to train SVM on different kind of feature and save frame level score vector for each frame of validation and testing videos.

Further, there are two main scripts to produce `classification and `detection' results. Detection script contains the implementation dynamic programming based solver to solve labelling problem intially proposed in [[1]](https://hal.archives-ouvertes.fr/hal-01082981/document).

### Citing 
If you find this work useful in your research, please consider citing:

	@article{singh2016untrimmed,
	   title={Untrimmed Video Classification for Activity Detection: submission to ActivityNet Challenge},
	   author={Singh, Gurkirt and Cuzzolin, Fabio},
	   journal={arXiv preprint arXiv:1607.01979},
	   year={2016}
	 }

This code is released without any gurantee. 

## Reference
[1] Georgios, Evangelidis, Gurkirt Singh and Radu Horaud. "Continuous gesture recognition from articulated poses." Workshop at the European Conference on Computer Vision. Springer International Publishing, 2014.
