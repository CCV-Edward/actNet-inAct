# Introduction
This repository was intended to host work in progress that I did in ActivityNet2016 
We recived 'second prize' in 'detection task' of the challege. Also, this secured 'first position' on average mAP.
This work was psotioned at 10th place in 'classification task'. 

Please refer to Evalation page for challenge [results](http://activity-net.org/challenges/2016/evaluation.html) to check the standing. Few new methods has appeared after challege on this page.

This work described in tenchnical report available at [arxiv](https://arxiv.org/abs/1607.01979).

## Repository Structure

This repository is built upon the fork of [activitynet/ActivityNet](https://github.com/activitynet/ActivityNet)

Crawler is modified for personal use, you can use it from original [repo] as well. It might be latest there.

matlab script implement the functionality of reading the video and converting it into frames
python script has same ability and much faster in doing that
in add to that python also has script to re-save the annotation according to frame level annotations 
and a action class ID; action class IDs are from 1-200 continuous number, unlike nodeId that is provided

### Citing 

If you find this work useful in your research, please consider citing:

	@article{singh2016untrimmed,
	   title={Untrimmed Video Classification for Activity Detection: submission to ActivityNet Challenge},
	   author={Singh, Gurkirt and Cuzzolin, Fabio},
	   journal={arXiv preprint arXiv:1607.01979},
	   year={2016}
	 }

This code is released without any gurantee. 
please let me know if I break somthing on guru094@gmail.com 
