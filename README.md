
##

This repository hosts the batch inference code of LightGlue, a lightweight feature matcher with high accuracy and blazing fast inference. It takes as input a set of keypoints and descriptors for each image and returns the indices of corresponding points. The architecture is based on adaptive pruning techniques, in both network width and depth - [check out the paper for more details](https://arxiv.org/pdf/2306.13643.pdf).



There are pretrained weights of LightGlue with [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566) local features.




## Data

The data needs to be in the following format:

```
<data_root>            # datadir 
├── <scene001>             
  ├── images           # RGB images
      ├── <00000>.jpg     
      ├── <00001>.jpg
      ├── <00002>.jpg
      ...
├── <scene002>      
  ├── images           # RGB images
      ├── <00000>.jpg     
      ├── <00001>.jpg
      ├── <00002>.jpg
      ...
├── <scene003>      
  ├── images           # RGB images
      ├── <00000>.jpg     
      ├── <00001>.jpg
      ├── <00002>.jpg
      ...
```



## UI links

- [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui): a web GUI to easily compare different matchers, including LightGlue.


## BibTeX Citation
If you use any ideas from the paper or code from this repo, please consider citing:

```txt
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and
               Paul-Edouard Sarlin and
               Marc Pollefeys},
  title     = {{LightGlue: Local Feature Matching at Light Speed}},
  booktitle = {ICCV},
  year      = {2023}
}
```


