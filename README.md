
##

This repository hosts the inference code of LightGlue, a lightweight feature matcher with high accuracy and blazing fast inference. It takes as input a set of keypoints and descriptors for each image and returns the indices of corresponding points. The architecture is based on adaptive pruning techniques, in both network width and depth - [check out the paper for more details](https://arxiv.org/pdf/2306.13643.pdf).

There are pretrained weights of LightGlue with [SuperPoint](https://arxiv.org/abs/1712.07629) and [DISK](https://arxiv.org/abs/2006.13566) local features.


## Installation and demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb)

Install this repo using pip:

```bash
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

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



## Other links
- [hloc - the visual localization toolbox](https://github.com/cvg/Hierarchical-Localization/): run LightGlue for Structure-from-Motion and visual localization.
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX): export LightGlue to the Open Neural Network Exchange format.
- [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui): a web GUI to easily compare different matchers, including LightGlue.
- [kornia](https://kornia.readthedocs.io) now exposes LightGlue via the interfaces [`LightGlue`](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LightGlue) and [`LightGlueMatcher`](https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LightGlueMatcher).

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


## License
The pre-trained weights of LightGlue and the code provided in this repository are released under the [Apache-2.0 license](./LICENSE). [DISK](https://github.com/cvlab-epfl/disk) follows this license as well but SuperPoint follows [a different, restrictive license](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE) (this includes its pre-trained weights and its [inference file](./lightglue/superpoint.py)).
