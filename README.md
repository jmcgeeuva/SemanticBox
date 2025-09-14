# This is an official pytorch implementation of SemanticBox: Bounding Box-Guided Caption Enhanced Action Recognition for Instructional Videos

## Overview

![SemanticBox](SemanticBox.png)


## Content 
 - [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Uodates](#updates)
- [Pretrained Models](#pretrained-models)
  * [Kinetics-400](#kinetics-400)
  * [Hmdb51 && UCF101](#HMDB51&&UCF101)
- [Testing](#testing)
- [Training](#training)
- [Contributors](#Contributors)
- [Citing_ActionClip](#Citing_ActionCLIP)
- [Acknowledgments](#Acknowledgments)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) >= 1.8
- [wandb](https://wandb.ai/)
- RandAugment
- pprint
- tqdm
- dotmap
- yaml
- csv

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

More detail information about libraries see [INSTALL.md](INSTALL.md).

## Data Preparation
We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.
We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/),
[Charades](https://prior.allenai.org/projects/charades). 
 
## Updates
- We now support single crop validation(including zero-shot) on Kinetics-400, UCF101 and HMDB51. The pretrained models see [MODEL_ZOO.md](MODEL_ZOO.md) for more information.
- we now support the model-training on Kinetics-400, UCF101 and HMDB51 on 8, 16 and 32 frames. The model-training configs see [configs/README.md](configs/README.md) for more information.
- We now support the model-training on your own datasets. The detail information see  [configs/README.md](configs/README.md).

## Pretrained Models
Training video models is computationally expensive. Here we provide some of the pretrained models.
We provide a large set of trained models in the ActionCLIP [MODEL_ZOO.md](MODEL_ZOO.md).                                                 

## Testing 
```
# test
bash scripts/run_test.sh  ./configs/education/ablation/all.yaml
```


## Training
Examples for training:
```
# train 
bash scripts/run_train.sh  ./configs/education/ablation/all.yaml
```

## Reference
ActionCLIP is written and maintained by [Mengmeng Wang](https://sallymmx.github.io/) and [Jiazheng Xing](https://april.zju.edu.cn/team/jiazheng-xing/).

# Acknowledgments
This work is funded in part by NSF under 2322993 and 2000487 and by the Gates Foundation
Views expressed here are those of the authors and do not necessarily reflect positions or policies of Gates the foundation.
<img width="3145" height="207" alt="image" src="https://github.com/user-attachments/assets/9b0fd657-1b51-4be9-b4b7-7c871e1fb6f5" />


