# CEPrompt
Official implementation and checkpoints for paper "CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition" (accepted to IEEE TCSVT 2024) [![paper](https://img.shields.io/badge/Paper-87CEEB)](https://doi.org/10.1109/TCSVT.2024.3424777) <be> 

### Abstract <br>
Facial expression recognition (FER) remains a challenging task due to the ambiguity and subtlety of expressions. To address this challenge, current FER methods predominantly prioritize visual cues while inadvertently neglecting the potential insights that can be gleaned from other modalities. Recently, vision-language pre-training (VLP) models integrated textual cues as guidance, culminating in a powerful multi-modal solution that has proven effective for a range of computer vision tasks. In this paper, we propose a Cross-Modal Emotion-Aware Prompting (CEPrompt) framework for FER based on VLP models. To make VLP models sensitive to expression-relevant visual discrepancies, we devise an Emotion Conception-guided Visual Adapter (EVA) to capture the category-specific appearance representations with emotion conception guidance. Moreover, knowledge distillation is employed to prevent the model from forgetting the pre-trained category-invariant knowledge. In addition, we design a Conception-Appearance Tuner (CAT) to facilitate the interaction of multi-modal information via cooperatively tuning between emotion conception and appearance prompts. In this way, semantic information about emotion text conception is infused directly into facial appearance images, thereby enhancing a comprehensive and precise understanding of expression-related facial details. Quantitative and qualitative experiments show that our CEPrompt outperforms state-of-the-art approaches on three real-world FER datasets.

<p align="center">
<img src="https://github.com/HaoliangZhou/CEPrompt/blob/master/ceprompt.png" width=100% height=100% 
class="center">
</p>

---
## Installation
1. Installation the package requirements
```
pip install -r requirements.txt
```

2. Download pretrained VLP(ViT-B/16) model from [OpenAI CLIP](https://github.com/openai/CLIP).

---
## Data Preparation
1. The downloaded [RAF-DB](http://www.whdeng.cn/RAF/model1.html) are reorganized as follow:
```
data/
├─ RAF-DB/
│  ├─ basic/
│  │  ├─ EmoLabel/
│  │  │  ├─ images.txt
│  │  │  ├─ image_class_labels.txt
│  │  │  ├─ train_test_split.txt
│  │  ├─ Image/
│  │  │  ├─ aligned/
│  │  │  ├─ aligned_224/  # reagliend by MTCNN
```
2. The downloaded [AffectNet](http://mohammadmahoor.com/affectnet/) are reorganized as follow:
```
data/
├─ AffectNet/
│  ├─ affectnet_info/
│  │  ├─ images.txt
│  │  ├─ image_class_labels.txt
│  │  ├─ train_test_split.txt
│  ├─ Manually_Annotated_Images/
│  │  ├─ 1/
│  │  │  ├─ images
│  │  │  ├─ ...
│  │  ├─ 2/
│  │  ├─ ./
```
3. The structure of three data-load and -split txt files are reorganized as follow:
```
% (1) images.txt:
idx | imagename
1 train_00001.jpg
2 train_00002.jpg
.
15339 test_3068.jpg

% (2) image_class_labels.txt:
idx | label
1 5
2 5
.
15339 7

% (3) train_test_split.txt:
idx | train(1) or test(0)
1 1
2 1
.
15339 0
```

---
## Model checkpoints
- Download model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1xd85nIySAkoMZQr281HbaEFFhkKHaHAA?usp=sharing).

---
## Training
### Train First Stage (EVA)
```
python3 train_fer_first_stage.py \  
--dataset ${DATASET} \ 
--data-path ${DATAPATH}
```
### Train Second Stage (CAT)
```
python3 train_fer_second_stage.py \  
--dataset ${DATASET} \  
--data-path ${DATAPATH} \  
--ckpt-path ${CKPTPATH}
```
### You can also run the script
```
bash stage1.sh
```
```
bash stage2.sh
```

---
## Evaluation
```
python3 train_fer_second_stage.py \ 
--eval \
--dataset ${DATASET} \       # dataset name
--data-path ${DATAPATH} \    # path to dataset
--ckpt-path ${CKPTPATH} \    # path to first stage ckpt
--eval-ckpt ${EVACKPTPATH}   # path to second stage ckpt
```

---
## Cite Our Work
If you find our work helps, please cite our paper.
```
@ARTICLE{Zhou2024CEPrompt,
  author={Zhou, Haoliang and Huang, Shucheng and Zhang, Feifei and Xu, Changsheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition}, 
  year={2024},
  volume={34},
  number={11},
  pages={11886-11899},
  doi={10.1109/TCSVT.2024.3424777}
}

``` 

---
## Contact
For any questions, welcome to create an issue or email to <a href="mailto:haoliangzhou6@gmail.com">haoliangzhou6@gmail.com</a>.




