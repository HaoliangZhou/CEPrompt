# CEPrompt
Official implementation and checkpoints for paper "CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition"

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

[comment]: <> (---)

[comment]: <> (## Cite Our Work)

[comment]: <> (If you find our work helps, please cite our paper.)

[comment]: <> (```bibtex)

[comment]: <> (```)

[comment]: <> (---)

[comment]: <> (## Contact)

[comment]: <> (For any questions, welcome to create an issue or email Haoliang &#40;[haoliangzhou6@gmail.com]&#40;mailto:haoliangzhou6@gmail.com&#41;&#41;.)


[comment]: <> (---)

[comment]: <> (## Acknowledgement)

[comment]: <> (- This codebase is based on [A]&#40;https:&#41; and [B]&#40;https:&#41; -- we thank the authors for their amazing works.)



