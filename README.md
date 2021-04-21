# Dataset Inference: Ownership Resolution in Machine Learning

Repository for the paper [Dataset Inference: Ownership Resolution in Machine Learning](https://openreview.net/pdf?id=hvdKKV2yt7T) by [Pratyush Maini](https://pratyushmaini.github.io), [Mohammad Yaghini]() and [Nicolas Papernot](https://papernot.fr). This work was presented at [ICLR 2021](http://iclr.cc/Conferences/2021/) as a Spotlight Presentation.

## What does this repository contain?
Code for training and evaluating all the experiments that support the aforementioned paper are provided in this repository. 
The instructions for reproducing the results can be found below.

## Dependencies
The repository is written using `python 3.8`. To install dependencies run the command:

`pip install -r requirements.txt` (TODO)

## Resolving Ownership
If you already have the extracted featured for the victim and potentially stolen models, you can proceed to inferring potential theft. A sample `jupyter notebook` to perform the same can be found at:
`src/notebooks/CIFAR10_rand.ipynb`   
You can download extracted features for our models from [this link](https://drive.google.com/drive/folders/1CLJ2a3H_oTX5b_4GLurVYoCpZUXQVpFr?usp=sharing). Save them in a directory names `files` in the root directory.

## Training your own models
`python train.py --batch_size 1000 --mode $MODE --normalize $NORMALIZE --model_id $MODEL_ID --lr_mode $LR_MODE --epochs $EPOCHS --dataset $DATASET --lr_max $LR_MAX --pseudo_labels $PSEUDO`
  > `batch_size` - Batch Size for Test Set -`default = 1000`  
  > `mode` - "Various attack strategies", type = str, default = 'teacher', choices = ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18']  
  > `normalize`  - The normalization is performed within the model and not in the dataloader to ease adversarial attack implementation. Please take note.  
  > `model_id` - Used to compute location to load the model. See directory structure in code. 
  > `pseudo_labels` - Used in case of label only model extraction

## Generating Features
`python generate_features.py --batch_size 500 --mode $MODE --normalize $NORMALIZE --model_id $MODEL_ID --dataset $DATASET --feature_type $FEATURE`
  > `batch_size` - Batch Size for Test Set -`default = 500`  
  > `mode` - "Various attack strategies", type = str, default = 'teacher', choices = ['zero-shot', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher','independent','pre-act-18']  
  > `normalize`  - The normalization is performed within the model and not in the dataloader to ease adversarial attack implementation. Please take note.  
  > `model_id` - Used to compute location to load the model. See directory structure in code.   
  > `feature_type` - 'topgd', 'mingd', 'rand'. For black-box method use Random

