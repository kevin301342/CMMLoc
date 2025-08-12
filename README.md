# CMMLoc
This repository is the official implementation of our CVPR 2025 paper: 

[CMMLoc: Advancing Text-to-PointCloud Localization with Cauchy-Mixture-Model Based Framework](https://arxiv.org/abs/2503.02593).

## Introduction
We discover the partial relevance characteristc in language-based point cloud localization task in large-scale urban environments and propose a Cauchy-Mixture-Model based framework CMMLoc to tackle the challenge. Given a city-scale point cloud and a textual query describing a target location, CMMLoc identifies the most likely position corresponding to the described location within the map. By modeling submaps with a Cauchy Mixture Model and incorporating additional designs that facilitate fine-grained interactions, our CMMLoc significantly outperforms previous approaches.


## Installation
Create a conda environment and install basic dependencies:
```bash
git clone git@github.com:anonymous0819/CMMLoc.git
cd CMMLoc

conda create -n cmmloc python=3.10
conda activate cmmloc

# Install the according versions of torch and torchvision
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install required dependencies
pip install -r requirements.txt
```

## Datasets & Backbone

The KITTI360Pose can be accessed [HERE](https://drive.google.com/drive/folders/1Zt2vFAMRqa780foh6asd4xjSEWiaYVzs)   

If you want to train the model, you need to download the pretrained object backbone [HERE](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view?usp=drive_link):

The KITTI360Pose and the pretrained object backbone is provided by Text2Pos ([paper](https://arxiv.org/abs/2203.15125), [code](https://github.com/mako443/Text2Pos-CVPR2022))

The direction folder is provided by Text2Loc([paper](https://arxiv.org/abs/2311.15977), [code](https://github.com/Yan-Xia/Text2Loc))

The final directory structure should be:

```
│CMMLoc/
├──dataloading/
├──datapreparation/
├──data/
│   ├──k360_30-10_scG_pd10_pc4_spY_all/
│       ├──cells/
│           ├──2013_05_28_drive_0000_sync.pkl
│           ├──2013_05_28_drive_0002_sync.pkl
│           ├──...
│       ├──poses/
│           ├──2013_05_28_drive_0000_sync.pkl
│           ├──2013_05_28_drive_0002_sync.pkl
│           ├──...
│       ├──direction/
│           ├──2013_05_28_drive_0000_sync.json
│           ├──2013_05_28_drive_0002_sync.json
│           ├──...
├──checkpoints/
│   ├──pointnet_acc0.86_lr1_p256.pth
├──...
```

## Load Pretrained Modules

Our pre-trained models are publicly available [Here](https://drive.google.com/drive/folders/1Hhml9yDpsuEzB4v8X1IXTMFpUHHUrSE0?usp=sharing). To run the evaluation, save them under

```
./checkpoints/coarse.pth
./checkpoints/fine.pth
./checkpoints/prealign_pointnet.pth
./checkpoints/prealign_color_encoder.pth
./checkpoints/prealign_mlp.pth
```

You should specify pointnet_path in args.py and the PATH_TO_PRETRAINED in the language_encoder.py and object_encoder.py in the fine stage.

## Train

After setting up the dependencies and dataset, our models can be trained using the following commands:

### Train Coarse Submap Retrieval

```bash
python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/   \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0005 \
  --lr_scheduler step \
  --lr_step 7 \
  --lr_gamma 0.4 \
  --temperature 0.1 \
  --ranking_loss contrastive \
  --hungging_model t5-large \
  --folder_name PATH_TO_COARSE
```

### Pre-alignment

```python
python -m training.prealign --batch_size 32 --fine_embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_PRE_ALIGN
```

### Train Fine Localization

```bash
python -m training.fine --batch_size 32 --fine_embed_dim 128 --shuffle --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 45 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-large \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --folder_name PATH_TO_FINE
```

## Evaluation

### Evaluation on Val Dataset

```bash
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```

### Evaluation on Test Dataset

```bash
python -m evaluation.pipeline --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/ \
    --use_features "class"  "color"  "position"  "num" \
    --use_test_set \
    --no_pc_augment \
    --no_pc_augment_fine \
    --hungging_model t5-large \
    --fixed_embedding \
    --path_coarse ./checkpoints/{PATH_TO_COARSE}/{COARSE_MODEL_NAME} \
    --path_fine ./checkpoints/{PATH_TO_FINE}/{FINE_MODEL_NAME} 
```
