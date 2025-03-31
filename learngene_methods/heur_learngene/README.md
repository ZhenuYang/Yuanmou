# Learngene: From Open-World to Your Learning Task

### Make dataset

Data division refers to [appendix](https://github.com/BruceQFWang/learngene/blob/main/Learngene_Appendix.pdf). You can download the processed Cifar100_division [here](https://drive.google.com/file/d/1MKWi7dsjp3RQkKrcLV7ljZxJ4sm3YTL5/view?usp=sharing).

Make continual data (source domain) and target data(target domain) on the CIFAR100 dataset:

```python
DATAPATH=YOUR_PATH_TO_THE_DATASET
cd utils/datasets
python heur_makedata.py --data_name 'cifar100' --num_imgs_per_cat_train 60 --path ./cifar100-open
```

Make continual data (source domain) and target data(target domain) on the ImageNet100 dataset:

```python
DATAPATH=YOUR_PATH_TO_THE_DATASET
cd utils/datasets
python heur_makedata.py --data_name 'mini-imagenet' --num_imgs_per_cat_train 60 --path ./mini_imagenet
```

### Train Learngene

Please make sure to set the dataset name and the dataset path accordingly.

```python
cd learngene_methods/heur_learngene
python train_learngene.py
```

### Extract Learngene

```python
cd learngene_methods/heur_learngene
python extract_learngene.py
```

 ### Reconsindividual model

Please make sure to set the dataset name and the dataset path accordingly. Reconstruct individual model:

```python
cd learngene_methods/heur_learngene
python initialize_w_learngene.py
```
