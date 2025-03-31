# Transformer as Linear Expansion of Learngene (AAAI 2024)


## Running TLEG

We provide the following shell codes for TLEG running. 

### Stage1: Training Aux-Net to obtain learngenes

In the first stage, we train Aux-Ti/S/B to obtain learngenes or you can directly download from [here](https://drive.google.com/drive/folders/1TROwdWr-V7Q1bAUFDH6zCZD4yANfDa6g).

### Training Aux-Ti

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 23001 \
        --use_env main.py \
        --config /configs/conf_train_aux_ti.yaml
```

### Training Aux-S

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 23002 \
        --use_env main.py \
        --config /configs/conf_train_aux_s.yaml
```

### Training Aux-B

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 23003 \
        --use_env main.py \
        --config /configs/conf_train_aux_b.yaml
```


### Stage2: Training Des-Net after initializing with learngenes

In the second stage, we train Des-Ti/S/B after initializing them with learngenes.


### Training Des-Ti

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 27001 \
        --use_env ./main.py \
        --config /configs/conf_train_des_ti.yaml
```
For training Des-Ti of different layers, you can choose `model-type` from [deit_tiny_patch16_224_L3, deit_tiny_patch16_224_L6, deit_tiny_patch16_224_L9, deit_tiny_patch16_224_L12].

Make sure you update the tiny learngene path `/path/to/tiny_learngene`, where you place the tiny learngene trained from the first stage.

### Training Des-S

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 28001 \
        --use_env ./main.py \
        --config /configs/conf_train_des_s.yaml
```
For training Des-S of different layers, you can choose `model-type` from [deit_small_patch16_224_L3, deit_small_patch16_224_L4, deit_small_patch16_224_L5, deit_small_patch16_224_L6, deit_small_patch16_224_L7, deit_small_patch16_224_L8, deit_small_patch16_224_L9, deit_small_patch16_224_L10, deit_small_patch16_224_L11, deit_small_patch16_224_L12], where we also train deit_small_patch16_224_L11 with 45 epochs for better performance.

Make sure you update the small learngene path `/path/to/small_learngene`, where you place the small learngene trained from the first stage.

### Training Des-B

```bash
python3 -u -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 29001 \
        --use_env ./main.py \
        --config /configs/conf_train_des_b.yaml
```
For training Des-B of different layers, you can choose `model-type` from [deit_base_patch16_224_L3, deit_base_patch16_224_L4, deit_base_patch16_224_L5, deit_base_patch16_224_L6, deit_base_patch16_224_L7, deit_base_patch16_224_L8, deit_base_patch16_224_L9, deit_base_patch16_224_L10, deit_base_patch16_224_L11, deit_base_patch16_224_L12].

Make sure you update the base learngene path `/path/to/base_learngene`, where you place the base learngene trained from the first stage.
