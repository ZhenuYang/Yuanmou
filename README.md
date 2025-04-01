# YuanMou (元谋)

<p align="center">
  <img src="./sources/logo.jpg" alt="logo" width="60%"/>
</p>



## 项目介绍

YuanMou is an open source **Learngene** toolbox based on PyTorch. **Learngene** is a novel machine learning paradigm, which first condense a larger, well-trained model, termed an *ancestry model*, into a tiny critical part as *learngene*, which contains core and generalizable knowledge from the ancestry model. Subsequently, learngene is expanded to initialize many variable-sized models for different downstream tasks and resource constraints, which are called *descendant models*. 
YuanMou 是一个基于 PyTorch 的开源 Learngene 工具箱。Learngene 是一种新颖的机器学习范式，其核心思想是首先将一个较大且经过良好训练的模型（称为 祖先模型）浓缩为一个微小但关键的部分，称为 Learngene，其中包含祖先模型的核心且可泛化的知识。随后，Learngene 可以扩展，用于初始化许多不同规模的模型，以适应不同的下游任务和资源限制，这些模型被称为 后代模型。

Currently, YuanMou includes four **Learngene** algorithms as follows.

+ [Heur Learngene]( https://arxiv.org/abs/2106.06788): AAAI 2022
+ [Auto Learngene](https://arxiv.org/abs/2305.02279): arXiv prePrint 2023
+ [Learngene Pool](https://arxiv.org/abs/2312.05743): AAAI 2024
+ [TLEG](https://arxiv.org/abs/2312.05614): AAAI 2024

<p align="center">
  <img src="./sources/fig1.png" alt="image1" width="50%"/>
</p>


## Advantages

We employed Heur Learngene and Auto Learngene on the currently well-known large language model [**Llama2-7B**](https://ai.meta.com/llama/) and demonstrated that, with vanilla [LoRa](https://arxiv.org/abs/2106.09685) as the baseline, **Learngene** has the following advantages：

+ ### Better performance

  By employing Heur Learngene and Auto Learngene, fine-tuning the same number of epochs on a large language model leads to improved performance.

<p align="center">
  <img src="./sources/fig2.jpg" alt="image2" width="40%"/>
</p>




+ ### Faster convergence

  By employing Heur Learngene and Auto Learngene, the number of epochs required for large language models to converge has been reduced by **30%** and **40%** respectively.

<p align="center">
  <img src="./sources/fig3.jpg" alt="image3" width="40%"/>
</p>




+ ### Fewer GPU days

  By employing Heur Learngene and Auto Learngene, the GPU days required for fine-tuning large language models have been reduced by **30%** and **40%** respectively, thereby reducing resource costs.

<p align="center">
  <img src="./sources/fig4.jpg" alt="image4" width="40%"/>
</p>




+ ### Fewer training samples

  By employing Heur Learngene and Auto Learngene, fine-tuning large language models requires only **60%** and **50%** of the data respectively, further reducing resource costs.

<p align="center">
  <img src="./sources/fig5.jpg" alt="image5" width="40%"/>
</p>



## Get Started

We provide a brief tutorial for YuanMou.

### Clone

```
git clone https://github.com/Learngene-YuanMou/YuanMou.git
cd YuanMou
```



### Requirements

- Python 3.8
- PyTorch 2.0.1 or higher
- torchvison 0.15.2 or higher
- tensorboard
- numpy
- yacs
- tqdm



### Preparing Datasets

| Dataset name  | Categories | Images     | link                                                         |
| ------------- | ---------- | ---------- | ------------------------------------------------------------ |
| CIFAR-100     | 100        | 50,000     | https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz     |
| Mini-ImageNet | 100        | 60,000     | https://github.com/twitter-research/meta-learning-lstm/tree/master/data/miniImagenet |
| ImageNet-1K   | 1000       | 14,197,122 | https://image-net.org/download.php                           |



#### Downloading Datasets

You can download dataset to the data directory from the link above or from paddle link. We here take `CIFAR-100`as an example.

```python
cd utils/datasets
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```



### Examples

Take Learngene Pool as an example. Note that you need to modify the `output_dir` and `data_path` in the `.yaml`.



#### Train and extract learngene

If we want to build the learngene pool with 18 instances, execute the following commands.

The learnegene extracted from the following line of code is deit_base9_patch16_224:

```python
cd learngene_methods/learngene_pool
python distill.py --config configs/conf_aux_base9.yaml
```

The default parameters of the experiment are shown in `configs/conf_aux_base9.yaml`. 

The learnegene extracted from the following line of code is deit_tiny9_patch16_224:

```python
cd learngene_methods/learngene_pool
python distill.py --config configs/conf_aux_tiny9.yaml
```

The default parameters of the experiment are shown in `configs/conf_aux_tiny9.yaml`. 



#### Build the learngene pool

In this section, we can construct the learngene pool from the extracted learngenes.

```python
cd learngene_methods/learngene_pool
python main.py --config configs/conf_build.yaml
```

The default parameters of the experiment are shown in `configs/conf_build.yaml`.



#### Initialize with learngene and test

In this section, we use learngene to initialize the descendant network and test the performence.

```python
cd learngene_methods/learngene_pool
python main.py --config configs/conf_ini.yaml
```

The default parameters of the experiment are shown in `configs/conf_ini.yaml`.

To build learngene pool and descendant models of different sizes, you only need to modify some hyper-parameters.



## License

This project is released under the [MIT license](https://github.com/Learngene-YuanMou/YuanMou/blob/master/LICENSE).



## Citations

If you use this toolbox in your research, please cite these papers.

<a name="HeurLearngene"></a>

```bibtex
@inproceedings{wang2021learngene,
   title={Learngene: From Open-World to Your Learning Task}, 
    author={Wang, Qiufeng and Geng, Xin and Lin, Shuxia and Xia, Shiyu and Qi, Lei and Xu, Ning},
   booktitle={AAAI}
   year={2022}
}
```

<a name="AutoLearngene"></a>

```bibtex
@misc{wang2023learngene,
      title={Learngene: Inheriting Condensed Knowledge from the Ancestry Model to Descendant Models}, 
     author={Qiufeng Wang and Xu Yang and Shuxia Lin and Jing Wang and Xin Geng},
     year={2023},
     eprint={2305.02279},
     archivePrefix={arXiv},
     primaryClass={cs.LG}
}
```

<a name="LearngenePool"></a>

```bibtex
@inproceedings{shi2024learngenepool,
  title={Building Variable-sized Models via Learngene Pool},
  author={Shi, Boyu and Xia, Shiyu and Yang, Xu and Chen, Haokun and Kou, Zhiqiang and Geng, Xin},
  booktitle={AAAI},
  year={2024}
}
```

<a name="TLEG"></a>

``` bibtex
@inproceedings{xia2024tleg,
  title={Transformer as Linear Expansion of Learngene},
  author={Xia, Shiyu and Zhang, Miaosen and Xu, Yang and Chen, Ruiming and Chen, Haokun and Xin, Geng},
  booktitle={AAAI},
  year={2024}
}
```



## Contacts

If you have any questions about our work, please do not hesitate to contact us by emails.

Wenxuan Zhu: zhuwx@seu.edu.cn

Yuankun Zu: zyk0418@seu.edu.cn



## Acknowledgements

Our project references the codes in the following repos.

+ [Heur Learngene](https://github.com/BruceQFWang/learngene)
+ [Deit](https://github.com/facebookresearch/deit)
