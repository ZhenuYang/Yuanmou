# YuanMou (元谋)

<p align="center">
  <img src="./sources/logo.jpg" alt="logo" width="60%"/>
</p>



## 项目介绍

YuanMou 是一个基于 PyTorch 的开源 **学习基因（learngene）** 工具箱。**学习基因** 是一种新颖的机器学习范式，其核心思想是首先将一个较大且经过良好训练的模型（称为 *祖先模型*）浓缩为一个微小但关键的部分，称为 *Learngene*，其中包含祖先模型的核心且可泛化的知识。随后，Learngene 可以扩展，用于初始化许多不同规模的模型，以适应不同的下游任务和资源限制，这些模型被称为 *后代模型*。
本项目将多种不同的学习基因算法统一到一个由配置文件驱动的命令行工具中，方便研究人员和工程师进行复现、对比和二次开发。

目前，YuanMou包括以下5种 **学习基因** 算法.

+ [Heur Learngene]( https://arxiv.org/abs/2106.06788): AAAI 2022
+ [Auto Learngene](https://arxiv.org/abs/2305.02279): arXiv prePrint 2023
+ [Learngene Pool](https://arxiv.org/abs/2312.05743): AAAI 2024
+ [TLEG](https://arxiv.org/abs/2312.05614): AAAI 2024
+ [SWS](https://arxiv.org/abs/2404.16897): IJCAI 2024

<p align="center">
  <img src="./sources/fig1.png" alt="image1" width="50%"/>
</p>


## 项目优势

我们分别在当前广为人知的大型语言模型 [**Llama2-7B**](https://ai.meta.com/llama/) 上应用了Heur Learngene 和Auto Learngene, 并以原始的 [LoRa](https://arxiv.org/abs/2106.09685) 作为基线, 展示了 **学习基因** 具有以下优势：

+ ### 更好的模型表现

  通过采用Heur Learngene 和Auto Learngene，在大型语言模型上进行相同数量的训练轮次微调，可以带来性能提升。

<p align="center">
  <img src="./sources/fig2.jpg" alt="image2" width="40%"/>
</p>




+ ### 更快的收敛速度

 通过采用Heur Learngene 和Auto Learngene，大型语言模型收敛所需的训练轮次分别减少了 **30%** 和 **40%** 。.

<p align="center">
  <img src="./sources/fig3.jpg" alt="image3" width="40%"/>
</p>




+ ### 更少的GPU时间

 通过使用 Heur Learngene 和 Auto Learngene，对大型语言模型进行微调所需的 GPU 时间分别减少了 30% 和 40%，从而降低了资源成本。

<p align="center">
  <img src="./sources/fig4.jpg" alt="image4" width="40%"/>
</p>




+ ### 更少的训练样本

  通过使用 Heur Learngene 和 Auto Learngene，对大型语言模型进行微调所需的训练数据量分别仅为 60% 和 50%，从而进一步降低了资源成本。

<p align="center">
  <img src="./sources/fig5.jpg" alt="image5" width="40%"/>
</p>



## 核心特性

- **:gear: 配置驱动**: 所有的实验参数，包括模型类型、数据集路径、超参数等，都通过 `.yaml` 配置文件进行管理，无需修改任何代码。
- **:package: 模块化设计**: 每种学习基因算法（例如 `heur_vgg` 或 `tleg_vit`）被封装在独立的模块中，代码结构清晰，互不干扰。
- **:rocket: 高度可扩展**: 框架设计允许轻松添加新的算法。只需在 `src/learngene/` 目录下创建一个新的模块，并编写相应的配置文件即可。
- **:computer: 统一的命令行接口**: 所有操作都通过唯一的命令行入口 `scripts/learngene` 完成，命令包括 `train`（训练祖先）, 和 `adapt`（适配后代）。

## 安装

1.  克隆本仓库:
    ```bash
    git clone https://github.com/your-username/learngene-toolkit.git
    cd Yuanmou
    ```

2.  (推荐) 创建并激活conda虚拟环境:
    ```bash
    conda create -n learngene python=3.8 -y
    conda activate learngene
    ```

3.  安装依赖:
    ```bash
    pip install -r requirements.txt
    ```

## 快速开始

所有操作都通过 `scripts/learngene` 命令行工具进行。首次使用前，请为其赋予执行权限：
```bash
chmod +x ./scripts/learngene
```
#### 示例1: 运行 heur_vgg 算法流程
这是一个基于VGG架构的传统学习基因算法。

1. 准备配置文件
打开 configs/ancestor_heur_vgg.yaml，修改 DATASET.PATH 为您的 "continualdataset" 路径。
打开 configs/descendant_heur_vgg.yaml，修改 DATASET.PATH 为您的 "inheritabledataset" 路径，并确保 GENE.ANCESTOR_MODEL_PATH 指向祖先模型的输出目录。

2. 训练祖先模型

```bash
./scripts/learngene train --config configs/ancestor_heur_vgg.yaml
```
训练完成后，模型将保存在 outputs/heur_vgg/ancestor_model (或您在配置中指定的其他位置)。

#### 示例2: 运行 tleg_vit 算法流程 

这是一个基于Vision Transformer和线性扩展思想的更现代、更强大的算法。

1. 准备配置文件
打开 configs/ancestor_tleg_vit.yaml，修改 DATASET.PATH 为您数据集 (如CIFAR-100) 的根路径。
打开 configs/descendant_tleg_vit.yaml，同样修改数据集路径，并确保 MODEL.LOAD_GENE 指向祖先模型训练后生成的 checkpoint.pth 文件。

2. 训练祖先模型 (即学习基因)

```bash
./scripts/learngene train --config configs/ancestor_tleg_vit.yaml
```

训练完成后，包含学习基因的模型 (checkpoint.pth) 将保存在 outputs/tleg_vit/ancestor_aux_tiny。
3. 适配不同深度的后代模型

您可以修改 configs/descendant_tleg_vit.yaml 中的 MODEL.TYPE 来生成不同大小的后代模型，例如从 deit_tiny_patch16_224_L3 (3层) 到 deit_tiny_patch16_224_L12 (12层)。

```bash
# 假设我们想生成一个6层的后代模型
# 确认配置文件中 MODEL.TYPE 为 deit_tiny_patch16_224_L6
./scripts/learngene adapt --config configs/descendant_tleg_vit.yaml
```
最终适配好的模型将保存在 outputs/tleg_vit/descendant_deit_tiny_L6。




## 引用

如果您希望在您的研究中引用我们的工具箱，请添加以下内容

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



## 联系

如果您在使用该项目时遇见无法解决的问题，请发邮件.

213233931@seu.edu.cn

