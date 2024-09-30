# UniTTA: Unified Benchmark and Versatile Framework Towards Realistic Test-Time Adaptation

Authors: [Chaoqun Du](https://scholar.google.com/citations?user=0PSKJuYAAAAJ&hl=en),
[Yulin Wang](https://www.wyl.cool/),
[Jiayi Guo](https://www.jiayiguo.net/),
[Yizeng Han](https://yizenghan.top/),
Jie Zhou,
[Gao Huang](https://www.gaohuang.net).

[![arXiv](https://img.shields.io/badge/arxiv-UniTTA-blue)](https://arxiv.org/pdf/2407.20080)

## Introduction

<p align="center">
    <img src="figures/benchmark.png" width= "500" alt="fig1" />
</p>

The general idea of SimPro addressing the ReaLTSSL
problem. (a) Current methods typically rely on predefined or
assumed class distribution patterns for unlabeled data, limiting
their applicability. (b) In contrast, our SimPro embraces a more
realistic scenario by introducing a simple and elegant framework
that operates effectively without making any assumptions about
the distribution of unlabeled data. This paradigm shift allows for
greater flexibility and applicability in diverse ReaLTSSL scenarios.

<p align="center">
    <img src="figures/framework.png" alt="fig1" />
</p>

The SimPro Framework Overview. This framework distinctively separates the conditional and marginal (class) distributions.
In the E-step (top), pseudo-labels are generated using the current parameters $\theta$ and $\pi$.
In the subsequent M-step (bottom), these pseudo-labels, along with the ground-truth labels, are utilized to compute the Cross-Entropy loss, facilitating the optimization of network parameters $\theta$ via gradient descent.
Concurrently, the marginal distribution parameter $\pi$ is recalculated using a closed-form solution based on the generated pseudo-labels.

## Get Started

### Requirements

```[bash]
conda create -n unitta python=3.9 -y
conda activate unitta

# install pytorch 2.2.2 (default for cuda 11)
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# install required packages
pip install -r requirements.txt

# install robustbench
cd robustbench
pip install -v .
```

### Datasets

Download [CIFAR-10-C](https://zenodo.org/record/2535967#.ZDETTHZBxhF), [CIFAR-100-C](https://zenodo.org/record/3555552#.ZDES-XZBxhE) and [ImageNet-C](https://zenodo.org/record/2235448). The default data path is set to `/home/data/` and the data structure should be as follows:

```[bash]
/home/data/
├── CIFAR-10-C
├── CIFAR-100-C
├── ImageNet-C
```

### Test-Time Adaptation

By default, we use 1 RTX3090 GPU and the run.sh will run all the combinations of methods and test settings on CIFAR-10-C, CIFAR-100-C and ImageNet-C.

You can also run the code for specific methods and test settings by modifying the run.sh file.

```[bash]
bash sh/run.sh
```

We have optimized the code for unitta sampler, thus the results may not be exactly the same as the paper.

## Citation

If you find this code useful, please consider citing our paper:

```[tex]
@misc{du2024unitta,
      title={UniTTA: Unified Benchmark and Versatile Framework Towards Realistic Test-Time Adaptation},
      author={Chaoqun Du and Yulin Wang and Jiayi Guo and Yizeng Han and Jie Zhou and Gao Huang},
      year={2024},
      eprint={2407.20080},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## Contact

If you have any questions, please feel free to contact the authors. Chaoqun Du: <dcq20@mails.tsinghua.edu.cn>.

## Acknowledgement

Our code is based on the [TRIBE](https://github.com/Gorilla-Lab-SCUT/TRIBE) (Towards Real-World Test-Time Adaptation: Tri-Net Self-Training with Balanced Normalization).
