# 🧠 Inductive Gradient Adjustment (IGA)  
*A theory-grounded gradient adjustment method for improving spectral bias in implicit neural representations (INRs).*

## 🔍 Overview

This repository “IGA-INR” contains the official implementation of **IGA**, as proposed in our paper: 
> **Inductive Gradient Adjustment for Spectral Bias in Implicit Neural Representations**<br>
> *Kexuan Shi, Hai Chen, Leheng Zhang, Shuhang Gu*   
> Accepted to **ICML 2025**·  [arXiv](https://arxiv.org/abs/2410.13271). [Openreview](https://openreview.net/forum?id=pYMZQtkp3F).

IGA is a practical and theory-grounded method that improves the training dynamics of Implicit Neural Representations (INRs) by adjusting gradients based on inductive generalization of sampled gradient transformation matrix derived from empirical Neural Tangent Kernel (eNTK) matrix. Compared to vanilla training methods, our IGA encourages better learning of high-frequency signals such as texture and edges without modifying model architectures. This repository provides everything needed to reproduce the experiments from the paper.

---
## ✨ Highlights

- 📈 **Consistent improvements** across diverse INR architectures and tasks.
- 🔬 **Theoretical and empirical justification** of eNTK-based and inductively generalized transformations.
- 🧩 **Plug-in training strategy**, applicable to existing INR pipelines with minimal changes.
- 🖥️ **Runs efficiently** on a single 24GB GPU (e.g., RTX 3090), with reasonable memory and compute overhead.

---
## 🧰 Environment
```bash
cd IGA-INR
conda create --name iga-inr --file requirements.txt

```
---
## 🚀 Getting Started
### Simple Function Approximation
To produce the experiment 1 of Section 4 of our paper, please:
```bash
cd simple_function_approximation/experiment1
sh train_1024.sh
sh train_2048.sh
sh train_4096.sh
sh train_8192.sh
```

To produce the experiment 2 of Section 4 of our paper, please:
```bash
cd simple_function_approximation/experiment2
sh train_relu.sh
sh train_siren.sh
```
