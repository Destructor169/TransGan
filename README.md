# TransGAN for 3D Point Cloud Generation

This repository contains the code and findings for the research project on adapting the Transformer-based Generative Adversarial Network (TransGAN) for 3D point cloud synthesis. This work explores the feasibility of leveraging a pure Transformer architecture, originally designed for 2D images, to model the complex spatial dependencies inherent in 3D data.

This project was developed by **Aditya Kumar, Vatsalraj Rathod, and Vedanshi Raiyani** at the *Indian Institute of Technology, Gandhinagar*.

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results and Analysis](#results-and-analysis)
- [Future Work](#future-work)
- [References](#references)

---

## 📌 Project Overview
Point clouds are a fundamental data structure for representing 3D objects and environments. While deep learning models like PointNet and TreeGAN have advanced point cloud processing, Transformer models offer unique advantages in capturing global context and long-range dependencies.

This project investigates the adaptation of TransGAN for generating 3D point clouds. We engineered a novel generation pipeline by modifying the TransGAN framework to handle sparse, unordered 3D coordinate data. The primary goal was to generate high-quality, realistic 3D shapes by training the model on the **ModelNet10** dataset.

---

## 🏗 Architecture
The model is a pure Transformer-based GAN, consisting of two main components:

- **Generator**: A Transformer-based network that takes a random latent vector as input and progressively upsamples it to generate a 3D point cloud of shape `(N, 3)`, where `N` is the number of points.
- **Critic (Discriminator)**: A Transformer-based network that takes a 3D point cloud (either real or generated) as input and outputs a scalar value indicating its realism.

The training follows the **WGAN-GP (Wasserstein GAN with Gradient Penalty)** loss framework to ensure stable training.

---

## 📂 Dataset
The model was trained on the **ModelNet10** dataset, a popular benchmark containing clean 3D CAD models from 10 object categories. The models are converted into point clouds, which serve as the "real" data for training the critic.

---

## 📊 Results and Analysis
The full generation pipeline was successfully deployed and trained. The results are **preliminary** and serve as a proof-of-concept for using Transformers in this domain.

- Generated point clouds capture the general spatial distribution of the training data.  
- However, they do not yet resolve into clear, well-defined object shapes.  
- Limitations include restricted training time and dataset size.

---
