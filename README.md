TransGAN for 3D Point Cloud Generation
This repository contains the code and findings for the research project on adapting the Transformer-based Generative Adversarial Network (TransGAN) for 3D point cloud synthesis. This work explores the feasibility of leveraging a pure Transformer architecture, originally designed for 2D images, to model the complex spatial dependencies inherent in 3D data.

This project was developed by Aditya Kumar, Vatsalraj Rathod, and Vedanshi Raiyani at the Indian Institute of Technology, Gandhinagar.

Table of Contents
Project Overview

Architecture

Dataset

Results and Analysis

Getting Started

Future Work

References

Project Overview
Point clouds are a fundamental data structure for representing 3D objects and environments. While deep learning models like PointNet and TreeGAN have advanced point cloud processing, Transformer models offer unique advantages in capturing global context and long-range dependencies.

This project investigates the adaptation of TransGAN for generating 3D point clouds. We engineered a novel generation pipeline by modifying the TransGAN framework to handle sparse, unordered 3D coordinate data. The primary goal was to generate high-quality, realistic 3D shapes by training the model on the ModelNet10 dataset.

Architecture
The model is a pure Transformer-based GAN, consisting of two main components:

Generator: A Transformer-based network that takes a random latent vector as input and progressively upsamples it to generate a 3D point cloud of shape (N, 3), where N is the number of points.

Critic (Discriminator): A Transformer-based network that takes a 3D point cloud (either real or generated) as input and outputs a scalar value indicating its realism.

The training follows the WGAN-GP (Wasserstein GAN with Gradient Penalty) loss framework to ensure stable training.

Dataset
The model was trained on the ModelNet10 dataset, a popular benchmark containing clean 3D CAD models from 10 object categories. The models are converted into point clouds, which serve as the "real" data for training the critic.

Results and Analysis
The full generation pipeline was successfully deployed and trained. The results are preliminary and serve as a proof-of-concept for using Transformers in this domain.

As seen in the qualitative results from our research, the generated point clouds capture the general spatial distribution of the training data but do not yet resolve into clear, well-defined object shapes. This is largely attributed to limitations in training time and dataset size.

A comparison between the expected high-fidelity point cloud shapes and the preliminary results achieved by the model.

Getting Started
Prerequisites

Python 3.8+

PyTorch

NumPy

Matplotlib (for visualization)

Installation

Clone the repository:

git clone [https://github.com/your-username/your-repo-link.git](https://github.com/your-username/your-repo-link.git)
cd your-repo-link

Install the required packages:

pip install -r requirements.txt

Usage

To train the model, run the training script:

python train.py --dataset_path /path/to/modelnet10

To generate new point clouds using a pre-trained model:

python generate.py --checkpoint /path/to/model.pth

Future Work
A comprehensive optimization strategy has been outlined to improve the fidelity of the generated shapes:

Extended Training: Run the model for significantly more epochs to allow for better convergence.

Hyperparameter Tuning: Conduct a systematic search for optimal parameter values (e.g., learning rate, batch size, network depth).

Larger Dataset: Train the model on a larger, more diverse dataset like ShapeNet to improve generalization.

Code Optimization: Profile and address any bugs or inefficiencies in the current implementation.

Quantitative Analysis: Implement metrics like Chamfer Distance (CD) and Fréchet Point Cloud Distance (FPD) for rigorous performance evaluation.

References
[1] VITA Group. (n.d.). TransGAN, GitHub repository.

[2] 3D Point Cloud Generative Adversarial Network. (n.d.). Papers with Code.

[3] Cui, R. (2021). Partial2Complete: Transformer Model Implementation, GitHub repository.

[4] Xie, Y., et al. (2020). PointStyleGAN: A Generative Adversarial Network for 3D Point Clouds.

[5] Sato, H., Takahashi, M. (2023). 3D Point Cloud Generation with Transformers. arXiv:2303.16450.
