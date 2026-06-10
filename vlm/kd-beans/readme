# Bean Plant Disease Classification via Knowledge Distillation

## Overview

This project demonstrates Knowledge Distillation (KD) for image classification using the Beans Dataset. A large Vision Transformer (ViT) model acts as the teacher, while a lightweight MobileNetV2 model acts as the student. The goal is to transfer knowledge from the teacher to the student, producing a smaller model suitable for deployment on resource-constrained devices while maintaining good classification performance.

---

## Dataset

The project uses the Beans Dataset from Hugging Face.

Dataset Link:
https://huggingface.co/datasets/AI-Lab-Makerere/beans

### Classes

* Angular Leaf Spot
* Bean Rust
* Healthy

### Dataset Split

| Split      | Images |
| ---------- | ------ |
| Train      | 1034   |
| Validation | 133    |
| Test       | 128    |
| Total      | 1295   |

The train, validation, and test splits are kept completely separate to avoid data leakage.

---

## Model Architecture

### Teacher Model

Model:
merve/beans-vit-224

Type:
Vision Transformer (ViT)

Description:
A pre-trained and fine-tuned Vision Transformer used to generate soft targets for knowledge distillation.

### Student Model

Model:
MobileNetV2

Type:
Lightweight Convolutional Neural Network

Description:
A compact model initialized with random weights and trained using knowledge distillation.

---

## Knowledge Distillation

The student model learns from:

1. Ground-truth labels (hard labels)
2. Teacher model predictions (soft labels)

### Loss Function

Total Loss = (1 − λ) × Cross Entropy Loss + λ × KL Divergence Loss

#### Cross Entropy Loss

Measures how well the student predicts the correct class labels.

#### KL Divergence Loss

Measures how closely the student's probability distribution matches the teacher's probability distribution.

### Distillation Parameters

| Parameter   | Value |
| ----------- | ----- |
| Temperature | 5     |
| Lambda      | 0.5   |
| Epochs      | 10    |

---

## Evaluation Metric

Accuracy

Accuracy is computed on the test split consisting of 128 unseen images.

---

## Results

| Model                           | Test Accuracy |
| ------------------------------- | ------------- |
| Teacher (ViT-base)              | 93.75%        |
| Student Before KD (MobileNetV2) | 33.59%        |
| Student After KD (MobileNetV2)  | 66.41%        |

### Observation

Knowledge Distillation significantly improves the student model's performance. The MobileNetV2 model increases from 33.59% accuracy before distillation to 66.41% accuracy after learning from the Vision Transformer teacher.


