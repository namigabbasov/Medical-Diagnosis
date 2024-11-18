# Medical Diagnosis Projects Repository
![Medical Diagnosis Projects Banner](banner-no-text-final.png)

## Overview

This repository contains a suite of projects exploring the application of deep learning and machine learning techniques to assist in medical diagnosis. The aim is to leverage advanced computational tools to complement the work of medical professionals by providing additional insights for diagnosis. While these models cannot replace traditional diagnostic methods or specialists, they demonstrate the immense potential of artificial intelligence in aiding medical decision-making.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Descriptions](#project-descriptions)
   - [Pneumonia Diagnosis](#1-pneumonia-diagnosis)
   - [Diabetic Retinopathy](#2-diabetic-retinopathy)
   - [Breast Cancer](#3-breast-cancer)
   - [Chronic Kidney Disease](#4-chronic-kidney-disease)
   - [Tuberculosis](#5-tuberculosis)
   - [Diabetes](#6-diabetes)
3. [Key Models and Tools](#key-models-and-tools)
4. [Results and Insights](#results-and-insights)
5. [Acknowledgments](#acknowledgments)

---

## Introduction

This repository explores the intersection of deep learning, machine learning, and medical diagnostics. It includes projects that analyze diverse data modalities such as X-ray images, fundus images, mammograms, and tabular datasets. Each project demonstrates the utility of AI in different medical fields, including pneumonia detection, diabetic retinopathy classification, breast cancer analysis, chronic kidney disease prediction, and tuberculosis diagnosis. Additionally, we investigate how statistical methods (NHST) can be merged with predictive modeling for diabetes research.

---

## Project Descriptions

### 1. Pneumonia Diagnosis
- **Objective**: Classify chest X-rays into two categories (Normal, Pneumonia) or three categories (Normal, Viral Pneumonia, Bacterial Pneumonia).
- **Datasets**: Large-scale publicly available pneumonia X-ray datasets.
- **Models**: 
  - Convolutional Neural Networks (CNNs)
  - Vision Transformers (ViT) and Swin Transformers
  - Pre-trained models from Hugging Face
- **Outcomes**: Effective classification of X-ray images using deep learning models.

---

### 2. Diabetic Retinopathy
- **Objective**: Classify fundus images into 5 categories of diabetic retinopathy severity.
- **Datasets**: Fundus image datasets with annotations for severity levels.
- **Models**:
  - CNNs, ViT, Swin Transformers
  - U-Net for image segmentation
  - Tabular data analysis using conventional machine learning algorithms
- **Outcomes**: Achieved accurate classification and segmentation of diabetic retinopathy to support ophthalmologists.

---

### 3. Breast Cancer
- **Objective**: Analyze imbalanced mammograms, histology images, and tabular data for breast cancer detection and prediction.
- **Datasets**: 
  - Mammograms
  - Wisconsin Breast Cancer dataset
- **Models**:
  - CNNs for image analysis
  - Machine learning algorithms (e.g., Random Forest, XGBoost) for tabular data
- **Challenges**: Addressed class imbalance issues using resampling techniques and evaluation metrics such as F1-score.

---

### 4. Chronic Kidney Disease
- **Objective**: Predict chronic kidney disease outcomes using tabular data.
- **Datasets**: Chronic Kidney Disease datasets with clinical features.
- **Models**: Machine learning algorithms like Logistic Regression, Decision Trees, and Gradient Boosting.
- **Outcomes**: Developed interpretable models to predict disease presence.

---

### 5. Tuberculosis
- **Objective**: Diagnose tuberculosis using both X-ray images and tabular data.
- **Datasets**: Tuberculosis-specific datasets combining imaging and clinical features.
- **Models**:
  - CNNs and Transformers for X-ray image classification
  - Machine learning algorithms for tabular data
- **Outcomes**: High accuracy in identifying tuberculosis using hybrid approaches.

---

### 6. Diabetes
- **Objective**: Merge Null Hypothesis Statistical Testing (NHST) with predictive modeling to explore diabetes-related trends.
- **Datasets**: Diabetes tabular datasets.
- **Models**: Conventional machine learning algorithms (e.g., SVM, Random Forest, Logistic Regression).
- **Outcomes**: Combined statistical and predictive modeling insights to study diabetes-related outcomes.

---

## Key Models and Tools

1. **Deep Learning Models**:
   - CNNs
   - Vision Transformers (ViT)
   - Swin Transformers
   - U-Net (for segmentation tasks)

2. **Machine Learning Algorithms**:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Support Vector Machines (SVM)

3. **Tools and Libraries**:
   - TensorFlow/Keras
   - PyTorch
   - Scikit-learn
   - Hugging Face Transformers

## Results and Insights
- Deep learning models showed good performance in classifying medical images.
- Combining image-based models with tabular data provided improved diagnostics in some projects.
- Addressing class imbalance and using interpretability techniques proved crucial for real-world applicability.

## Acknowledgments
This repository uses publicly available datasets and pre-trained models to advance medical diagnosis research. Special thanks to the creators of these datasets and frameworks like TensorFlow, PyTorch, and Hugging Face for their contributions to open-source AI development.
