# Terrain Classification ML Mini-Project

Machine Learning project for automated terrain classification using image data.

## Project Information

**Course:** UE23CS352A Machine Learning  
**Due Date:** October 13, 2025  
**Team Members:** [Satwik R Bankapur, Sathwik H S]

## Overview

This project implements terrain classification using both traditional machine learning and deep learning approaches. The goal is to classify terrain images into 5 distinct categories: desert, forest, mountain, grassland, and rocky terrain.

## Dataset

- **Source:** Kaggle Terrain Dataset
- **Total Images:** 1,000
- **Image Size:** 224x224x3 pixels
- **Number of Classes:** 5
- **Classes:** Desert, Forest, Mountain, Grassland, Rocky

## Methods

Three different approaches were implemented and compared:

1. **Support Vector Machine (SVM)** - Traditional ML with RBF kernel
2. **Random Forest** - Ensemble method with 100 trees
3. **Convolutional Neural Network (CNN)** - Deep learning with MobileNetV2 transfer learning

## Results

| Model | Accuracy |
|-------|----------|
| SVM | 100.0% |
| Random Forest | 100.0% |
| CNN (MobileNetV2) | 97.0% |

## Implementation

### Environment
- Google Colab
- Python 3.x
- TensorFlow/Keras for deep learning
- Scikit-learn for traditional ML

### Features
- Color features (RGB and HSV statistics)
- Texture features (gradient-based)
- Total of 15 engineered features

### CNN Architecture
- Base: MobileNetV2 (pre-trained on ImageNet)
- Custom classification head with dropout
- Data augmentation pipeline

## Files

terrain-classification-ml-project/
├── ML_terrain_classification.ipynb (Main notebook)
├── README.md
├── results/
│ ├── terrain_classification_results.csv
│ └── detailed_predictions.csv
└── docs/
└── project_report.pdf


## How to Run

1. Open `terrain_classification.ipynb` in Google Colab
2. Upload your Kaggle API key (kaggle.json)
3. Run all cells sequentially
4. Results will be saved to CSV files

## Installation

Required libraries:
pip install tensorflow scikit-learn opencv-python matplotlib seaborn pandas numpy


## Key Findings

- Traditional ML models achieved perfect accuracy due to effective feature engineering
- CNN with transfer learning achieved 97% accuracy
- All models demonstrated excellent generalization on test data
- Feature extraction from images is crucial for traditional ML performance

## Technical Stack

- **Deep Learning:** TensorFlow, Keras
- **Machine Learning:** Scikit-learn
- **Image Processing:** OpenCV
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy

## Authors

Satwik R Bankapur
Sathwik H S
Computer Science Students

## Acknowledgments

- Course instructors and TAs
- Kaggle for dataset hosting
- Google Colab for computational resources
