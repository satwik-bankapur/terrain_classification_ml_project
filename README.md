# Terrain Classification using Machine Learning

This repository contains a machine learning project for classifying terrain types using proprioceptive data from a robotic platform. The primary goal is to implement and evaluate different models for accurate terrain identification, which is crucial for autonomous robot navigation.

The project focuses on the [BorealTC proprioceptive dataset](https://github.com/norlab-ulaval/BorealTC) and compares the performance of a 1D Convolutional Neural Network (CNN) against a Support Vector Machine (SVM) baseline.

## Project Overview

The core task is to classify terrain into one of four categories: `hard`, `unstructured`, `soft`, and `indoor`. The approach involves:

1.  **Data Preparation**: Loading and preprocessing the BorealTC dataset. This includes segmenting the time-series IMU data (`wx`, `wy`, `wz`, `ax`, `ay`, `az`) into fixed-length windows and normalizing them.
2.  **Model Implementation**:
    *   A **1D Convolutional Neural Network (CNN)** based on the architecture described by Choi & Thomasson.
    *   A **Support Vector Machine (SVM)** to serve as a baseline for comparison.
3.  **Training and Evaluation**: Training both models on the processed IMU data and evaluating their performance using accuracy, confusion matrices, and classification reports.

## Key Results

The models were trained and evaluated on the BorealTC IMU data. The final test accuracies highlight the effectiveness of the deep learning approach for this sequential data task.

| Model        | Test Accuracy |
| :----------- | :------------ |
| **1D CNN**   | **0.8911**    |
| SVM Baseline | 0.7037        |

The 1D CNN model significantly outperformed the SVM baseline, demonstrating its superior ability to capture temporal patterns and relevant features from the sensor data for terrain classification.

**1D CNN Classification Report:**
```
              precision    recall  f1-score   support

        hard       0.88      0.95      0.91       878
unstructured       0.85      0.81      0.83       647
        soft       0.94      0.77      0.84       442
      indoor       0.92      0.98      0.95       642

    accuracy                           0.89      2609
   macro avg       0.90      0.88      0.88      2609
weighted avg       0.89      0.89      0.89      2609
```

## Challenges

*   **Dataset Differences**: The initial plan involved using both the SAIL-R and BorealTC datasets. However, significant differences in sensor types, features, and data structure made creating a unified processing pipeline challenging.
*   **Missing Values**: The BorealTC dataset has distinct patterns of missing values across different sensor modalities (e.g., Velocity/Current vs. IMU), requiring separate handling of these feature sets.
*   **SAIL-R Dataset Size**: The SAIL-R dataset was relatively small after segmentation, making it difficult to train a complex model like a CNN without significant overfitting. This led to the decision to focus exclusively on the larger BorealTC dataset.

## Future Work

*   **Explore Other Models**: Implement and evaluate other models like LSTMs or Transformers to potentially improve classification accuracy.
*   **Multi-modal Fusion**: Investigate techniques for combining information from different BorealTC modalities (e.g., IMU and Velocity/Current data) to leverage the strengths of each sensor type.
*   **Hyperparameter Tuning**: Systematically tune the hyperparameters of the 1D CNN to further optimize performance.
*   **Cross-validation**: Implement cross-validation to obtain more robust performance estimates and reduce dependency on a single train/test split.
*   **Real-time Implementation**: Analyze the computational requirements for deploying the trained model on a robot for real-time terrain classification.

## How to Use

The entire project is contained within the `Terrain_Classification.ipynb` Jupyter notebook. You can run it in an environment like Google Colab or a local Jupyter instance.

1.  **Clone the Repository (Optional):**
    ```sh
    git clone https://github.com/satwik-bankapur/terrain_classification_ml_project.git
    cd terrain_classification_ml_project
    ```

2.  **Open and Run the Notebook:**
    *   Open `Terrain_Classification.ipynb`.
    *   The notebook will automatically clone the required BorealTC dataset repository:
        ```python
        !git clone https://github.com/norlab-ulaval/BorealTC.git
        ```
    *   Execute the cells in order to perform data loading, preprocessing, model training, and evaluation.

### Dependencies

The main libraries used in this project are:
*   `tensorflow`
*   `scikit-learn`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
