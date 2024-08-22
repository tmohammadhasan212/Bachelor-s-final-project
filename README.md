# Bachelor-s-final-project
Machine learning project on a Heart dataset implemented for my bachelor's final project


# Optimum Classification Model for Heart Attack Dataset

## Project Overview
This project aims to identify the most effective classification model for predicting heart attack occurrences based on a given dataset. Through the application of various classification algorithms and rigorous hyperparameter tuning using grid search, the project seeks to optimize predictive accuracy, thereby contributing to improved medical diagnosis and patient care.

## Dataset Description
The dataset used in this project includes various medical attributes related to heart attack occurrences. Below is a summary of the features:

- **age**: Age of the individual (years).
- **sex**: Gender (0: female, 1: male).
- **cp**: Type of chest pain experienced.
- **trestbps**: Resting blood pressure (mm Hg).
- **chol**: Serum cholesterol level (mg/dl).
- **fbs**: Fasting blood sugar (> 120 mg/dl, 0: false, 1: true).
- **restecg**: Resting electrocardiographic results.
- **thalach**: Maximum heart rate during exercise.
- **exang**: Exercise-induced angina (0: no, 1: yes).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: Slope of the peak exercise ST segment.
- **ca**: Number of major vessels colored by fluoroscopy.
- **thal**: Thallium stress test results.
- **target**: Heart attack occurrence (0: no, 1: yes).

## Methodology
The methodology followed in this project is structured as follows:

1. **Data Preprocessing**:
   - Clean the dataset by handling missing values, outliers, and irrelevant features.
   - Split the dataset into training and testing subsets to evaluate model performance.

2. **Classification Algorithms**:
   - Implement various classification algorithms, including Decision Trees, Random Forest, Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and AdaBoost.

3. **Hyperparameter Tuning**:
   - Utilize grid search to explore and identify the best combination of hyperparameters for each classification model.

4. **Model Evaluation**:
   - Assess model performance using metrics such as accuracy, precision, recall, and F1-score.
   - Apply cross-validation techniques to ensure model robustness.

5. **Comparison and Selection**:
   - Compare the performance of different models to identify the optimal classification model for the Heart Attack dataset.

## Expected Outcome
The expected outcome of this project is to determine the classification model with the highest predictive accuracy. This model will offer reliable predictions for heart attack occurrences, providing valuable insights for medical professionals in diagnosing and treating patients.

## How to Use
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/heart-attack-classification.git
   cd heart-attack-classification

