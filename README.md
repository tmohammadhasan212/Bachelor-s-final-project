# ❤️ Heart Attack Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-purple?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Heart%20Attack-red?style=for-the-badge)

A machine learning project for **early prediction of heart attack risk** using classification algorithms.

This project analyzes a heart attack dataset, performs exploratory data analysis, preprocesses the data, trains multiple machine learning models, and compares their performance.

---

## 📌 Project Overview

The main goal of this project is to predict whether a person has a higher or lower chance of a heart attack based on medical and health-related features.

The project is implemented in a **Jupyter Notebook** and uses several machine learning classification models, including Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Machine, Naive Bayes, Decision Tree, and a simple Artificial Neural Network.

---

## ✨ Features

- 📊 Exploratory Data Analysis  
- 🧹 Data preprocessing  
- 📈 Data visualization  
- ⚖️ Feature scaling  
- 🤖 Multiple classification models  
- 🧠 Artificial Neural Network model  
- 📉 Confusion matrix evaluation  
- ✅ Accuracy comparison between models  
- 📁 Local CSV dataset included  

---

## 🧱 Project Structure

```text
Bachelor-s-final-project/
├── README.md
├── classification-on-heart-attack-dataset.ipynb
└── heart.csv
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Main programming language |
| 📓 Jupyter Notebook | Development environment |
| 📊 Pandas | Data manipulation |
| 🔢 NumPy | Numerical operations |
| 📈 Matplotlib | Data visualization |
| 🌊 Seaborn | Statistical visualization |
| 🤖 Scikit-learn | Machine learning models |
| 🧠 TensorFlow / Keras | Neural network model |

---

## 📂 Dataset

The dataset used in this project is `heart.csv`.

It contains **303 records** and **14 columns** related to heart health and medical measurements.

### Dataset Columns

| Column | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender of the patient |
| `cp` | Chest pain type |
| `trtbps` | Resting blood pressure |
| `chol` | Cholesterol level |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalachh` | Maximum heart rate achieved |
| `exng` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slp` | Slope of the peak exercise ST segment |
| `caa` | Number of major vessels |
| `thall` | Thalassemia stress test result |
| `output` | Target variable |

### Target Variable

```text
0 = Lower chance of heart attack
1 = Higher chance of heart attack
```

---

## 🤖 Machine Learning Models

The following models are used in the notebook:

- Logistic Regression
- K-Nearest Neighbors Classifier
- Random Forest Classifier
- Support Vector Machine
- Naive Bayes
- Decision Tree Classifier
- Artificial Neural Network

---

## 📊 Model Performance

The notebook reports the following accuracy results for the classical machine learning models:

| Model | Accuracy |
|---|---:|
| 🟢 Naive Bayes | 96.77% |
| 🌲 Random Forest Classifier | 93.55% |
| 📈 Logistic Regression | 90.32% |
| 🌳 Decision Tree Classifier | 87.10% |
| 👥 K-Nearest Neighbors | 83.87% |
| ⚙️ Support Vector Machine | 83.87% |

The Artificial Neural Network section also shows a training accuracy of approximately:

```text
90.44%
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/tmohammadhasan212/Bachelor-s-final-project.git
cd Bachelor-s-final-project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment.

### 🪟 Windows

```bash
.venv\Scripts\activate
```

### 🍎 macOS / Linux

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn notebook
```

For the neural network section, also install:

```bash
pip install tensorflow keras
```

Optional package for ANN visualization:

```bash
pip install ann-visualizer
```

---

## ▶️ How to Run

Open the notebook:

```bash
jupyter notebook classification-on-heart-attack-dataset.ipynb
```

Then run the notebook cells step by step.

---

## 📌 Important Note About Dataset Path

The notebook was originally written with a Kaggle-style dataset path:

```python
heart = pd.read_csv("../input/heart-attack-uci/heart.csv")
```

Since this repository already contains `heart.csv`, you may need to change that line to:

```python
heart = pd.read_csv("heart.csv")
```

This allows the notebook to run locally from the repository folder.

---

## 🔍 Project Workflow

```text
1. Import required libraries
2. Load the heart attack dataset
3. Explore the dataset
4. Check dataset information and statistics
5. Visualize feature distributions
6. Analyze feature relationships
7. Scale the input features
8. Split data into training and testing sets
9. Train multiple classification models
10. Evaluate models using accuracy and confusion matrix
11. Compare model performance
12. Build and train a neural network model
```

---

## 📈 Visualizations

The notebook includes several visual analysis steps, such as:

- Histogram plots
- Count plots
- Pair plots
- Correlation heatmap
- Scatter plots
- Model accuracy comparison plot

These visualizations help understand the relationship between medical features and heart attack risk.

---

## 🧪 Evaluation Metrics

The project uses common classification evaluation methods:

- Accuracy Score
- Confusion Matrix
- Classification Report

These metrics help compare the performance of different machine learning models.

---

## 🏆 Best Performing Model

Based on the accuracy results shown in the notebook, **Naive Bayes** achieved the highest accuracy among the classical machine learning models.

```text
Naive Bayes Accuracy: 96.77%
```

---

## ⚠️ Disclaimer

This project is created for **academic and educational purposes only**.

It should not be used as a real medical diagnosis system.  
Medical decisions should always be made by qualified healthcare professionals.

---

## 🌱 Future Improvements

Possible future improvements:

- 📦 Add a `requirements.txt` file
- 🧹 Clean and organize the notebook
- 🔁 Add cross-validation
- 🎯 Add hyperparameter tuning
- 📊 Add more evaluation metrics such as Precision, Recall, F1-score, and ROC-AUC
- 💾 Save trained models using Pickle or Joblib
- 🌐 Build a simple web app for prediction
- 🧪 Test the model on more medical datasets
- 🏥 Improve medical interpretation of features

---

## 👨‍💻 Author

Created by [tmohammadhasan212](https://github.com/tmohammadhasan212).

---

## 💙 Final Note

This project demonstrates how machine learning can be used to analyze medical data and support early risk prediction tasks.
