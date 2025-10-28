# 🧠 HypothyroidismPrediction_AI_ML_Model  

> **Predicts hypothyroidism using machine learning algorithms.**  
> This repository contains the code and resources for an AI/ML model designed to predict hypothyroidism based on clinical and demographic features.  

---

## 🧾 Abstract  

Hypothyroidism is a prevalent endocrine disorder caused by an underactive thyroid gland, leading to insufficient production of thyroid hormones. This project presents a machine learning-based approach to predict hypothyroidism using demographic and biochemical data.  
The proposed model leverages algorithms such as **Logistic Regression**, **Support Vector Machine (SVM)**, and **Random Forest** to achieve efficient and interpretable classification. The system aims to support healthcare professionals by enabling **early diagnosis**, **risk prediction**, and **data-driven clinical decision-making**.  

---

## 🩺 Introduction  

**Hypothyroidism**, or **underactive thyroid**, occurs when the thyroid gland fails to produce enough hormones to regulate metabolism and energy utilization.  
Symptoms often include fatigue, weight gain, cold intolerance, and depression. If undiagnosed, it can lead to serious health issues such as infertility, heart disease, and myxedema.  

Early detection and intervention are therefore crucial. This project develops a **predictive machine learning model** capable of classifying patients as hypothyroid or normal using structured clinical data.  

The goal is to:
- Automate the diagnostic process using **data-driven AI techniques**
- Enhance **accuracy** and **speed** of thyroid disorder detection
- Support clinicians with interpretable, reproducible decision models  

---

## 📊 Dataset  

The dataset used in this project includes clinical and demographic variables relevant to thyroid function evaluation.  

### **Dataset Description**  
This dataset is curated for **binary classification** of hypothyroidism (presence or absence).  
It includes key biochemical markers and demographic indicators known to influence thyroid health.  

### **Features Overview**

| Feature | Type | Description |
|----------|------|-------------|
| **Age** | Numerical | Patient’s age (in years). Certain age groups have higher hypothyroidism risk. |
| **Sex** | Categorical | Patient’s sex (Male/Female). Females are more prone to thyroid disorders. |
| **TSH (Thyroid Stimulating Hormone)** | Numerical | Concentration of TSH (mIU/L). Elevated levels typically indicate hypothyroidism. |
| **T3 (Triiodothyronine)** | Numerical | Concentration of T3 (ng/dL or pmol/L). Low levels suggest hypothyroidism. |
| **TT4 (Total Thyroxine)** | Numerical | Concentration of TT4 (μg/dL or nmol/L). Low TT4 is a major diagnostic marker. |
| **T3 Measured** | Binary | Indicates if the T3 test was performed (Yes/No). Missing tests can be informative. |
| **Target Variable (Hypothyroid)** | Binary | 1 = Hypothyroid, 0 = Normal (used for classification). |

**Objective:**  
To build a supervised learning model that accurately predicts hypothyroidism from patient data using biochemical and demographic features.  

---

## 🧪 Methodology  

1. **Data Preprocessing**  
   - Handled missing values and outliers.  
   - Encoded categorical variables (e.g., `Sex`, `T3 Measured`).  
   - Normalized numerical features to improve model convergence.  

2. **Feature Selection**  
   - Used correlation matrices and feature importance from Random Forest to identify key predictors.  

3. **Model Training**  
   - Implemented multiple models:  
     - Logistic Regression  
     - Support Vector Machine (SVM)  
     - Random Forest Classifier  
   - Split data into training (80%) and testing (20%) sets.  

4. **Evaluation**  
   - Compared models based on accuracy, precision, recall, and F1-score.  
   - Used confusion matrices and ROC-AUC curves to visualize performance.  

---

## 📈 Performance Metrics  

Performance was evaluated using standard classification metrics to ensure reliability and robustness.

| Metric | Description |
|---------|-------------|
| **Accuracy** | Overall correctness of predictions. |
| **Precision** | Fraction of correctly predicted positives among all positive predictions. |
| **Recall (Sensitivity)** | Fraction of true hypothyroid cases correctly identified. |
| **F1-Score** | Harmonic mean of precision and recall, balancing both. |

These metrics are crucial for medical diagnosis applications, where **false negatives** (missed cases) can have severe consequences.  

---

## 🤖 Model Development  

The following machine learning algorithms were implemented and compared:
- **Logistic Regression:** For interpretable and baseline classification.  
- **Support Vector Machine (SVM):** For high-dimensional separation with kernel optimization.  
- **Random Forest:** For robust ensemble learning and feature importance analysis.  
Hyperparameter tuning was performed using **GridSearchCV** to optimize performance and generalization.  

---
## Libraries Used
-Python  – Programming language
-pandas – Data manipulation and analysis
-numpy – Numerical computations
-scikit-learn – Machine learning modeling and evaluation
-matplotlib – Visualization of results
-seaborn – Statistical and correlation-based visualization
------

## 📊 Results and Discussion  

- **Random Forest** achieved the highest overall performance, balancing accuracy and interpretability.  
- **SVM** performed competitively on precision but required more computation.  
- **Logistic Regression** provided strong baseline results and clear interpretability.  
Visualization techniques such as **confusion matrices** and **ROC curves** highlighted model reliability and trade-offs between sensitivity and specificity.  
The findings demonstrate that ensemble models like Random Forest can effectively predict hypothyroidism when trained on structured medical datasets.  

---
##🧭 Conclusion

This research-oriented project demonstrates how machine learning techniques can assist in the early detection of hypothyroidism.
By combining clinical data with AI-driven methods, the system enhances diagnostic accuracy and reduces manual intervention.
Key Outcomes:

-Developed and compared multiple ML models for hypothyroidism prediction.

-Achieved reliable accuracy and recall using Random Forest Classifier.

-Provided a framework for future integration into medical decision support systems.

-----

## 🧩 Dependencies  

Ensure the following dependencies are installed before running the project:  

