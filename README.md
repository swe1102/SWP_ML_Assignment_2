# Credit Card Default Prediction & Model Evaluation

## a. Problem Statement
This dataset consists of various factors which may lead to a credit card user to default on their payment. Using a model to predict if a card holder may/ may not default on the payments gives financial institutions room to deploy stricter rules and maintain appropriate card spending limits.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository ("Default of Credit Card Clients")
- **Instances**: 30,000 clients
- **Features**: 23 input variables including credit limit, demographic factors (gender, education, marital status, age), and historical monthly repayment statuses/payment tracking from April to September 2005.
- **Target**: `default payment next month` (Binary Classification: 1 = default, 0 = no default).

## c. GitHub Repository Link
https://github.com/swe1102/SWP_ML_Assignment_2

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8095 | 0.7180 | 0.7812 | 0.8095 | 0.7765 | 0.3541 |
| **Decision Tree** | 0.7245 | 0.6120 | 0.7380 | 0.7245 | 0.7305 | 0.2210 |
| **KNN** | 0.7930 | 0.7015 | 0.7620 | 0.7930 | 0.7710 | 0.3120 |
| **Naive Bayes** | 0.7750 | 0.7100 | 0.7680 | 0.7750 | 0.7710 | 0.3200 |
| **Support Vector Machine** | 0.8175 | 0.7290 | 0.7950 | 0.8175 | 0.7910 | 0.3890 |

### Observations on Model Performance
- **Logistic Regression**: Solid baseline model with good accuracy (80.95%) and clean probabilistic decision boundaries.
- **Decision Tree**: Has high variance and slight overfitting on training features, resulting in lower accuracy (72.45%) and MCC.
- **KNN**: Moderate classification performance (79.30%), but computationally sensitive to high feature dimensionality.
- **Naive Bayes**: Shows strong precision across specific features, but strict independence assumptions slightly hamper overall recall.
- **Support Vector Machine (SVM)**: Outperforms other classifiers across key metrics, achieving the highest Accuracy (81.75%), AUC (0.7290), and MCC (0.3890).

**Overall Winner for Dataset**: **Support Vector Machine (SVM)**
