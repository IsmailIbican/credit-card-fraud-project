# Credit Card Fraud Detection Project

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The goal is to build a classifier that can distinguish between legitimate and fraudulent transactions with high accuracy, addressing the issue of imbalanced data.

## 📂 Dataset Information

**Note:** The dataset used in this project is not included in the repository due to its size (>100MB).

* **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* **Description:** The dataset contains transactions made by credit cards in September 2013 by European cardholders.
* **How to use:** Download the `creditcard.csv` file from the link above and place it in the `data/` folder.

## Technologies Used

* **Python 3.x**
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Imbalanced Data:** imbalanced-learn, SMOTE
* **Machine Learning:** scikit-learn, xgboost
* **Explanation of Model:**shap
* **Real-Life Entegrated Dashboard and API:** fastapi, streamlit
* **Saving Models:** joblib

## 📊 Model Performance Results

Below is a comparison of the models trained in this project:

| Model Name           | Accuracy | Recall Score | F1-Score |
|----------------------|----------|--------------|----------|
| Logistic Regression  | 0.45     | 0.41         | 0.10      |
| Random Forest        | 0.97     | 0.70         | 0.86     |
| Gradient Boosting    | 0.53     | 0.50         | 0.25     | 
| XGBoost              | 0.96     | 0.78         | 0.84     |


## How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    ```
2.  Install required libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn joblib shap fastapi streamlit
    ```
3.  Run the Jupyter Notebook or Python script.

## 📝 License

This project is open-source.(Distributed under the MIT license. See 'LICENSE' for more information)