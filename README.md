# 🧠 Handling Mixed Variables in Machine Learning

This project demonstrates how to preprocess and build pipelines for datasets containing **mixed variables** (numerical, categorical, datetime, boolean, etc.).

---

## 📌 What are Mixed Variables?

Most real-world datasets contain multiple data types together:
- 🔢 **Numerical** → Age, Salary, Height  
- 🔤 **Categorical** → Gender, City, Education  
- 📅 **Datetime** → Joining Date, Transaction Time  
- ✅ **Boolean** → Yes/No, True/False  

Machine learning models cannot directly process these raw variables.  
👉 That’s why we use **preprocessing + pipelines**.

---

## 🚀 Workflow

1. **Data Loading**  
2. **Identify variable types** (numeric, categorical, datetime, boolean)  
3. **Preprocessing**  
   - Imputation for missing values  
   - Scaling numerical features  
   - Encoding categorical features  
   - Feature extraction from datetime  
4. **Pipeline Creation**  
5. **Model Training & Evaluation**  

---

## ⚙️ Example Code (Scikit-learn Pipeline)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
df = pd.DataFrame({
    "age": [25, 30, 22, None],
    "salary": [50000, 60000, None, 45000],
    "gender": ["Male", "Female", "Female", None],
    "city": ["Delhi", "Mumbai", "Delhi", "Chennai"],
    "purchased": [1, 0, 1, 0]
})

X = df.drop("purchased", axis=1)
y = df["purchased"]

# Define feature types
numeric_features = ["age", "salary"]
categorical_features = ["gender", "city"]
