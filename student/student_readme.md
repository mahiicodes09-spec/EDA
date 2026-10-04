# Student GPA Prediction

## 1. Dataset Overview

* **Dataset:**g Student Performance Dataset
* **Rows:** 2,000
* **Columns:** 7
* **Target Variable:** `GPA`
* **Problem Type:** Regression

## 2. Exploratory Data Analysis (EDA)

### Step 1: Data Loading and Inspection

Loaded the dataset using Pandas and performed initial inspection using:

* `df.head()` – Viewed the first five rows.
* `df.tail()` – Viewed the last five rows.
* `df.shape` – Checked dataset dimensions.
* `df.columns` – Examined column names.
* `df.info()` – Checked data types and non-null values.
* `df.describe()` – Generated statistical summaries.

### Step 2: Data Quality Checking

Checked the dataset for:

* Missing values using `isnull().sum()`.
* Duplicate records using `duplicated().sum()`.
* Data types and non-null counts using `info()`.
* Statistical ranges using `describe()`.

### Step 3: Univariate Analysis

Analyzed individual numerical columns to understand their distributions.

Used visualizations such as histograms to examine the distribution of study hours, sleep hours, activity hours, and GPA.

### Step 4: Bivariate Analysis

Explored relationships between independent variables and GPA using visualizations such as scatter plots.

Examined how daily study time, sleep, social activities, extracurricular activities, and physical activity relate to GPA.

### Step 5: Correlation Analysis

Calculated correlations between numerical features and visualized them using a correlation heatmap.

Examined positive and negative correlations between the input features and GPA, along with relationships among the independent variables.

### Step 6: Outlier Analysis

Examined numerical variables for extreme values using statistical summaries and visual inspection.

## 3. Feature Selection and Data Preparation

Selected the following five independent variables:

* `Study_Hours_Per_Day`
* `Extracurricular_Hours_Per_Day`
* `Sleep_Hours_Per_Day`
* `Social_Hours_Per_Day`
* `Physical_Activity_Hours_Per_Day`

Selected `GPA` as the target variable.

```python
X = df.iloc[:, 1:6]
y = df["GPA"]
```

## 4. Train-Test Split

Split the dataset into training and testing subsets using Scikit-learn.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

* Training data: 80%
* Testing data: 20%
* Random state: 42

## 5. Model Training

Used **Linear Regression** to predict GPA.

### Step 1: Importing the Model

```python
from sklearn.linear_model import LinearRegression
```

### Step 2: Model Initialization

```python
model = LinearRegression()
```

### Step 3: Model Fitting

```python
model.fit(X_train, y_train)
```

Trained the model using the selected independent variables and corresponding GPA values.

### Step 4: Prediction

```python
y_pred = model.predict(X_test)
```

Generated GPA predictions using the testing dataset.

## 6. Model Evaluation

Evaluated the model using the following regression metrics.

### Mean Absolute Error (MAE)

python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)


### Mean Squared Error (MSE)

`python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)


### R² Score

python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)


### Residual Analysis

Calculated residuals to examine the difference between actual and predicted GPA values.

python
residuals = y_test - y_pred


Used residual plots to inspect the prediction errors.

## 7. Saving the Trained Model

Saved the trained model using Joblib.

python
import joblib

joblib.dump(model, "student_gpa_model.pkl")

The model was saved as `student_gpa_model.pkl` for later use in the application.

## 8. Streamlit Application

Created an interactive application using Streamlit in `student_app.py`.

### Steps Performed

* Imported Streamlit, Pandas, and Joblib.
* Loaded the saved model using joblib.load().
* Created number input fields for the five independent variables.
* Collected user inputs.
* Organized the inputs into a DataFrame with matching feature names.
* Used the trained model to predict GPA.
* Displayed the predicted GPA in the application.

### Running the Application


python -m streamlit run student_app.py


## 9. Project Files

student/
│
├── student_eda.ipynb
├── student_app.py
├── student_gpa_model.pkl
└── README.md

* student_eda.ipynb – EDA, data preparation, model training, and evaluation.
* student_app.py – Streamlit application.
* student_gpa_model.pkl – Saved Linear Regression model.
* README.md – Documentation of the steps performed.
