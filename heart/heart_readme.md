Heart Disease Prediction Using Logistic Regression

Workflow

1. Exploratory Data Analysis (EDA)

- Imported required libraries: Pandas, NumPy, Matplotlib, and Seaborn.
- Loaded the dataset using "pd.read_csv()".
- Displayed the first five rows using "df.head()".
- Checked dataset dimensions using "df.shape".
- Inspected column names using "df.columns".
- Checked data types and non-null values using "df.info()".
- Generated descriptive statistics using "df.describe()".
- Checked missing values using "df.isnull().sum()".
- Checked duplicate records using "df.duplicated().sum()".
- Examined unique values in each column.
- Analyzed the target variable distribution using "value_counts()" and a count plot.
- Visualized numerical feature distributions using histograms.
- Visualized categorical feature distributions using count plots.
- Compared numerical features against the target using boxplots.
- Compared categorical features against the target using count plots.
- Generated a correlation heatmap for numerical features.
- Inspected numerical features for potential outliers using boxplots.

2. Data Preprocessing

- Separated input features ("X") and the target variable ("y").
- Identified categorical columns using "select_dtypes()".
- Applied one-hot encoding using "pd.get_dummies()" with "drop_first=True".
- Split the dataset into training and testing sets using "train_test_split()".
- Used an 80:20 train-test split.
- Applied stratification using "stratify=y".
- Standardized numerical features using "StandardScaler".
- Fitted the scaler on the training data and transformed both training and testing data.

3. Model Training

- Imported "LogisticRegression" from Scikit-learn.
- Initialized the model using "LogisticRegression(max_iter=1000)".
- Trained the model using "model.fit(X_train, y_train)".

4. Model Prediction

- Generated predictions on the test data using "model.predict(X_test)".
- Compared the first ten predicted values with the actual target values.

5. Model Evaluation

- Calculated model accuracy using "accuracy_score()".
- Generated a confusion matrix using "confusion_matrix()".
- Interpreted true positives, true negatives, false positives, and false negatives.
- Calculated precision, recall, and F1-score from the confusion matrix.
- Prepared to generate the complete classification report using "classification_report()".

6. Results

- Achieved 88.59% test accuracy.
- Obtained 89.32% precision for class 1.
- Obtained 90.20% recall for class 1.
- Obtained 89.75% F1-score for class 1.
- Recorded the confusion matrix: "[[71, 11], [10, 92]]".