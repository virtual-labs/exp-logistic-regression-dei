The objective of this experiment is to classify patients into two categories—dengue positive and dengue negative, based on a set of clinical features. The input to the model consists of multiple independent variables representing clinical attributes, while the output is a binary dependent variable indicating the presence or absence of the disease.

The dataset has 5000 rows and 17 columns out of which 16 columns (features) are Independent variables or Input features (X) and the last column i.e. dengue is dependent variable or output (Y). In which, each row represent samples of one patient.

1.  **Import Libraries**: Initialize the environment by importing the required libraries: `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.
2.  **Load Dataset**: Load the `Dengue_dataset.csv` file into a DataFrame for processing.
3.  **Data Analysis**: Conduct initial data exploration using `head()`, `tail()`, `info()`, `describe()`, and `isnull().sum()` to understand the dataset structure and quality.
4.  **Define Target and Features**: Identify the target variable `Dengue` and define the feature set by excluding the target column from the dataset.
5.  **Split Dataset**: Partition the dataset into training (80%) and testing (20%) sets to facilitate model validation.
6.  **Feature Selection**: Select the appropriate numerical features required for the model training process.
7.  **Model Training**: Train the Logistic Regression classifier using the training dataset.
8.  **Predict Outputs**: Use the trained model to generate predictions for the test data.
9.  **Model Evaluation**: Assess performance using key metrics such as Accuracy, Precision, Recall, and F1-Score.
10. **Confusion Matrix**: Plot the confusion matrix for both training and testing datasets to visualize prediction accuracy.
11. **ROC Curve**: Plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC) value.
12. **Sigmoid Visualization**: Visualize the sigmoid curve for individual features to observe the probabilistic relationship with the target variable.
