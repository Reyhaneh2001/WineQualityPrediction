# **Wine Classification using Machine Learning**

## **Project Overview**
In this project, we aim to classify different types of wines based on a variety of chemical features using machine learning models. The dataset, **Wine dataset**, contains 13 chemical features such as alcohol content, malic acid levels, ash content, and more, which are used to distinguish between three different classes of wines.

The goal is to preprocess the data, train multiple machine learning models, evaluate their performance using various metrics, and visualize the results. We will also conduct **hyperparameter tuning** to optimize the models and select the best-performing one.

## **Steps Taken in the Project**
1. **Exploratory Data Analysis (EDA)**: We start by visualizing the correlations between different features and performing a Principal Component Analysis (PCA) to reduce the dimensionality and visualize the data in 2D.
   
2. **Data Preprocessing**: We standardize the features using `StandardScaler` and split the data into training and testing sets, ensuring a balanced representation of wine classes in both sets.

3. **Model Training**: We evaluate multiple machine learning models, including:
   - **Logistic Regression**
   - **Ridge Classifier**
   - **k-Nearest Neighbors (k-NN)**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Extra Trees Classifier**
   - **Support Vector Machine (SVM)**
   - **Gradient Boosting**
   - **AdaBoost**
   - **Naive Bayes**
   - **Linear Discriminant Analysis (LDA)**

4. **Model Evaluation**: For each model, we use **accuracy**, **F1 score**, **precision**, **recall**, and **AUC** as performance metrics. We also plot **confusion matrices** to visualize the model's classification performance and learning curves to observe the model's behavior during training.

5. **Hyperparameter Tuning**: Using **GridSearchCV**, we fine-tune the hyperparameters for each model to find the optimal combination that improves performance.
