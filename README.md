# Identify Fraud from Enron Data
## Final Project of Machine Learning Course
## Udacity Data Analyst Nanodegree
### Final project can be found in final_project folder.
- **final_project.html**: This document contains the details of what I did for the final project.
- **final_project_v2.ipynb**: This is the second version of the jupyter notebook file of the final project.
- **poi_id.py**: It contains the features and a classifier I choose for finding person of interest (POI) in the enron data set.
- **tester.py**: Accuracy, precision and recall can be calculated with this script.

### Note: This is the second version of the final project. Below is the list of updates I made.

- Created a new feature "eso_deferred_income" = deferred_income / exercised_stock_options
- Used **SelectKBest** and **feature importances** in **Decision Tree** to filter the low ranking features.
- Explained the scaling method I employed for the project more explicitly.
- Modified **GridSearchCV** scroing parameter to **f1** and **average_precision**. By default, it tries to maximize accuracy and we can get a high accuracy just by assuming everyone is non-POI.
- **GridSearchCV** was run multiple times over cross validation iterations. This was the main reason why the code ran so slow. To fix this, I took the best parameters from **GridSearchCV** and fixed those parameters in the model.
- **Linear Regression** and **Lasso Regression** are used when we have a continuous output. However, in this project, the output is discrete values: POI or non-POI. **Logistic Regression** is more appropriate algorithm to use.
- Answers to the project questionnaire in the **final thoughts** section.
