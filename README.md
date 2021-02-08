# Supervised_Machine_Learning
**Credit Risk Prediction Using  the Machine Learning Model**

## Background
The purpose of this analysis is to build and evaluate several machine
learning models to predict credit risk. Being able to predict credit risk with different machine learning algorithms can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud. 


## Objects
1. Define machine learning algorithm used in data analytics and create training and test groups from a given data set.
2. Implement and interpret the logistic regression, decision tree, random forest, and support vector machine (SVM) algorithms.
3. Compare the advantages and disadvantages of each supervised learning
algorithm and determine which supervised learning algorithm is best
used for a given data set or scenario.
4. Use ensemble and resampling techniques to to resolve class imbalance and improve model performance.


## Software/Tools/Libraries
* Jupyter Notebook 6.1.4 with numpy, pandas, pathlib, collections.   scikit-learn with train_test_split, LogisticRegression, balanced_accuracy_score, confusion_matrix,
imbalanced-learn with classification_report_imbalanced,  RandomOverSampler, SMOTE, RandomUnderSampler, SMOTEENN.

* Data Source: 
LoanStats_2019Q1.csv (https://2u-data-curriculum-team.s3.amazonaws.com/datavizonline/module_17/Module-17-Challenge-Resources.zip)



## Task 1. Use Resampling Models to Predict Credit Risk

**Objects**: Using the knowledge of the imbalanced-learn and scikit-learn libraries, evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. Use the oversampling RandomOverSampler and SMOTE algorithms, and  undersampling ClusterCentroids algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classification. 

### Results

### Naive Random Oversampling
**Class imbalance** refers to a situation in which the existing classes in a dataset aren't equally represented. In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

![oversampling](images/Random_overSampling.png)

### SMOTE Oversampling
The SMOTE Oversampling is synthetic minority oversampling technique. In SMOTE, new instances are interpolated to deal with unbalanced datasets. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

![SMOTE](images/SMOTE_oversampling.png)

### Cluster Centroid Undersampling
Undersampling is another technique to address class imbalance. Undersampling takes the opposite approach of oversampling and the size of the majority class is decreased. In random undersampling, randomly selected instances from the majority class are removed until the size of the majority class is reduced, typically to that of the minority class.

![undersampling](images/RandomUnderSampling.png)

### Summary

Herein, we compared three sampling methods, Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling,to predict the credit risk. We use the LogisticRegression classifier and evaluate the model’s performance. Looking through the different sampling models, the accuracy score of three models were 66, 63, and 59 % and the recall rate for low risk 67, 64, 57 %. All three sampling show that the precision ("pre" column) and recall ("rec" column) are high for the majority class, precision is low for the minority class.
Eveno thouth the Oversampling outperform other sampling models, it's important to note that although SMOTE reduces the risk of oversampling by deleting the outliers. 


## Task 2. Use the SMOTEENN Sampling Method

**Objects**: Use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from task 1.

### SMOTEENN Sampling
SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:
1. Oversample the minority class with SMOTE.
2. Clean the resulting data with an undersampling strategy. If the two
nearest neighbors of a data point belong to two different classes, that
data point is dropped.

![SMOTEENN](images/SMOTEENN_combination_sampling.png)

### Summary
Looking through the different sampling models, the accuracy score of three SMOTEENN Sampling were 64 % and the recall rate for low risk 57 %. This combinatorial sampling also show that the precision ("pre" column) and recall ("rec" column) are high for the majority class, precision is low for the minority class. However, resampling with SMOTEENN did not show the sharp imporemoment, but some of the metrics show an improvement over undersampling.


## Task 3. Use Ensemble Classifiers to Predict Credit Risk


### Balanced Random Forest Classifying


![random_forest](images/BalancedRandomForestClassifier.png)

![sorted_feature_descending_order](images/sorted_feature_descending_order.png)

### Easy Ensemble Classifying

![easy_ensemble](images/EasyEnsembleClassifier.png)

## Summary





## Overall Summary

This analysis is trying to find the best model that can detect if a loan is high risk or not. Becasue of that, we need to find a model that lets the least amount of high risk loans pass through undetected. That correlating statistic for this is the recall rate for high risk. 
