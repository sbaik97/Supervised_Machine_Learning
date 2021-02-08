# Supervised_Machine_Learning
Credit Risk Prediction Using  the Machine Learning Model

The purpose of this analysis was to create a supervised machine learning model that could accurately predict credit risk. In order to complete this task, I used 6 different methods, which are:

1. Naive Random Oversampling
2. SMOTE Oversampling
3. Cluster Centroid Undersampling
4. SMOTEENN Sampling
5. Balanced Random Forest Classifying
6. Easy Ensemble Classifying

Through each of these methods, I split my data into training and testing datasets, and compiled accuracy scores, confusion matries, and classification reports as my results.

## Results

### Naive Random Oversampling

![oversampling](images/Random_overSampling.png)

### SMOTE Oversampling


![SMOTE](images/SMOTE_oversampling.png)

### Cluster Centroid Undersampling


![undersampling](images/RandomUnderSampling.png)

### SMOTEENN Sampling

![SMOTEENN](images/SMOTEENN_combination_sampling.png)

### Balanced Random Forest Classifying


![random_forest](images/BalancedRandomForestClassifier.png)

### Easy Ensemble Classifying

![easy_ensemble](images/EasyEnsembleClassifier.png)

## Summary

This analysis is trying to find the best model that can detect if a loan is high risk or not. Becasue of that, we need to find a model that lets the least amount of high risk loans pass through undetected. That correlating statistic for this is the recall rate for high risk. Looking through the different models, the ones that scored the highest were:

1. Easy Ensemble Classifying (91%)
2. SMOTEENN Sampling (63.0%)
3. Naive Random Oversampling (66.1%)

While this is the most important statistic that is pulled from this analysis, another important statistic is recall rate for low risk as it shows how many low risk loans are flagged as high risk. Looking through the different models, the ones that scored the highest were:

1. Balanced Random Forest Classifying (100%)
2. Easy Ensemble Classifying (94%)

After taking these two statistics over the others, we can look at the accurary score to get a picture of how well the model performs in general. The models with the highest accuracy scores were:

1. Easy Ensemble Classify (92.3%)
2. SMOTEENN Sampling (68.1%)
3. Balanced Random Forest Classifying (64.8%)

After factoring in these three main statistics, the model that I would recommend to use for predicting high risk loans is the Easy Ensemble Classifying model.
