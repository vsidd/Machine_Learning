# Spam Classifier
This program builds a spam classifier using machine learning and ElasticSearch.

A dataset consisting of 75000 emails (both spam and ham) is split into two, 80% for training and 20% for testing.

Two different approaches are followed to build the spam classifier model.

### Part1_learnAndClassify:
This program classifies the emails based on user defined features.
A full matrix representation model is used for both training and testing.
Liblinear's linear regression classifier is used and an accuracy of 50% is achieved.


### Part2_learnAndClassify:
This program classifies emails by considering all the unigrams available in the dataset as features.
With the feature list coming upto 2.5 million words (both proper english and garbage), 
a sparse matrix representation is required for the machine learning algorithm to work.
Liblinear's linear regression classifier is used with the sparse matrix representation for training. 
The testing phase gave an accuracy of 99% in this approach.