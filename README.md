# Capstone-Project

This project is to analyse Arvato’s dataset provided by Udacity during the Data-science Nano degree program. 

There are two datasets with information about customers from a mail order company and similar information about the general population.

Goal is to perform a clustering algorithm on general population dataset and apply that on customer’s dataset to identify potential customer base for the company so that those people can be targeted by the company.

Later a supervised algorithm will be applied on the similar dataset to predict whether the user will respond to the mailout campaign.

Datasets provided by Udacity as part of Datascience Nano degree Capstone project.

Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).

Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).

Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).

Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).

Libraries used

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.utils import resample

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc
