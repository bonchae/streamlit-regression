# Credits: 
# https://datasciencechalktalk.com/2019/10/22/building-machine-learning-apps-with-streamlit/
# https://towardsdatascience.com/streamlit-101-an-in-depth-introduction-fc8aad9492f2

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Classifiers
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#for validating your classification model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


df = pd.read_csv('https://raw.githubusercontent.com/bonchae/data/master/heartattack_train.csv') 


