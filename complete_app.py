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

st.title('Predict 2nd Heart Attack')

df = pd.read_csv('https://raw.githubusercontent.com/bonchae/data/master/heartattack_train.csv') 

if st.checkbox('Show dataframe'):
    st.write(df)

if st.checkbox('Do you want to know the overall probability of a 2nd heart attack?'):
    a = (df.groupby('Heart_Attack').size()/df.shape[0]).round(2).reset_index() 
    a.columns=['Heart Attack','Probability']
    fig=px.bar(a,x='Heart Attack',y='Probability',text='Probability')
    st.plotly_chart(fig)

st.subheader('Scatter plot')

new_df = df[['Age','Cholesterol','Trait_Anxiety']]

col1 = st.selectbox('Which feature on x?', new_df.columns)
col2 = st.selectbox('Which feature on y?', new_df.columns)

# create figure using plotly express
fig = px.scatter(df, x =col1,y=col2, color='Heart_Attack')

# Plot!
st.plotly_chart(fig)

st.subheader('Histogram')

feature = st.selectbox('Which feature?', df.columns[0:6])

# Filter dataframe
fig2 = px.histogram(df, x=feature, color="Heart_Attack", marginal="rug")
st.plotly_chart(fig2)

#mappling or replacing
df = df.replace({'Heart_Attack': 'No'}, {'Heart_Attack': '0'})
df = df.replace({'Heart_Attack': 'Yes'}, {'Heart_Attack': '1'})
df['Heart_Attack'] = df['Heart_Attack'].astype(int)
df =  pd.get_dummies(df, columns=["Marital_Status", "Gender", "Weight_Category", "Stress_Management"],
                         prefix=["Marital_Status", "Gender", "Weight_Category", "Stress_Management"],
                         drop_first=True)


##########################################################
# Accepting user data for predicting its Member Type
def accept_user_data():
    age = st.slider("Choose your age: ", min_value=0,   
                       max_value=100, value=35, step=1)
    cho = st.slider("Choose your cholesterol level: ", min_value=120,
                       max_value=250, value=180, step=1)
    anx = st.slider("Choose your trait anxiety level: ", min_value=20,
                     max_value=100, value=50, step=1)
    mar = st.selectbox("What is your martital status?", ('Single', 'Married', 'Divorced', 'Widow'))
    if mar=='Single':
        mar1=0
        mar2=0
        mar3=0
    elif mar=='Married':
        mar1=1
        mar2=0
        mar3=0
    elif mar=='Divorced':
        mar1=0
        mar2=1
        mar3=0
    elif mar=='Widow':
        mar1=0
        mar2=0
        mar3=1
    gen = st.selectbox("Are you a male?", 
                    ('Yes', 'No'))
    if gen=='Yes':
        gen1=1
    else:
        gen1=0
    wei = st.selectbox("What is your weight category?", 
                    ('Normal', 'Overweight', 'Obese'))
    if wei=='Normal':
        wei1=0
        wei2=0
    elif wei=='Overweight':
        wei1=1
        wei2=0        
    elif wei=='Obese':
        wei1=0
        wei2=1
    tra=st.selectbox("Did you receive stress management training?", 
                    ('Yes', 'No'))
    if tra=='Yes':
        tra1=1
    else:
        tra1=0  

    user_prediction_data = [[age,cho,anx,mar1,mar2,mar3,gen1,wei1,wei2,tra1]]
    return user_prediction_data
##########################################################

st.subheader('Machine Learning models')
 
y = df['Heart_Attack'].values
X = df.drop(['Heart_Attack'], axis=1).values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
 
alg = ['Decision Tree', 'Logistic Regression']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)
 
    try:
        st.subheader('Predict your own input')
        if(st.checkbox("Want to predict on your own Input? ")):
            user_prediction_data = accept_user_data()
            st.write("You entered : ", user_prediction_data)  
            pred = dtc.predict(user_prediction_data)
            st.write("The Predicted Class is: ", pred)
            pred_prob = dtc.predict_proba(user_prediction_data)
            st.write("The Probability is: ", pred_prob)           
    except:
    	pass

elif classifier == 'Logistic Regression':
    lr = LogisticRegression(solver='lbfgs', max_iter=500)
    lr.fit(X_train, y_train)
    acc = lr.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_lr = lr.predict(X_test)
    cm=confusion_matrix(y_test, pred_lr)
    st.write('Confusion matrix: ', cm)

    try:
        st.subheader('Predict your own input')
        if(st.checkbox("Want to predict on your own Input? ")):
            user_prediction_data = accept_user_data()
            st.write("You entered : ", user_prediction_data)  
            pred = lr.predict(user_prediction_data)
            st.write("The Predicted Class is: ", pred)
            pred_prob = lr.predict_proba(user_prediction_data)
            st.write("The Probability is: ", pred_prob)            
    except:
    	pass

st.markdown("## Party time!")
st.write("Yay! You're done with this tutorial of Streamlit. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()