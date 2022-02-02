# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:01:22 2021

@author: pc
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html = True
 )
    
@st.cache
def get_data(fileName):
    als_hx= pd.read_csv(fileName)
    return als_hx
    
with header:
    st.title('Welcome to my awesome data science project!')
    st.text('In this project I look into the transactions of taxis..')
    
with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on ....' )
    
    als_hx = get_data('data/als_hx.csv')
    st.write(als_hx.head())
    
    st.subheader('Pick-up location ID distribution on the NYC dataset')
    sns.distplot(als_hx['onset_delta'])
    st.pyplot()
    
with features:
    st.header('The features I created')
    
    st.markdown('* **first feature:** I created this feature because of this... I calculated this...')
    st.markdown('* **second feature:** I created this feature because of this... I calculated this...')
    
with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performace changes')
    
    sel_col, disp_col = st.columns(2)
    
    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value = 100, value=20, step=10)
    
    n_estimators = sel_col.selectbox('How many trees should be there be?', options=[100,200,300, 'No limit'] )
    
    input_features = sel_col.text_input('Input features', 'AB')
    
    
    disp_col.subheader('MSE')
    disp_col.subheader('Surivival curve')
    
    
    
    
    