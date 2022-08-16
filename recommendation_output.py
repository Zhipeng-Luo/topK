
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict


@st.cache
def load_output(url):
    df = pd.DataFrame()
    tem = pd.read_csv(url)
    df['title'] = tem['title']
    df['category'] = tem['genre'].str.lower()
    df['score'] = tem['prediction']
    return df

def load_history(url):
    df = pd.DataFrame()
    tem = pd.read_excel(url)
    df['title'] = tem['Title']
    df['category'] = tem['Category']
    df['date'] = tem['Date'].astype(str)
    return df

# data load
user = load_history('ReviewHistory_user_8009.xlsx')

st.set_page_config(layout="wide")

st.title('Recommendation Dashboard')

# Introduction
st.header('1. Introduction')
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    This is the dashboard to compare the recommended results by 5 different models based on the e-commerce data of 
    [Amazon](https://snap.stanford.edu/data/web-Amazon.html). After data manipulation and filtering, there are totally 
    over 5 million pieces of reviews, 830k unique users, 63k unique items, and 36 main categories of products. 

    """)

with col2:
    metrics = {'SLi-Rec':[0.4128, 0.6699], 'SAS-Rec':[0.3929, 0.61], 'LightGBM':[0.0725,0.1631],'Wide&Deep':[0.1256,0.2781],'xDeepFM':[0.1881,0.3497]}
    metrics_df = pd.DataFrame(data=metrics, index=['NDCG@10', 'Hit@10'])
    st.dataframe(metrics_df)
    st.markdown("""
    *The metrics NDCG@10, Hit@10 has been used in a manner described in the SLi-Rec and SASRec papers. 
    Each positive sample is taken from the test set and an additional 50 negative samples are generated from the products
    not seen by the reviewer in this test record.
    """)

# Understand the specific user
st.header('2. The User with Most Reviews')
st.markdown("""
    To check and study the differences of the recommended results, we pick the user who did 324 times of reviews in this dataset as example.
    First of all, we would like to understand his behavior.

    """)

st.subheader('2.1 Categories of the Reviewed Products')

# generate the category dataframe
categories = user['category'].str.split('|')
d = defaultdict(int)
for cat_list in categories:
    for cat in cat_list:
        d[cat]+=1
df_category = pd.DataFrame(d.items(), columns=['category', 'counts']).sort_values(by='counts', ascending=False)

fig1 = px.bar(df_category, x='category',y='counts')
st.plotly_chart(fig1, use_container_width=True)

st.subheader('2.2 The Latest 10 Reviews')
st.dataframe(user.head(10))

with st.sidebar:
    st.title('Categories')
    with st.form(key='my_form'):
        text_input = st.text_input(label='Search:')
        submit_button = st.form_submit_button(label='Submit')

st.header('3. Top10 Output')
st.subheader('3.1 Hybrid Models')

# xDeepFM
st.subheader('xDeepFM')
df_xdeepfm = load_output('xdeepfm_output_8009.csv')
if submit_button:
    df_xdeepfm_out = df_xdeepfm[df_xdeepfm['category'].str.contains(text_input.lower())].sort_values(by='score', ascending=False)
    if len(df_xdeepfm_out) >10:
        st.markdown('There are more than 10 items can be recommended. The top 10 items are:')
        st.dataframe(df_xdeepfm_out.head(10))
    elif len(df_xdeepfm_out) >0:
        st.markdown('There are less than 10 items can be recommended, including:')
        st.dataframe(df_xdeepfm_out)
    else:
        st.markdown('No items under such category, please resubmit your interests!')
else:
        st.markdown('Please search the item !')

# Wide & Deep
st.subheader('Wide & Deep')
df_wnd = load_output('wnd_output_8009.csv')
if submit_button:
    df_wnd_out = df_wnd[df_wnd['category'].str.contains(text_input.lower())].sort_values(by='score', ascending=False)
    if len(df_wnd_out) >10:
        st.markdown('There are more than 10 items can be recommended. The top 10 items are:')
        st.dataframe(df_wnd_out.head(10))
    elif len(df_wnd_out) >0:
        st.markdown('There are less than 10 items can be recommended, including:')
        st.dataframe(df_wnd_out)
    else:
        st.markdown('No items under such category, please resubmit your interests!')
else:
        st.markdown('Please search the item !')

# LightGBM
st.subheader('LightGBM')
df_LGBM = load_output('LGBM_output_8009.csv')
if submit_button:
    df_LGBM_out = df_LGBM[df_LGBM['category'].str.contains(text_input.lower())].sort_values(by='score', ascending=False)
    if len(df_LGBM_out) >10:
        st.markdown('There are more than 10 items can be recommended. The top 10 items are:')
        st.dataframe(df_LGBM_out.head(10))
    elif len(df_LGBM_out) >0:
        st.markdown('There are less than 10 items can be recommended, including:')
        st.dataframe(df_LGBM_out)
    else:
        st.markdown('No items under such category, please resubmit your interests!')
else:
        st.markdown('Please search the item !')

# SLi-Rec
st.subheader('SLi-Rec')
df_sli = load_output('sli_output_8009.csv')
if submit_button:
    df_sli_out = df_sli[df_sli['category'].str.contains(text_input.lower())].sort_values(by='score', ascending=False)
    if len(df_sli_out) >10:
        st.markdown('There are more than 10 items can be recommended. The top 10 items are:')
        st.dataframe(df_sli_out.head(10))
    elif len(df_sli_out) >0:
        st.markdown('There are less than 10 items can be recommended, including:')
        st.dataframe(df_sli_out)
    else:
        st.markdown('No items under such category, please resubmit your interests!')
else:
        st.markdown('Please search the item !')

# SAS-Rec
st.subheader('SAS-Rec')
df_sas = load_output('sas_output_8009.csv')
if submit_button:
    df_sas_out = df_sas[df_sas['category'].str.contains(text_input.lower())].sort_values(by='score', ascending=False)
    if len(df_sas_out) >10:
        st.markdown('There are more than 10 items can be recommended. The top 10 items are:')
        st.dataframe(df_sas_out.head(10))
    elif len(df_sas_out) >0:
        st.markdown('There are less than 10 items can be recommended, including:')
        st.dataframe(df_sas_out)
    else:
        st.markdown('No items under such category, please resubmit your interests!')
else:
        st.markdown('Please search the item !')


st.header('Reference')
st.markdown("""

    [Microsoft-Recommenders](https://github.com/microsoft/recommenders)

    [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/)

    [SLi-Rec](https://www.microsoft.com/en-us/research/uploads/prod/2019/07/IJCAI19-ready_v1.pdf)

    [SAS-Rec](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf)

    [Wide & Deep Learning](https://arxiv.org/pdf/1606.07792.pdf)

    [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender System](https://arxiv.org/pdf/1803.05170.pdf)

    """)