import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px
import ast
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns


map_df = pd.read_csv('datasets/geomap.csv')
st.header('Area wise car count and Avg price(in Crore) Of Pakistan')
fig = px.scatter_map(
    map_df,
    lat="latitude",
    lon="longitude",
    hover_name='city',
    size='number_of_cars',
    color="avg_price",
    zoom=5,
    height=600
)

st.plotly_chart(fig,use_container_width=True)







st.header('Word Cloud of Important Features of Car')
with open('pkl/comfort.pkl','rb') as file:
    comfort = pickle.load(file)
    
with open('pkl/safety.pkl','rb') as file:
    safety = pickle.load(file)    

with open('pkl/interior.pkl','rb') as file:
    interior = pickle.load(file)

with open('pkl/exterior.pkl','rb') as file:
    exterior = pickle.load(file)
    
    
    
def create_word_cloud(x):
    fig,ax = plt.subplots(figsize=(10,5))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',   
        max_words=100
    ).generate(x)

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')  
    st.pyplot(fig)
    
x = st.selectbox('Select Feature Type',['interior','exterior','safety','comfort'])

if x=='interior':
    create_word_cloud(interior)
    
elif x=='exterior':
    create_word_cloud(exterior)

elif x=='safety':
    create_word_cloud(safety)
    
else:
    create_word_cloud(comfort)


df = pd.read_csv('datasets/Pakwheels_Confirmed.csv')
st.header('Pie Chart of body type w.r.t Company')

company = st.selectbox('Select Company',df['company'].unique().tolist())
fig = px.pie(df[df['company']==company],names='body_type')
st.plotly_chart(fig,use_container_width=True)



st.header('Density Plot of Price vs Body type')
a = st.selectbox('Select 1st Type',df['body_type'].unique().tolist(),key='b1')
b = st.selectbox('Select 2nd Type',df['body_type'].unique().tolist(),key='b2')

figures = plt.figure(figsize=(15,6))
sns.distplot(df[df['body_type']==a]['price'],label='Sedan')
sns.distplot(df[df['body_type']==b]['price'],label='SUV')
plt.legend()
st.pyplot(figures)
    
    
st.header('Price of Company w.r.t to Body Type')
c = st.selectbox('Select Company',df['company'].unique().tolist(),key='p')


figures1 = px.box(df[df['company']==c],x='body_type',y='price') 
st.plotly_chart(figures1,use_container_width=True)
