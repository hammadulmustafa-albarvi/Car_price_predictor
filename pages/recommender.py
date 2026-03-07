import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle


with open('pkl/similarity.pkl','rb') as file:
    similarity = pickle.load(file)
    
    
names = pd.read_csv('datasets/name.csv')

st.title('Electric Cars Recommender System')
def get_similar_car(x):
  idx = names[names['name']==x].index[0]
  cars_idx = sorted(list(enumerate(similarity[idx])),key=lambda x:x[1],reverse=True)[1:6]
  for i in cars_idx:
    st.text(names['name'].iloc[i[0]])
    
    

x = st.selectbox('Select Car',names['name'].unique().tolist())

get_similar_car(x)