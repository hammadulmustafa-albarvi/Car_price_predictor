import streamlit as st 
import pandas as pd 
import numpy as np 
# from sklearn.metrics import r2_score,mean_absolute_error

import pickle

with open('pkl/df.pkl','rb') as file:
    df = pickle.load(file)
    
    
with open('pkl/price_model.pkl','rb') as file:
    pipeline = pickle.load(file)
    
whole_data = pd.read_csv('datasets/pakwheels_website.csv')

# X = whole_data.drop(columns=['price'])
# y = np.log1p(whole_data['price'])
# X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)
# a = pipeline.fit(X_train,y_train)
# y_pred = pipeline.predict(X_test)
# score = r2_score(y_test,y_pred)
# mean_abs_error = mean_absolute_error(np.expm1(y_test),np.expm1(y_pred))
# print(f'r2 : {score} , mae : {mean_abs_error}')

st.title('Family Car Price Predictor')




year = st.number_input("Year", min_value=1970, step=1)
km = st.number_input("Enter KM Travelled", min_value=0, step=1)
fuel_types =  st.selectbox('Fuel Type',df['fuel_type'].unique().tolist())
register_city  = st.selectbox('Area of Registration',df['city'].unique().tolist())
assemble = st.selectbox('Assembly type',df['assembly'].unique().tolist())
engine = int(st.selectbox('Engine(cc)',df['engine'].unique().tolist()))
body  = st.selectbox('Body Type',df['body_type'].unique().tolist())
int_score = st.selectbox('Interior Quality',df['interior_scores'].unique().tolist())
ext_score = st.selectbox('Exterior Quality',df['exterior_scores'].unique().tolist())
comfort_score = st.selectbox('Comfort Quality',df['comfort_scores'].unique().tolist())
company = st.selectbox('Company',df['company'].unique().tolist())

st.session_state['price'] = 0
if st.button('Predict'):
    test_data = [[year, km, fuel_types, register_city, assemble, engine, body,int_score,ext_score,comfort_score,company] ]
    df_test = pd.DataFrame(test_data,columns=df.columns)
    predicted = np.expm1(pipeline.predict(df_test))[0]
    low = predicted-0.065
    upper = predicted+0.065
    st.session_state['price'] = predicted
    # st.session_state['price'] = predicted
    st.text(f'The price predicted by model is  {round(predicted,2)}')
    st.text("The price of car is between {} and {}".format(round(low,2),round(upper,2)))
    st.header('Cars')
    st.dataframe(whole_data[(whole_data['year']==year)    & (whole_data['engine']==engine) & (whole_data['body_type']==body)    & (whole_data['company']==company) & (whole_data['price']>=low) & (whole_data['price']<=upper)])
    
    
    
st.title('Inference Module')

with open('pkl/betas.pkl','rb') as file:
    un_stand = pickle.load(file)
    
company_names = []
body_types = []

for i in list(un_stand.keys()):
  if 'body_type' in i:
    body_types.append(i.split('_')[-1])
    
for i in list(un_stand.keys()):
  if 'company_' in i:
    company_names.append(i.split('_')[-1])
    
    
st.header('Choose a feature On which you want to see How Price changes')

choice = st.selectbox('Choose One Feature',['Year','Km travelled','Engine','Body Type','Company'])

if choice=='Year':
    y = st.number_input('Select Year',min_value=1970,step=1) - year
    a = np.expm1(un_stand['year'] * y) * 100 
    if a < 0:
        st.text(f"The price would decrease by {round(abs(a),2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
    
    else :
        st.text(f"The price would increase by {round(a,2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
        
        
elif choice=='Km travelled':
    y = st.number_input('Select Km Travelled',min_value=0,step=1) - km
    a = np.expm1(un_stand['Km_travelled'] * y) * 100 
    if a < 0:
        st.text(f"The price would decrease by {round(abs(a),2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
    
    else :
        st.text(f"The price would increase by {round(a,2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
        
        
        
elif choice=='Engine':
    y = st.number_input('Select Engine in (cc)',min_value=0,step=1) - engine
    a = np.expm1(un_stand['engine'] * y) * 100 
    if a < 0:
        st.text(f"The price would decrease by {round(abs(a),2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
    
    else :
        st.text(f"The price would increase by {round(a,2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
        
            

    
elif choice=='Body Type':
    y = st.selectbox('Select Body Type Different From the one you selected for prediction',body_types)
    
    a = np.expm1(un_stand[f'body_type_{y}'] - un_stand[f'body_type_{body}']) * 100 
    
    if a < 0:
        st.text(f"The price would decrease by {round(abs(a),2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
    
    else :
        st.text(f"The price would increase by {round(a,2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
        
        
else:
    y = st.selectbox('Select Body Type Different From the one you selected for prediction',company_names)
    
    a = np.expm1(un_stand[f'company_{y}'] - un_stand[f'company_{company}']) * 100 
    
    if a < 0:
        st.text(f"The price would decrease by {round(abs(a),2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
    
    else :
        st.text(f"The price would increase by {round(a,2)}% , The new approx price would be {round(st.session_state['price'] + st.session_state['price'] * (a/100),2)}")
