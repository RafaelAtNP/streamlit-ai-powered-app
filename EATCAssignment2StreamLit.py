#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[70]:


df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')


# In[71]:


df.head()


# In[72]:


df['Attack Source'].value_counts()


# In[73]:


df = df.drop(['Country','Year', 'Defense Mechanism Used'], axis=1)


# In[74]:


df.shape


# In[75]:


df.columns


# In[ ]:




le_AT = LabelEncoder().fit(df['Attack Type'])
le_TI = LabelEncoder().fit(df['Target Industry'])
le_SVT = LabelEncoder().fit(df['Security Vulnerability Type'])
df['Attack Type'] = le_AT.fit_transform(df['Attack Type'])
df['Target Industry'] = le_TI.fit_transform(df['Target Industry'])
df['Security Vulnerability Type'] = le_SVT.fit_transform(df['Security Vulnerability Type'])
# In[76]:


y = df['Attack Source']
X = df.drop(columns=['Attack Source'])


# In[77]:


X = pd.get_dummies(X, dtype='int64')
X.head()


# In[78]:


y = pd.get_dummies(y, dtype='int64')
y.head()


# In[79]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7)


# In[80]:


X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=0.5)


# In[81]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# In[82]:


input_shape = [X_train.shape[1]]
input_shape


# In[83]:


model = Sequential()
optimizer = Adam(learning_rate=0.01)
early_stop = EarlyStopping(patience=5, restore_best_weights=True, start_from_epoch=5)
bin_acc = BinaryAccuracy(threshold=0.5)

model.add(Dense(units=128, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4))) # first hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu', input_shape=128, kernel_regularizer=regularizers.l2(1e-5))) # second hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=4, activation='sigmoid', input_shape=32)) # output layer

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[bin_acc])


# In[84]:


history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=32, epochs=64, callbacks=[early_stop])


# In[85]:


y_pred = model.predict(X_test)
pd.DataFrame(y_pred, columns=y_test.columns).head()

#financial_loss = X['Financial Loss (in Million $)'].mean()
#user_affected = X['Number of Affected Users'].mean()
#incident_reslution_time = X['Incident Resolution Time (in Hours)'].mean()
    
#df_new = pd.DataFrame([['Man-in-the-Middle', 'Telecommunications', financial_loss,user_affected,'Social Engineering',incident_reslution_time]], 
#                        columns=['Attack Type', 'Target Industry', 'Financial Loss (in Million $)',
#                        'Number of Affected Users','Security Vulnerability Type','Incident Resolution Time (in Hours)'])
#df_new_encoded = pd.get_dummies(df_new, dtype='int64')
#df_new_encoded = df_new_encoded.reindex(columns=training_columns, fill_value=0)


#df_new_scaled = scaler.transform(df_new_encoded)

#If Submit button is preseed, do model predict the phishing based on entered traffic number.
# Calculate the phishing number
#result = model.predict(df_new_scaled)

# 5. Reverse one-hot for multi-label
#threshold = 0  # Adjust as needed
#classes = ['Hacker Group', 'Insider', 'Nation-state', 'Unknown']
#pred_labels = [classes[i] for i, val in enumerate(result[0]) if val >= threshold]
#pred_labels
# In[86]:


st.title("Predict Attack Source")
st.write("""
    This is an AI Application that predicts the attack source based on attack tye, target industry and security vulnerability""")


# In[100]:


st.title('ML Model Deployment')

training_columns = X.columns
attack_type = st.text_input('Attack Type', value='')
target_industry = st.text_input('Target Industry', value='')
sec_vuln_type = st.text_input('Security Vulnerability Type', value='')

if st.button('Predict'):

    financial_loss = X['Financial Loss (in Million $)'].mean()
    user_affected = X['Number of Affected Users'].mean()
    incident_reslution_time = X['Incident Resolution Time (in Hours)'].mean()
    
    df_new = pd.DataFrame([[attack_type, target_industry, financial_loss,user_affected,sec_vuln_type,incident_reslution_time]], 
                          columns=['Attack Type', 'Target Industry', 'Financial Loss (in Million $)',
                                   'Number of Affected Users','Security Vulnerability Type','Incident Resolution Time (in Hours)'])
    df_new_encoded = pd.get_dummies(df_new, dtype='int64')
    df_new_encoded = df_new_encoded.reindex(columns=training_columns, fill_value=0)


    df_new_scaled = scaler.transform(df_new_encoded)
    
    #If Submit button is preseed, do model predict the phishing based on entered traffic number.
    # Calculate the phishing number
    result = model.predict(df_new_scaled)

    # 5. Reverse one-hot for multi-label
    threshold = 0  # Adjust as needed
    classes = ['Hacker Group', 'Insider', 'Nation-state', 'Unknown']
    pred_labels = [classes[i] for i, val in enumerate(result[0]) if val >= threshold]

    # 6. Show result
    if pred_labels:
        st.success(f"Predicted labels: {pred_labels}")
    else:
        st.warning("No labels predicted above threshold.")

    
    # Display the predicted phishing number
    #st.success(f'The prediction is: {result}')


# In[ ]:




