#!/usr/bin/env python
# coding: utf-8

# # EATC Assignment StreamLit Application Notebook

# In[297]:


# import libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import streamlit as st

st.set_page_config(
    page_title="Cybersecurity Intelligence Dashboard",
    layout="wide"
)

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy


# ## 1  Data Preprocessing

# In[299]:


# raw dataset
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')


# In[300]:


# creating duplicate df to retain for visualization creation
df_raw = df.copy() 


# In[301]:


# creating df for model training data
df_model = df.drop(['Year', 'Defense Mechanism Used'], axis=1)


# In[302]:


# creating features and labels
y = df['Attack Source']
X = df.drop(columns=['Attack Source'])


# In[303]:


# one-hot encoding categorical labels
X = pd.get_dummies(X, dtype='int64')


# In[304]:


# one-hot encoding labels (multi-class classification)
y = pd.get_dummies(y, dtype='int64')


# In[305]:


# initial train-test split (training set=7%, test+validation set=30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7)


# In[306]:


# second train-test split (testing and validation sets, 15% each)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=0.5)


# In[307]:


# scaling numerical data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# In[308]:


# setting initial input shape for model 
input_shape = [X_train.shape[1]]


# In[309]:


# setting final output shape for model
n_outputs = y_train.shape[1] 


# In[310]:


# setting fixed decision threshold (can adjust accordingly)
THRESHOLD = 0.265


# ## 2 Modelling

# ### 2.1 Model Development

# In[313]:


# instantiating model
model = Sequential()


# In[314]:


# instantiating training parameters and metrics
optimizer = Adam(learning_rate=0.01)
early_stop = EarlyStopping(patience=5, restore_best_weights=True, start_from_epoch=5)
bin_acc = BinaryAccuracy(threshold=0.5)


# In[315]:


# adding layers to model
model.add(Dense(units=128, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4))) # first hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu', input_shape=128, kernel_regularizer=regularizers.l2(1e-5))) # second hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=4, activation='sigmoid', input_shape=32)) # output layer


# In[316]:


# compiling model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[bin_acc])


# In[317]:


# training model
history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=32, epochs=64, callbacks=[early_stop])


# In[318]:


# making predictions
y_pred = model.predict(X_test)


# ### 2.2 Model Evaluation

# #### 2.2.1 Training Binary Accuracy Curve

# In[321]:


# plotting binary accuracy curve
history_df = pd.DataFrame(history.history)
history_df.loc[:, 'binary_accuracy'].plot()
plt.xlabel('Epoch')
plt.ylabel('Binary accuracy')
plt.show()


# #### 2.2.2 Binary Validation Accuracy (binary_val_accuracy) Curve

# In[323]:


# plotting binary_val_accuracy curve
history_df = pd.DataFrame(history.history)
history_df.loc[:, 'val_binary_accuracy'].plot()
plt.xlabel('Epoch')
plt.ylabel('Binary Validation Accuracy')
plt.show()


# #### 2.2.3 Validation Loss (val_loss) curve

# In[325]:


# plotting val_loss curve
history_df = pd.DataFrame(history.history)
history_df.loc[:, 'val_loss'].plot()
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.show()


# ## 3 StreamLit

# In[327]:


# Streamlit web app UI 
st.title("Cybersecurity Intelligence Dashboard")
st.caption("Vulnerability Heatmaps • Attack Source Profiling")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Vulnerability Heatmaps", "Attack Source Prediction"])


# In[328]:


# tab 1: vulnerability heatmaps 
with tab1:
    st.subheader("Heatmaps by Region and Industry")

    # Two filter columns
    col1, col2 = st.columns(2)
    with col1:
        years_sel = st.multiselect("Filter by Year(s)", sorted(df_raw['Year'].unique()), default=sorted(df_raw['Year'].unique()))
    with col2:
        atk_sel = st.multiselect("Filter by Attack Type(s)", sorted(df_raw['Attack Type'].unique()), default=sorted(df_raw['Attack Type'].unique()))

    # Filter dataset based on selections
    dff = df_raw[df_raw['Year'].isin(years_sel) & df_raw['Attack Type'].isin(atk_sel)]

    # Create heatmap for incidents by country & industry
    heatmap_data = dff.groupby(['Country', 'Target Industry']).size().reset_index(name='Count')
    if not heatmap_data.empty:
        fig = px.density_heatmap(
            heatmap_data, x='Target Industry', y='Country', z='Count',
            color_continuous_scale='Reds', title="Incidents by Country & Industry"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for the selected filters.")

    st.markdown("---")

    # Create heatmap for vulnerabilities by country
    st.subheader("Heatmap: Country × Security Vulnerability Type")
    hv = dff.groupby(['Country', 'Security Vulnerability Type']).size().reset_index(name='Count')
    if not hv.empty:
        fig2 = px.density_heatmap(
            hv, x='Security Vulnerability Type', y='Country', z='Count',
            color_continuous_scale='Blues', title="Vulnerabilities by Country"
        )
        st.plotly_chart(fig2, use_container_width=True)


# In[329]:


# tab 2: attack source prediction
with tab2:
    st.subheader("Predict Attack Source")

    # User inputs for prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        attack_type = st.selectbox('Attack Type', sorted(df_raw['Attack Type'].unique()))
    with col2:
        target_industry = st.selectbox('Target Industry', sorted(df_raw['Target Industry'].unique()))
    with col3:
        sec_vuln_type_input = st.selectbox('Security Vulnerability Type', sorted(df_raw['Security Vulnerability Type'].unique()))

    # Use dataset averages for numerical inputs
    financial_loss = float(df_model['Financial Loss (in Million $)'].mean())
    user_affected = float(df_model['Number of Affected Users'].mean())
    incident_resolution_time = float(df_model['Incident Resolution Time (in Hours)'].mean())

    # predict button
    if st.button('Predict Attack Source'):
        training_columns = X.columns  # Store training feature names
        # Create new data row for prediction
        df_new = pd.DataFrame([[attack_type, target_industry, financial_loss, user_affected, sec_vuln_type_input, incident_resolution_time]],
                              columns=['Attack Type', 'Target Industry', 'Financial Loss (in Million $)',
                                       'Number of Affected Users', 'Security Vulnerability Type', 'Incident Resolution Time (in Hours)'])

        # One-hot encode and align with training data columns
        df_new_encoded = pd.get_dummies(df_new, dtype='int64')
        df_new_encoded = df_new_encoded.reindex(columns=training_columns, fill_value=0)

        # Apply scaling
        df_new_scaled = scaler.transform(df_new_encoded)

        # Predict probabilities
        result = model.predict(df_new_scaled)

        # Map predictions to attack source labels
        classes = list(y.columns)
        pred_labels = [(classes[i], val) for i, val in enumerate(result[0]) if val >= THRESHOLD]

        # Display results
        if pred_labels:
            st.success(f"Predicted Attack Source(s): {" | ".join(f"{label}: {prob:.2f}" for label, prob in pred_labels)}")
        else:
            st.warning("No labels predicted above threshold.")


# In[ ]:




