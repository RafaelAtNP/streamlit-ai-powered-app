#!/usr/bin/env python
# coding: utf-8

# # EATC Assignment StreamLit Application Notebook

# In[206]:


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

from sklearn.model_selection import train_test_split, TunedThresholdClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from keras import Sequential, regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.metrics import BinaryAccuracy


# ## 1  Data Preprocessing

# ### 1.1 Reading the data

# In[263]:


# loading dataset
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')


# In[265]:


# creating duplicate dataset to use for visualization creation
df_raw = df.copy() 


# In[267]:


# creating duplicate dataset to use as training data for model
df_model = df.copy()


# ### 1.2 Feature Engineering

# #### 1.2.1 Identifying correlations

# In[271]:


# creating heatmap to observe feature correlations
sns.heatmap(df.apply(lambda col: pd.factorize(col)[0]).corr())
plt.show()


# Based on the correlation heatmap, the Financial Loss (in Million $) incurred has a high correlation of around 0.9. This could be because the financial losses incurred due to cyberattacks are directly proportionate to the number of users impacted, i.e. cost increases with each user's system affected / customers churned, etc.  
# 
# Therefore, the two features can be combined to create a new feature Financial Loss per User to retain the data patterns provided by both features while reducing the dimensionality of the data and reducing correlation between features, increasing performance and mitigating the risk of overfitting. 

# #### 1.2.2 Creating new features

# In[275]:


# creating Financial Loss per User feature
df_model['Financial Loss per User'] = ((df_model['Financial Loss (in Million $)'] * 1000000) / df_model['Number of Affected Users']).round(2)


# In[277]:


# removing correlated features from dataset
df_model.drop(['Financial Loss (in Million $)', 'Number of Affected Users'], axis=1, inplace=True)


# In[280]:


# dropping unecessary columns
df_model = df_model.drop(['Year', 'Defense Mechanism Used'], axis=1)


# #### 1.2.3 Creating Features and Labels for Modelling

# In[283]:


# creating features and labels
y = df_model['Attack Source'] 
X = df_model.drop(columns=['Attack Source'])


# In[285]:


# one-hot encoding categorical labels
X = pd.get_dummies(X, dtype='int64')


# In[289]:


# one-hot encoding labels (multi-class classification)
y = pd.get_dummies(y, dtype='int64')


# In[293]:


# initial train-test split (training set=70%, test+validation set=30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7)


# In[295]:


# second train-test split (testing and validation sets, 15% each)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, train_size=0.5)


# In[297]:


# scaling numerical data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)


# In[299]:


# setting initial input shape for model 
input_shape = [X_train.shape[1]]


# In[301]:


# setting final output shape for model
n_outputs = y_train.shape[1] 


# ## 2 Modelling

# ### 2.1 Model Development

# #### 2.1.1 Instantiating model, parameters, metrics, and callbacks

# In[306]:


# instantiating model
model = Sequential()


# In[308]:


# instantiating training parameters and metrics
optimizer = Adam(learning_rate=0.001)
early_stop = EarlyStopping(patience=5, restore_best_weights=True, start_from_epoch=5)
bin_acc = BinaryAccuracy(threshold=0.5)


# In[310]:


# adding layers to model
model.add(Dense(units=128, activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(1e-4))) # first hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu', input_shape=128, kernel_regularizer=regularizers.l2(1e-5))) # second hidden layer
model.add(Dropout(0.3))
model.add(Dense(units=4, activation='sigmoid', input_shape=32)) # output layer


# In[312]:


# compiling model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[bin_acc])


# In[314]:


# training model
history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=32, epochs=64, callbacks=[early_stop])


# In[334]:


# determining and setting decision threshold
from sklearn.metrics import f1_score

# predict using validation set
y_prob = model.predict(X_val) 

# search thresholds from 0.0 to 1.0
thresholds = np.arange(0.0, 1.01, 0.01)
best_threshold, best_f1 = 0, -1

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    f1 = f1_score(y_val, y_pred, average="macro")  # can also use 'micro' if necessary
    
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

# print(f"Best Threshold: {best_threshold:.2f}  |  Best F1: {best_f1:.4f}")

THRESHOLD = best_threshold 


# ### 2.2 Model Evaluation

# #### 2.2.1 Training Binary Accuracy Curve

# In[320]:


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

# In[326]:


# plotting val_loss curve
history_df = pd.DataFrame(history.history)
history_df.loc[:, 'val_loss'].plot()
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.show()


# ## 3 StreamLit

# In[338]:


# Streamlit web app UI 
st.title("Cybersecurity Intelligence Dashboard")
st.caption("Attack Source Profiling • Vulnerability Heatmaps")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Attack Source Prediction", "Vulnerability Heatmaps"])


# In[340]:


# tab 1: attack source prediction
with tab1:
    st.subheader("Predict Attack Source")

    # User inputs for prediction
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        country = st.selectbox('Country', sorted(df_raw['Country'].unique()))
    with col2:
        attack_type = st.selectbox('Attack Type', sorted(df_raw['Attack Type'].unique()))
    with col3:
        target_industry = st.selectbox('Target Industry', sorted(df_raw['Target Industry'].unique()))
    with col4:
        sec_vuln_type_input = st.selectbox('Security Vulnerability Type', sorted(df_raw['Security Vulnerability Type'].unique()))

    # Use dataset averages for numerical inputs 
    financial_loss = float(df_model['Financial Loss per User'].mean())
    incident_resolution_time = float(df_model['Incident Resolution Time (in Hours)'].mean())

    # predict button
    if st.button('Predict Attack Source'):
        training_columns = X.columns  # Store training feature names
        # Create new data row for prediction
        prediction_data = pd.DataFrame([[country, attack_type, target_industry, financial_loss, sec_vuln_type_input, incident_resolution_time]])

        # One-hot encode and align with training data columns
        prediction_data = pd.get_dummies(prediction_data, dtype='int64')
        prediction_data  = prediction_data.reindex(columns=training_columns, fill_value=0)

        # Apply scaling
        prediction_data = scaler.transform(prediction_data)

        # Predict probabilities
        result = model.predict(prediction_data)

        # Map predictions to attack source labels
        classes = list(y.columns)
        pred_labels = [(classes[i], val) for i, val in enumerate(result[0]) if val >= THRESHOLD] 
        pred_labels = pred_labels.sort(key=lambda x: x[1], reverse=True) # sort values by probability (highest to lowest)

        # Display results
        if pred_labels:
            st.success(f"Predicted Attack Source(s): {" | ".join(f"{label}: {prob * 100:.2f}%" for label, prob in pred_labels)}")
        else:
            st.warning("No labels predicted above threshold.")


# In[342]:


# tab 2: vulnerability heatmaps 

with tab2:
    st.subheader("Vulnerability Heatmaps")
    
    # Two filter columns
    col1, col2 = st.columns(2)
    
    with col1:
        years_sel = st.multiselect("Filter by Year(s)", sorted(df_raw['Year'].unique()), default=sorted(df_raw['Year'].unique()))
        
    with col2:
        atk_sel = st.multiselect("Filter by Attack Source(s)", sorted(df_raw['Attack Source'].unique()), default=sorted(df_raw['Attack Source'].unique()))

    
    # Filter dataset based on selections
    dff = df_raw[df_raw['Year'].isin(years_sel) & df_raw['Attack Source'].isin(atk_sel)]    

    # Create heatmap for incidents by country & industry
    st.subheader("Heatmap: Country x Industry")
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


# In[ ]:




