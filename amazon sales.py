#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the libraries we required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')


# In[3]:


# load the dataset
df=pd.read_csv('Amazon Sales Data.csv')
df


# In[4]:


# check the brief info of the dataset
df.head()


# In[5]:


# ckeck the bottom 5 records
df.tail()


# In[6]:


# check the basic info of the dataset
df.info()


# In[9]:


# dividing the features based on their datatypes
continuous_features=[]
categorical_features=[]
continuous_or_discrete_count=[]
for i in df.columns:
    if df[i].dtypes=='float64':
        continuous_features.append(i)
    elif df[i].dtypes=='object':
        categorical_features.append(i)
    else:
        continuous_or_discrete_count.append(i)
print('continuous_features:',continuous_features)
print('categorical_features:',categorical_features)
print('continuous_or_discrete:',continuous_or_discrete_count)


# In[10]:


# shape of the dataset
df.shape


# In[11]:


# index of the dataset
df.index


# In[12]:


# check the columns of the dataset
df.columns


# In[13]:


# check the sales channel unique values
df['Sales Channel'].unique()


# In[14]:


# sales channel value counts
df['Sales Channel'].value_counts()


# In[15]:


# check the duplicated record
df.duplicated().sum()


# no duplicates

# In[16]:


# check the null values
df.isnull().sum()


# no null values

# In[18]:


# Total profit wise top 5 countries
df.sort_values(by='Total Profit',ascending=False,ignore_index=True).head()


# In[19]:


# Profit wise regions in descending order
df.groupby('Region')['Total Profit'].sum().sort_values(ascending=False)


# In[20]:


# region wise total revenue by using group by
df.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False)


# In[21]:


# region wise value counts
df['Region'].value_counts()


# In[22]:


# countplot for the sales channel feature
plt.figure(figsize=(8,6))
sns.countplot(x='Sales Channel',data=df,edgecolor='linen',alpha=0.7,)
plt.title('Sales channel and their count')
plt.xlabel('Sales Channel')
plt.ylabel('Count')
plt.show()


# In[30]:


# check the outliers are present in the dataset by using boxplot
sns.set_theme(style="ticks")
for i in continuous_features:
    print(f'\t\t----- Boxplot of {i} -----')
    sns.boxplot(x=df[i],color=np.random.rand(4,))
    plt.show()


# Based on the boxplot there is an outliers

# In[32]:


# check the distribution of a dataset
sns.set_theme(style='dark')
for i in continuous_features:
    sns.displot(x=df[i],kde=True,color=np.random.rand(3,))
    plt.show()


# Based on the above charts its a right skewed distribution

# In[ ]:




