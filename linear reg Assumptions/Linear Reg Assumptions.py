#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:


dataset = pd.read_csv('ad.csv')


# In[3]:


dataset


# In[4]:


x = dataset[['TV','Radio','Newspaper']]


# In[5]:


x


# In[6]:


y = dataset['Sales']


# In[7]:


y


# In[10]:


import statsmodels.api as sm


# In[13]:


X_constant = sm.add_constant(x)


# In[14]:


X_constant


# In[15]:


model = sm.OLS(y,X_constant).fit()


# In[16]:


model.summary()


# In[18]:


#Autocorrelation
#Durbin-Waiston            2.804 little corelation


# In[19]:


#Normality


# In[21]:


sns.distplot(dataset['TV'])
plt.show()


# In[22]:


sns.distplot(dataset['Radio'])
plt.show()


# In[25]:


sns.distplot(dataset['Newspaper'])
plt.show


# In[27]:


sns.distplot(dataset['Sales'])
plt.show()


# In[28]:


q_q_plot = sm.qqplot(model.resid,fit=True)


# In[29]:


#linearity


# In[35]:


plt.scatter(dataset['TV'],dataset['Sales'])
plt.xlabel('TV')
plt.ylabel('SALES')
plt.show()


# In[39]:


plt.scatter(dataset['Radio'],dataset['Sales'])
plt.xlabel('Radio')
plt.ylabel('SALES')
plt.show()


# In[40]:


plt.scatter(dataset['Newspaper'],dataset['Sales'])
plt.xlabel('Newspaper')
plt.ylabel('SALES')
plt.show()


# In[41]:


#Multilinearity


# In[42]:


indep = dataset[['TV','Radio','Newspaper']]


# In[43]:


indep


# In[44]:


indep.corr()


# In[45]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[46]:


indep


# In[47]:


indep.values


# In[48]:


vif = [variance_inflation_factor(indep.values,i) for i in range(3)]


# In[50]:


vif


# In[ ]:




