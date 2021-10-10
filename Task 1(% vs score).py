#!/usr/bin/env python
# coding: utf-8

# Prediciton using supervised learning by AREEB AHMAD

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#importing all the libraries


# In[5]:


link = "http://bit.ly/w-data"
dataset = pd.read_csv(link)
#storing data from link in dataset 


# In[6]:


dataset.head()
#printing first five rows of data


# In[7]:


#now we will plot a graph from the given data using matplotlib


# In[27]:


dataset.plot(kind='scatter', x='Hours', y='Scores', color = 'g')


# In[43]:


x=dataset.iloc[:,0:1]
y=dataset.iloc[:,1:2]
#dividing values in labels and targets as X and Y respectively


# In[48]:


from sklearn.linear_model import LinearRegression
#importing linear regression model


# In[49]:


lin_reg=LinearRegression()
#assigning object lin_reg for LinearRegression class


# In[50]:


lin_reg.fit(x,y)
#fitting model for x and y 


# In[53]:


yp=lin_reg.predict(x)
#predicting values for input x


# In[57]:


from sklearn.metrics import r2_score
r2_score(y,yp)
#checking accuracy of model


# In[58]:


#95 % accurcy


# In[59]:


x=dataset.iloc[:,0:1]
y=dataset.iloc[:,1:2]
dataset.plot(kind='scatter', x='Hours', y='Scores', color='b')
plt.plot(x,lin_reg.coef_[0]*x+lin_reg.intercept_,color='r')
plt.grid()
plt.show
#plotting the regression line for predictions


# In[67]:


Hours=9.25
prediction=lin_reg.predict([[Hours]])
print(f"number of hours={Hours}")
print(f"Predicted Score={prediction}")


# In[ ]:




