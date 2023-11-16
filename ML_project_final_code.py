#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 
# # Data Cleaning

# In[3]:


df = pd.read_csv('movie.csv')
df.head()


# In[4]:


df = df.drop(['Unnamed: 0','Name of movie','Year of relase','Description','Director','Star'],axis=1)


# In[5]:


df.head()


# In[6]:


df[df['Metascore'] == '^^^^^^']
df['Metascore'] = df['Metascore'].replace('^^^^^^','0')
df['Gross collection'] = df['Gross collection'].replace('*****','0')


# In[7]:


df.head()


# In[8]:


cols = df.columns
cols
for col in cols:
    df[col] = df[col].astype('string')


# In[9]:


data = df.copy()
data.head()


# In[10]:


j = 0
for i in data['Gross collection']:
    if i == '0':
        data['Gross collection'][j] = i
        j = j+1
    else:
        n = len(i)
        z = i
        i = i[1:n-1]
        data['Gross collection'][j] = i
        j = j+1
    #df[i] = df[1:n-1]
data.head()


# In[11]:


j = 0
for i in data['Votes']:
    n = len(i)
    z = i
    i = i.replace(',','')
    data['Votes'][j] = i
    j = j+1
    #df[i] = df[1:n-1]
data.head()


# In[12]:


data['Gross collection'] = data['Gross collection'].astype('float')


# In[13]:


cols = data.columns
cols
for col in cols:
    data[col] = data[col].astype('float')
data.corr()


# In[14]:


cols = data.columns
cols
for col in cols:
    data[col] = data[col].replace(0,np.median(data[col]))
data.head()


# In[15]:


new_data = data[data['Gross collection']>5]
new_data = new_data.drop('Metascore',axis=1)
#new_data = new_data[new_data['Metascore']>50]
new_data.shape


# In[16]:


x = data.drop('Gross collection',axis=1)
y = data['Gross collection']


# In[17]:


info = data.describe()
sns.heatmap(info,annot=True,fmt='.2f')


# In[18]:


cols = x.columns
n = len(cols)
cols = cols[:n]
for i in cols:
    x[i] =  (x[i]- min(x[i]))/(max(x[i]-min(x[i])))


# In[19]:


y


# # Visulization

# In[20]:


plt.figure(figsize=(10,10))
#df.columns
cols = x.columns
i = 0
for col in cols:
    plt.subplot(2,2,i+1)
    sns.scatterplot(x=x[col],y=y,color='red')
    i = i+1


# In[21]:


plt.figure(figsize=(10,10))
#df.columns
cols = x.columns
i = 0
for col in cols:
    plt.subplot(2,2,i+1)
    sns.distplot(x[col])
    i = i+1


# In[22]:


info = x.describe()
sns.heatmap(info,annot=True,fmt='.2f')


# In[23]:


info = x.corr()
sns.heatmap(info,annot=True,fmt='.2f')


# In[24]:


x.shape


# # Model Creation and Hyperparameter Tuning

# In[25]:


x_train = x[:720]
y_train = y[:720]
x_test = x[721:800]
y_test = y[721:800]


# In[26]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model = model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[27]:


ypred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,ypred))
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,ypred))
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
ypred1 = ypred
from sklearn.metrics import r2_score
print("R2 score:",r2_score(y_test, ypred))


# In[28]:


sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
x = np.arange(0,np.max(y_test),0.1)
y = np.arange(0,np.max(y_test),0.1)
sns.lineplot(x=x,y=y,label='Actual')
plt.xlabel("Actual Collection")
plt.ylabel("Predicted Collection")
plt.title('LinearRegression')
plt.show()


# In[29]:


from sklearn.model_selection import GridSearchCV


# In[30]:


n_estimators=[100,200,300]
max_depth=[int(x) for x in np.linspace(10,200,50)]
#max_depth = int(max_depth)

random_grid = {
    'n_estimators':n_estimators,
    'max_depth':max_depth,
}


# In[31]:


from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor()
forest_params = random_grid
clf = GridSearchCV(rfc, forest_params)
clf.fit(x_train, y_train)
clf.best_params_


# In[32]:


clf.best_params_


# In[33]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=37, n_estimators=100)
model = model.fit(x_train,y_train)
ypred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,ypred))
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,ypred))
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
ypred2 = ypred
from sklearn.metrics import r2_score
print("R2 score:",r2_score(y_test, ypred))


# In[34]:


sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
x = np.arange(0,np.max(y_test),0.1)
y = np.arange(0,np.max(y_test),0.1)
sns.lineplot(x=x,y=y,label='Actual')
plt.xlabel("Actual Collection")
plt.ylabel("Predicted Collection")
plt.title('Random Forest')
plt.show()


# In[ ]:





# In[35]:


n_estimators=[100,200,300]
learning_rate=np.arange(0.1,1,0.1)
#max_depth = int(max_depth)
loss = ['square','linear']

random_grid = {
    'n_estimators':n_estimators,
    'loss':loss,
    'learning_rate':learning_rate
    
}


# In[36]:


from sklearn.ensemble import AdaBoostRegressor
rfc = AdaBoostRegressor()
forest_params = random_grid
clf = GridSearchCV(rfc, forest_params)
clf.fit(x_train, y_train)
clf.best_estimator_


# In[37]:


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor(learning_rate=0.1, loss='square', n_estimators=100)
model = model.fit(x_train,y_train)
ypred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,ypred))
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,ypred))
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
ypred3 = ypred
from sklearn.metrics import r2_score
print("R2 score:",r2_score(y_test, ypred))


# In[38]:


sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
x = np.arange(0,np.max(y_test),0.1)
y = np.arange(0,np.max(y_test),0.1)
sns.lineplot(x=x,y=y,label='Actual')
plt.xlabel("Actual Collection")
plt.ylabel("Predicted Collection")
plt.title('Adaboost Regressor')
plt.show()


# In[ ]:





# In[39]:


n_estimators=[100,200,300]
max_depth=[int(x) for x in np.linspace(10,200,50)]
#max_depth = int(max_depth)
learning_rate = np.arange(0.1,1,0.1)
max_leaves = np.arange(10,60,10)
random_grid = {
    'n_estimators':n_estimators,
    'max_depth':max_depth,
    'max_leaves':max_leaves
}


# In[40]:


from xgboost import XGBRegressor
rfc = XGBRegressor()
forest_params = random_grid
clf = GridSearchCV(rfc, forest_params)
clf.fit(x_train, y_train)
clf.best_estimator_


# In[41]:


from xgboost import XGBRegressor
model = XGBRegressor(max_delta_step=0, max_depth=10, max_leaves=10, min_child_weight=1)
model = model.fit(x_train,y_train)
ypred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,ypred))
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,ypred))
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
ypred4 = ypred
from sklearn.metrics import r2_score
print("R2 score:",r2_score(y_test, ypred))


# In[42]:


sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
x = np.arange(0,np.max(y_test),0.1)
y = np.arange(0,np.max(y_test),0.1)
sns.lineplot(x=x,y=y,label='Actual')
plt.xlabel("Actual Collection")
plt.ylabel("Predicted Collection")
plt.title('XgBoost Regressor')
plt.show()


# In[43]:


report1 = pd.DataFrame()
report1['Actual'] = y_test
report1['Predicted'] = ypred1
report1.head()


# In[44]:


report2 = pd.DataFrame()
report2['Actual'] = y_test
report2['Predicted'] = ypred2
report2.head()


# In[45]:


report3 = pd.DataFrame()
report3['Actual'] = y_test
report3['Predicted'] = ypred3
report3.head()


# In[46]:


report4 = pd.DataFrame()
report4['Actual'] = y_test
report4['Predicted'] = ypred4
report4.head()


# In[ ]:





# In[ ]:





# In[ ]:




