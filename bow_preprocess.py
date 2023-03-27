#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


bow=pd.read_csv('bowling.csv')
bow.head()


# In[3]:


bow.info()


# In[4]:


bow.isnull().sum()


# In[5]:


bow.describe()


# In[6]:


sns.countplot(x=bow.PlayerName.value_counts(),data=bow)


# In[7]:


player_stats=bow[['PlayerName','Overs','Maidens','Runs','Wickets','Economy','Dream11_score']].groupby('PlayerName').sum().sort_values("Wickets",ascending=False)


# In[8]:


highest_wkts=bow['Wickets'].argmax()
bow.iloc[highest_wkts]


# In[9]:


highest_score=bow['Dream11_score'].argmax()
bow.iloc[highest_score]


# In[10]:


player_stats


# In[11]:


player_team=bow[['PlayerName','Team1']].groupby('PlayerName').sample()
player_team.set_index("PlayerName",inplace=True)
player_team
player_team.loc['Yuzvendra Chahal']


# In[12]:


player_stats=player_stats.merge(player_team,on='PlayerName')
player_stats


# In[13]:


sns.countplot(x=player_stats.head(10).Team1)


# In[14]:


player_stats.loc['Wanindu Hasaranga']


# In[15]:


team_stats=player_stats.groupby('Team1').sum().sort_values('Wickets',ascending=False)
team_stats


# In[16]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()


# In[17]:


categorical_features=bow[['PlayerName','Team1','Team2']]


# In[18]:


for column in categorical_features:
    bow[column]=label_encoder.fit_transform(bow[column])
bow.head()


# In[19]:


x=bow.drop('Dream11_score',axis=1)
x


# In[20]:


y=bow['Dream11_score']
y


# In[21]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[22]:


len(x_train),len(y_train)


# In[23]:


len(x_test),len(y_test)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


# In[25]:


models={'LinearRegressor':LinearRegression(),
        'LassoRegressor':Lasso(alpha=0.1),
        'RidgeRegressor':Ridge(alpha=1),
        'RandomForestRegressor':RandomForestRegressor(),
        'DecisionTreeRegressor':DecisionTreeRegressor(),
        'SVR':SVR()
       }

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name+' trained')


# In[26]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2score=[]
rmse=[]
mse=[]
mae=[]
#checking the scores for regressors
for name,model in models.items():
    y_pred=model.predict(x_test)
    rmse.append(mean_squared_error((y_test), (y_pred), squared=False))
    r2score.append(r2_score((y_test),(y_pred)))
    mse.append(mean_squared_error((y_test),(y_pred)))
    mae.append(mean_absolute_error((y_test),(y_pred)))


# In[27]:


Models = ['Linear Regression','Lasso Regression','Ridge Regression','Randon Forest Regression','Decision Tree Regression',
          'Support Vector Regression']
model_performance = pd.DataFrame({
    'Model':Models,
    'R2_Score':r2score,
    'Mean_squared_error':mse,
    'Root_Mean_Squared_Error':rmse,
    'Mean_Absolute_error':mae
    })
model_performance


# In[28]:


l=LinearRegression()
ls=Lasso()
rd=Ridge()
ds=DecisionTreeRegressor()  #accuracy:98
rfr = RandomForestRegressor() #accuracy:97


# In[29]:


#function to intake model and predict the values and return the accuracies

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    
    print('RMSE :',mean_squared_error((test_labels), (predictions), squared=False))
    print('R2 :',r2_score((test_labels),(predictions)))
    print('MSE :',mean_squared_error((test_labels),(predictions)))
    print('MAE :',mean_absolute_error((test_labels),(predictions)))
    
    return accuracy
#fitting the base model
ds.fit(x_train,y_train)

#Different scores from the base model before tuning
evaluate(ds,x_test,y_test)


# In[30]:


ds.predict([[86,8,2,3.0,0,32,0,10.67]])


# In[31]:


import pickle
pickle.dump(ds,open('bow_model.pkl','wb'))


# In[ ]:




