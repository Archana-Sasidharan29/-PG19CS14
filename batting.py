# %% [markdown]
# 

# %%
import pandas as pd
import numpy as np
import seaborn as sns

# %%
bat=pd.read_csv('batting.csv')
bat.head()

# %%
bat.info()

# %%
bat.isnull().sum()

# %%
bat.describe()

# %%
sns.countplot(x=bat.Player.value_counts(),data=bat)

# %%
player_stats=bat[['Player','Runs','Balls','4s','6s','Ducks','SR','Dream11_score']].groupby('Player').sum().sort_values("Runs",ascending=False)

# %%
highest_runs=bat['Runs'].argmax()
bat.iloc[highest_runs]

# %%
highest_score=bat['Dream11_score'].argmax()
bat.iloc[highest_score]

# %%
player_stats

# %%
player_team=bat[['Player','Team1']].groupby('Player').sample()
player_team.set_index("Player",inplace=True)
player_team
player_team.loc['Jos Buttler']

# %%
player_stats=player_stats.merge(player_team,on='Player')
player_stats

# %%
sns.countplot(x=player_stats.head(10).Team1)

# %%
player_stats.loc['Virat Kohli']

# %%
team_stats=player_stats.groupby('Team1').sum().sort_values('Runs',ascending=False)
team_stats

# %%
#Label encoding
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()

# %%
categorical_features=bat[['Player','Team1','Team2']]

# %%
for column in categorical_features:
    bat[column]=label_encoder.fit_transform(bat[column])
bat.head()

# %%
#splitting dependant and independant variables
x=bat.drop('Dream11_score',axis=1)
x

# %%
y=bat['Dream11_score']
y

# %%
#splitting as training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# %%
#Fitting with the models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# %%
#fitting models
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

# %%
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

# %%
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

# %%
l=LinearRegression()
ls=Lasso()
rd=Ridge()
rfr = RandomForestRegressor()

# %%
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
rfr.fit(x_train,y_train)

#Different scores from the base model before tuning
evaluate(rfr,x_test,y_test)

# %%
rfr.predict([[40,2,8,34,30,3,1,0,113.33]])

# %%
import pickle
pickle.dump(rfr,open('bat_model.pkl','wb'))


