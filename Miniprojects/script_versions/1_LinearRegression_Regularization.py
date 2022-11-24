# # Mini-Project: Linear Regression with Regularization

""" Problem Statement

Predict the bike-sharing counts per hour based on features including weather, day, time, humidity, wind speed, season e.t.c.
"""
# ### Downloading the dataset

get_ipython().system('wget -qq https://cdn.iisc.talentsprint.com/CDS/MiniProjects/Bike_Sharing_Dataset.zip')
get_ipython().system('unzip Bike_Sharing_Dataset.zip')

# Loading the Required Packages
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score


# ### Data Loading

# Reading Our Dataset
bikeshare = pd.read_csv('hour.csv')



# * Identify continuous features
# 
# * Identify categorical features
# 
# * Apply scaling on continuous features 
# 
# * Apply one-hot encoding on categorical features
# 
# * Create features by concatenating all one hot encoded features and scaled features except target variables
# 
# * Find the coefficients of the features using normal equation and find the cost (error)
# 
# * Apply batch gradient descent technique by taking one target variable (cnt) and find the best coefficients
# 
# * Apply SGD Regressor using sklearn
# 
# * Apply linear regression using sklearn by taking two target variables (casual, registered)
# 
# * Apply Lasso, Ridge, Elasticnet Regression

# ### EDA &  Visualization

# #### Visualize the hour (hr) column with an appropriate plot and find the busy hours of bike sharing

bikeshare.groupby('hr').sum('cnt')['cnt'].plot.bar()


# #### Visualize the distribution of count, casual and registered variables


# distribution of casual
sns.distplot(bikeshare.casual);


# distribution of registered
sns.distplot(bikeshare.registered);


# distribution of count
sns.distplot(bikeshare.cnt);


# #### Describe the relation of weekday, holiday and working day

# Working days from 1-5 (mon-fri)
bikeshare[bikeshare.workingday==1].weekday.unique()


# Holiday possible on working days
bikeshare[bikeshare.holiday==1].weekday.unique()

# Not a holiday, not a working day (Sun, Sat)
bikeshare[(bikeshare.holiday==0) & (bikeshare.workingday==0)].weekday.unique()


# #### Visualize the monthly wise count of both casual and registered for the year 2011 and 2012 separately.
# 
# Hint: Stacked barchart

# stacked bar chart for year 2011
bikeshare[bikeshare.yr==0].groupby('mnth').sum(['casual','registered'])[['casual','registered']].plot.bar(stacked=True);
plt.title("Casual and Registered in 2011")
plt.show()

# stacked bar chart for year 2012
bikeshare[bikeshare.yr==1].groupby('mnth').sum(['casual','registered'])[['casual','registered']].plot.bar(stacked=True)
plt.title("Casual and Registered in 2012")
plt.show()


# #### Analyze the correlation between features with heatmap

sns.heatmap(bikeshare.iloc[:,:].corr(), cmap='RdBu')


# #### Visualize the box plot of casual and registered variables to check the outliers


fig, axes = plt.subplots(nrows=1,ncols=2)
sns.boxplot(data=bikeshare,y="casual",orient="v",ax=axes[0])
sns.boxplot(data=bikeshare,y="registered",orient="v",ax=axes[1])
plt.show()


# ### Pre-processing and Data Engineering

# #### Drop unwanted columns


bikeshare1 = bikeshare.drop(['instant', 'dteday'], axis = 1)
bikeshare1.shape


# Identifying categorical and continuous variables
cont_features = ['temp','atemp','hum','windspeed'] #,'casual','registered','cnt']
categorical_features = ['season', 'yr', 'mnth','hr','holiday','weekday','weathersit']


# #### Apply scaling on the continuous variables
# 
# **Note:** Include the target variables

std_scaler = StandardScaler()
scaled_data = pd.DataFrame(std_scaler.fit_transform(bikeshare1[cont_features]), columns = cont_features)
scaled_data.shape

# scaled features + categorical in one dataframe
scaled_data
for i in categorical_features:
    scaled_data[i] = bikeshare1[i].values
scaled_data.head(2)


# #### Apply one-hot encode on the categorical data
# 
# Hint: `sklearn.preprocessing.OneHotEncoder`

onehot = OneHotEncoder()
onehot_encoded = onehot.fit_transform(scaled_data[categorical_features]).toarray()
onehot_encoded.shape

onehot_encoded


# #### Specify features and targets after applying scaling and one-hot encoding

features = np.concatenate((scaled_data[['temp','atemp','hum','windspeed']].values, onehot_encoded), axis=1)
#features = scaled_data.values
features.shape

scaled_target = bikeshare1[['casual','registered','cnt']]
scaled_target.shape


# ### Implement the linear regression by finding the coefficients using below approaches (3 points)
# 
# * Find the coefficients using Normal equation and find the error
# 
# * Implement batch gradient descent
# 
# * SGD Regressor from sklearn

# #### Select the features and target and split the dataset
# 
# As there are 3 target variables, choose the count (`cnt`) variable.

target1 = bikeshare1[['cnt']]


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(features, target1)
x_train.shape, y_train.shape


# #### Implementation using Normal Equation
# 
# $\theta = (X^T X)^{-1} . (X^T Y)$
# 
# $θ$ is the hypothesis parameter that define it the best
# 
# $X$ is the input feature value of each instance
# 
# $Y$ is Output value of each instance

# In[27]:


y = y_train.values
# Adding ones to X
X = np.append(np.ones((x_train.shape[0],1)),x_train, axis=1)
X.shape


# In[28]:


# X_transpose * X
X_t = np.transpose(X)
X_Xt_dot = X_t.dot(X)

# inverse of (X * X_transpose)
temp1 = np.linalg.inv(X_Xt_dot)
temp1.shape


# In[29]:


# X_transpose * Y
temp2 = X_t.dot(y)
# Inverse of (X_transpose * X) * (X_transpose * Y)
coefs = temp1.dot(temp2)
coefs.shape


# In[30]:


# Above steps in one line
y = y_train.values
X_b = np.concatenate((np.ones((x_train.shape[0], 1)), x_train),axis=1)
theta_star = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_star.shape


# In[59]:


ypredict = X_b.dot(theta_star.ravel())
mean_squared_error(ypredict, y_train.values)


# In[32]:


def Calc_MSE(X, y_test, coefficients):

    X = np.append(np.ones((X.shape[0],1)),X, axis=1)
    score = mean_squared_error(y_test.values, X.dot(coefficients))
    return score


# In[33]:


# objective(X,y, coefs)
train_error = Calc_MSE(x_train, y_train, coefs)
test_error = Calc_MSE(x_test, y_test, coefs)
train_error, test_error


# In[58]:


from scipy.linalg import lstsq

x_train, x_test, y_train, y_test = train_test_split(features, target1)
y = y_train.values
X = np.append(np.ones((x_train.shape[0],1)),x_train, axis=1)
p, res, rnk, s = lstsq(X, y)
print(np.sum((X.dot(p)-y)**2)/X.shape[0])


# #### Implementing Linear regression using batch gradient descent
# 
# Initialize the random coefficients and optimize the coefficients in the iterative process by calculating cost and finding the gradient.
# 
# Hint: [link](https://medium.com/@lope.ai/multivariate-linear-regression-from-scratch-in-python-5c4f219be6a)

# In[34]:


X = x_train
y = y_train
# Adding ones to X
X = np.append(np.ones((X.shape[0],1)),X, axis=1)
X.shape


# In[35]:


def cost_function(X, Y, B):
  return mean_squared_error(Y, X.dot(B))


# In[36]:


def batch_gradient_descent(X, Y, B, alpha, iterations):
  cost_history = [0] * iterations
  m = len(Y)
  for iteration in range(iterations):
    #print(iteration)
    h = X.dot(B)
    loss = h - Y #change the variable name
    gradient = X.T.dot(loss) / m
    B = B - alpha * gradient
    cost = cost_function(X, Y, B)
    cost_history[iteration] = cost
  return B, cost_history

B = np.random.randn(X.shape[1])
alpha = 0.005
iter_ = 50000
newB, cost_history = batch_gradient_descent(X, y.values.ravel(), B, alpha, iter_)
newB, cost_history[-1]


# In[37]:


# test error
X_test = np.append(np.ones((x_test.shape[0],1)),x_test, axis=1)
cost_function(X_test,y_test, newB)


# #### SGD Regressor
# 
# Use the SGD regressor from sklearn with one target variable and find the error
# 
# Hint: [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)

# In[38]:


sgd = linear_model.SGDRegressor()
sgd = sgd.fit(x_train, y_train)
print("score is ",sgd.score(x_test, y_test))
mean_squared_error(sgd.predict(x_test), y_test )


# ### Linear regression using sklearn
# 
# Implement the linear regression model using sklearn with two variables in target (casual, registered)
# 
# Hint: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

# #### Select the features  and split the data into train and test

# In[39]:


target2 = bikeshare1[['casual','registered']]


# In[40]:


xtrain, xtest, ytrain, ytest = train_test_split(features, target1)
xtrain.shape, ytrain.shape


# In[41]:


regr_linear = linear_model.LinearRegression()
regr_linear.fit(xtrain, ytrain)
predicted = regr_linear.predict(xtest)


# #### Calculate the mean squared error of the actual and predicted data

# In[42]:


mse_linear = mean_squared_error(ytest, predicted, multioutput = 'uniform_average')
mse_linear


# #### Calculate the $R^2$ (coefficient of determination) of the actual and predicted data

# In[43]:


r2_score(ytrain, regr_linear.predict(xtrain)), r2_score(ytest, predicted)


# #### summarize the importance of features and create a bar chart
# 
# Prediction is the weighted sum of the input values e.g. linear regression, and extensions that add regularization, such as ridge regression and the elastic net find a set of coefficients to use in the weighted sum to make a prediction. These coefficients can be used directly as a crude type of feature importance score.
# 
# This assumes that the input variables have the same scale or have been scaled prior to fitting a model.

# In[44]:


# coefficients of casual variable
importance_casual = regr_linear.coef_[0,]
plt.bar([x for x in range(len(importance_casual))], importance_casual)
plt.xticks(range(59))
plt.show()


# In[45]:


# coefficients of registered variable
# importance_registered = regr_linear.coef_[1,]
# plt.bar([x for x in range(len(importance_registered))], importance_registered)
# plt.xticks(range(59))
# plt.show()


# ### Regularization methods

# #### Apply lasso regression
# 
# * Apply Lasso regression with different alpha values given below and find the best alpha that gives the least error.
# * Calculate the metrics for the actual and predicted
# 
# Hint: [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

# In[46]:


# setting up alpha
alpha = [0.0001, 0.001,0.01, 0.1, 1, 10, 100]


# In[47]:


for a in alpha:
    regr_lasso = linear_model.Lasso(alpha = a)
    regr_lasso.fit(xtrain, ytrain)
    mse_lasso_sk = mean_squared_error(ytest, regr_lasso.predict(xtest), multioutput = 'uniform_average')
    print(a, "=====",mse_lasso_sk) 


# In[48]:


# with best alpha chosen from above
regr_lasso = linear_model.Lasso(alpha = 0.0001)
regr_lasso.fit(xtrain, ytrain)
mse_lasso_sk = mean_squared_error(ytest, regr_lasso.predict(xtest), multioutput = 'uniform_average')
print("Lasso MSE:",mse_lasso_sk)
print("Lasso r2_score",r2_score(ytrain, regr_lasso.predict(xtrain)))


# #### Apply ridge regression
# 
# * Apply Lasso regression with different alpha values given and find the best alpha that gives the least error.
# * Calculate the metrics for the actual and predicted
# 
# Hint: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

# In[49]:


for a in alpha:
    regr_ridge = linear_model.Ridge(alpha = a)
    regr_ridge.fit(xtrain, ytrain)
    mse_ridge_sk = mean_squared_error(ytest, regr_ridge.predict(xtest), multioutput = 'uniform_average')
    print(a, "=====",mse_ridge_sk)


# In[50]:


# with best alpha chosen from above
regr_ridge = linear_model.Ridge(alpha = 0.001)
regr_ridge.fit(xtrain, ytrain)
mse_ridge_sk = mean_squared_error(ytest, regr_ridge.predict(xtest), multioutput = 'uniform_average')
print("Ridge MSE:",mse_ridge_sk)
print("Ridge r2_score:",r2_score(ytrain, regr_ridge.predict(xtrain)))


# #### Apply elasticnet regression
# 
# * Apply elasticnet regression with different alpha values given and find the best alpha that gives the least error.
# * Calculate the metrics for the actual and predicted
# 
# Hint: [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

# In[51]:


for a in alpha:
    elasticnet_regr = linear_model.ElasticNet(alpha=a)
    elasticnet_regr.fit(xtrain, ytrain)
    mse_elatic_sk = mean_squared_error(ytest, elasticnet_regr.predict(xtest), multioutput = 'uniform_average')
    print(a,"====",mse_elatic_sk)


# In[52]:


# Elasticnet
elasticnet_regr = linear_model.ElasticNet(alpha=0.01)
elasticnet_regr.fit(xtrain, ytrain)
mse_elatic_sk = mean_squared_error(ytest, elasticnet_regr.predict(xtest), multioutput = 'uniform_average')
print("Elasticnet MSE:",mse_elatic_sk)
print("Elasticnet r2_score",r2_score(ytrain, elasticnet_regr.predict(xtrain)))


# * **Use the two variables (`Casual, Registered`) in target and find the error by implementing Linear Regression model from sklearn**
# * Describe your interpretation of the methods that are used to implement linear regression covered in this mini project.
# * Comment on performance of the algorithms/methods used.
# * Comment about the nature of the data and fitment of linear regression for this data.
# * Can you perform a non linear curve fitting using linear regression? If yes, How?
# 
