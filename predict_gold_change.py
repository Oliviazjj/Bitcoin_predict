#Pandas — to read the CSV file
#Numpy — perform calculations on data
#Scikit learn — build the predictive models
#Matplotlib — visualise the output


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt


# get accuracy for gold only 

df_gold = pd.read_csv("gold_price_edit.csv", delimiter = ',') # read csv file
# print("head is ", df.head(1))
rows_gold = df_gold.values.tolist()  # convert dataframe into a list
rows_gold.reverse() #reverse rows since we want latest date be the end of the list
# print("row count is", rows)


# result will be up/down based on the price of previous day
x_train_gold = [] #training dataset
y_train_gold = [] #training class set
x_test_gold = [] #testing dataset
y_test_gold = [] #testing class set
X_gold = []
Y_gold = []

print("row length i s", len(rows_gold))
for row in rows_gold:
	X_gold.append(row[2])
	X_gold.append(row[5])
	change = 0 if (float(row[1])-float(row[2])==0) else (1 if (float(row[1])-float(row[2])>0) else -1) 
	Y_gold.append(change)
print("x is ", X_gold)
print("y is ", Y_gold)
X_gold = np.array(X_gold)
Y_gold = np.array(Y_gold)
X_gold = X_gold.reshape(-1,2)
print("x shape is ", X_gold.shape)
x_train_gold, x_test_gold, y_train_gold, y_test_gold = train_test_split(X_gold,Y_gold,train_size=0.9,test_size=0.1) # 90% of data is used for training and remaining data for testing

print("x train shape is ", x_train_gold.shape)
print("x test shape is ", x_test_gold.shape)
print("y train shape is ", y_train_gold.shape)
print("y test shape is ", y_test_gold.shape)

# Convert lists into numpy arrays
x_train_gold = np.array(x_train_gold)
y_train_gold = np.array(y_train_gold)
x_test_gold = np.array(x_test_gold)
y_test_gold = np.array(y_test_gold)


# Linear Regression model
clf_lr = LinearRegression()
clf_lr.fit(x_train_gold,y_train_gold)
y_pred_lr = clf_lr.predict(x_test_gold)

# Support Vector Machine with a Radial Basis Function as kernel
clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf_svr.fit(x_train_gold,y_train_gold)
y_pred_svr = clf_svr.predict(x_test_gold)

# Random Forest Regressor
clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(x_train_gold,y_train_gold)
y_pred_rf = clf_rf.predict(x_test_gold)

# Gradient Boosting Regressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(x_train_gold,y_train_gold)
y_pred_gb = clf_gb.predict(x_test_gold)


f,(ax1,ax2) = plt.subplots(1,2,figsize=(30,10))

# Linear Regression
ax1.scatter(range(len(y_test_gold)),y_test_gold,label='data')
ax1.plot(range(len(y_test_gold)),y_pred_lr,color='green',label='LR model')
ax1.legend()

# Support Vector Machine
ax2.scatter(range(len(y_test_gold)),y_test_gold,label='data')
ax2.plot(range(len(y_test_gold)),y_pred_svr,color='orange',label='SVM-RBF model')
ax2.legend()

f1,(ax3,ax4) = plt.subplots(1,2,figsize=(30,10))

# Random Forest Regressor
ax3.scatter(range(len(y_test_gold)),y_test_gold,label='data')
ax3.plot(range(len(y_test_gold)),y_pred_rf,color='red',label='RF model')
ax3.legend()

# Gradient Boosting Regressor
ax4.scatter(range(len(y_test_gold)),y_test_gold,label='data')
ax4.plot(range(len(y_test_gold)),y_pred_gb,color='black',label='GB model')
ax4.legend()

print("Accuracy of Gold Linear Regerssion Model:",clf_lr.score(x_test_gold,y_test_gold))
print("Accuracy of Gold SVM-RBF Model:",clf_svr.score(x_test_gold,y_test_gold))
print("Accuracy of Gold Random Forest Model:",clf_rf.score(x_test_gold,y_test_gold))
print("Accuracy of Gold Gradient Boosting Model:",clf_gb.score(x_test_gold,y_test_gold))