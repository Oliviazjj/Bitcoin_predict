
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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time


start_time = time.time()
# get training and testing data for bitcoin
df_bitcoin = pd.read_csv("bitcoin_price.csv", delimiter = ',') # read csv file
df_bitcoin["Open"] = (df_bitcoin["Open"]-df_bitcoin["Open"].min()) / (df_bitcoin["Open"].max()-df_bitcoin["Open"].min())
# print("df_bitcoin[\"Open\"] is ", df_bitcoin["Open"])
df_bitcoin["Close"] = (df_bitcoin["Close"]-df_bitcoin["Close"].min()) / (df_bitcoin["Close"].max()-df_bitcoin["Close"].min())
# print("df_bitcoin[\"Close\"] is ", df_bitcoin["Close"])
rows_bitcoin = df_bitcoin.values.tolist()  # convert dataframe into a list
rows_bitcoin.reverse() #reverse rows since we want latest date be the end of the list

# result will be up/down based on the price of previous day
x_train_bitcoin = [] #training dataset
y_train_bitcoin = [] #training class set
x_test_bitcoin = [] #testing dataset
y_test_bitcoin = [] #testing class set
X_bitcoin = []
Y_bitcoin = []

print("row length i s", len(rows_bitcoin))
for row in rows_bitcoin:
	X_bitcoin.append(row[2])
	X_bitcoin.append(row[6])
	Y_bitcoin.append(row[5])
X_bitcoin = np.array(X_bitcoin)
Y_bitcoin = np.array(Y_bitcoin)
X_bitcoin = X_bitcoin.reshape(-1,2)
print("x shape is ", X_bitcoin.shape)
x_train_bitcoin, x_test_bitcoin, y_train_bitcoin, y_test_bitcoin = train_test_split(X_bitcoin,Y_bitcoin,train_size=0.9,test_size=0.1) # 90% of data is used for training and remaining data for testing

print("x train shape is ", x_train_bitcoin.shape)
print("x test shape is ", x_test_bitcoin.shape)
print("y train shape is ", y_train_bitcoin.shape)
print("y test shape is ", y_test_bitcoin.shape)

# Convert lists into numpy arrays
x_train_bitcoin = np.array(x_train_bitcoin)
y_train_bitcoin = np.array(y_train_bitcoin)
x_test_bitcoin = np.array(x_test_bitcoin)
y_test_bitcoin = np.array(y_test_bitcoin)


















# get training and testing data for gold
df_gold = pd.read_csv("gold_price_edit.csv", delimiter = ',') # read csv file
# print("head is ", df.head(1))
df_gold["Open"] = (df_gold["Open"]-df_gold["Open"].min()) / (df_gold["Open"].max()-df_gold["Open"].min())
print("df_gold[\"Open\"] is ", df_gold["Open"])
df_gold["Price"] = (df_gold["Price"]-df_gold["Price"].min()) / (df_gold["Price"].max()-df_gold["Price"].min())
print("df_gold[\"Price\"] is ", df_gold["Price"])
# print("csv value is ", df.values)
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
	Y_gold.append(row[1])
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

















# # get accuracy from different models by using bitcoin as training and gold as testing data
# print("\n\n\n")
# print("-------accuracy by using bitcoin as training data and gold as testing data------")

# # Linear Regression model
# clf_lr = LinearRegression()
# clf_lr.fit(x_train_bitcoin,y_train_bitcoin)
# y_pred_lr = clf_lr.predict(x_test_gold)

# # Support Vector Machine with a Radial Basis Function as kernel
# clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
# clf_svr.fit(x_train_bitcoin,y_train_bitcoin)
# y_pred_svr = clf_svr.predict(x_test_gold)

# # Random Forest Regressor
# clf_rf = RandomForestRegressor(n_estimators=100)
# clf_rf.fit(x_train_bitcoin,y_train_bitcoin)
# y_pred_rf = clf_rf.predict(x_test_gold)

# # Gradient Boosting Regressor
# clf_gb = GradientBoostingRegressor(n_estimators=200)
# clf_gb.fit(x_train_bitcoin,y_train_bitcoin)
# y_pred_gb = clf_gb.predict(x_test_gold)


# f,(ax1,ax2) = plt.subplots(1,2,figsize=(30,10))

# # Linear Regression
# ax1.scatter(range(len(y_test_gold)),y_test_gold,label='data')
# ax1.plot(range(len(y_test_gold)),y_pred_lr,color='green',label='LR model')
# ax1.legend()

# # Support Vector Machine
# ax2.scatter(range(len(y_test_gold)),y_test_gold,label='data')
# ax2.plot(range(len(y_test_gold)),y_pred_svr,color='orange',label='SVM-RBF model')
# ax2.legend()

# f1,(ax3,ax4) = plt.subplots(1,2,figsize=(30,10))

# # Random Forest Regressor
# ax3.scatter(range(len(y_test_gold)),y_test_gold,label='data')
# ax3.plot(range(len(y_test_gold)),y_pred_rf,color='red',label='RF model')
# ax3.legend()

# # Gradient Boosting Regressor
# ax4.scatter(range(len(y_test_gold)),y_test_gold,label='data')
# ax4.plot(range(len(y_test_gold)),y_pred_gb,color='black',label='GB model')
# ax4.legend()

# print("Accuracy of BTC-GOLD Linear Regerssion Model:",clf_lr.score(x_test_gold,y_test_gold))
# print("Accuracy of BTC-GOLD SVM-RBF Model:",clf_svr.score(x_test_gold,y_test_gold))
# print("Accuracy of BTC_GOLD Random Forest Model:",clf_rf.score(x_test_gold,y_test_gold))
# print("Accuracy of BTC-GOLD Gradient Boosting Model:",clf_gb.score(x_test_gold,y_test_gold))





# get accuracy from different models by using gold as training data and bitcoin as testing data 
print("\n\n\n")
print("-------accuracy by using gold as training data and bitcoin as testing data------")
# Linear Regression model
clf_lr = LinearRegression()
clf_lr.fit(X_gold,Y_gold)
y_pred_lr = clf_lr.predict(X_bitcoin)

# # Support Vector Machine with a Radial Basis Function as kernel
# clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
# clf_svr.fit(X_gold,Y_gold)
# y_pred_svr = clf_svr.predict(X_bitcoin)

# Random Forest Regressor
clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(X_gold,Y_gold)
y_pred_rf = clf_rf.predict(X_bitcoin)

# Gradient Boosting Regressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_gold,Y_gold)
y_pred_gb = clf_gb.predict(X_bitcoin)


f,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(30,10))

# Linear Regression
ax1.scatter(range(len(Y_bitcoin)),Y_bitcoin,label='data')
ax1.plot(range(len(Y_bitcoin)),y_pred_lr,color='green',label='LR model')
ax1.legend()

# # Support Vector Machine
# ax2.scatter(range(len(y_test_bitcoin)),y_test_bitcoin,label='data')
# ax2.plot(range(len(y_test_gold)),y_pred_svr,color='orange',label='SVM-RBF model')
# ax2.legend()

# f1,(ax3,ax4) = plt.subplots(1,2,figsize=(30,10))

# Random Forest Regressor
ax2.scatter(range(len(Y_bitcoin)),Y_bitcoin,label='data')
ax2.plot(range(len(Y_bitcoin)),y_pred_rf,color='red',label='RF model')
ax2.legend()

# Gradient Boosting Regressor
ax3.scatter(range(len(Y_bitcoin)),Y_bitcoin,label='data')
ax3.plot(range(len(Y_bitcoin)),y_pred_gb,color='black',label='GB model')
ax3.legend()
plt.show(block=True)

print("Accuracy of GOLD-BTC Linear Regerssion Model:",clf_lr.score(X_bitcoin,Y_bitcoin))
# print("Accuracy of GOLD-BTC SVM-RBF Model:",clf_svr.score(X_bitcoin,Y_bitcoin))
print("Accuracy of GOLD-BTC Random Forest Model:",clf_rf.score(X_bitcoin,Y_bitcoin))
print("Accuracy of GOLD-BTC Gradient Boosting Model:",clf_gb.score(X_bitcoin,Y_bitcoin))

print("--- %s seconds ---" % (time.time() - start_time))
