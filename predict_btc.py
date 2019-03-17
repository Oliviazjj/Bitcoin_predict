
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


df = pd.read_csv("gemini_BTCUSD_1hr.csv", delimiter = ',', skiprows=[0]) # read csv file
# print("head is ", df.head(1))
# print("csv value is ", df.values)
rows = df.values.tolist()  # convert dataframe into a list
rows.reverse() #reverse rows since we want latest date be the end of the list
# print("row count is", rows)



# result will be up/down based on the price of previous day
x_train = [] #training dataset
y_train = [] #training class set
x_test = [] #testing dataset
y_test = [] #testing class set
X = []
Y = []
for row in rows[:-2]:
	# time = row[1].replace(':', '-')
	# time = time.replace(' ', '-')
	# time_list = int(''.join(time.split('-')))
	time_list = int(''.join(row[0].split('-')))
	# print("time list is ", time_list)
	X.append(time_list)
	Y.append(row[5])
x_train, x_test, y_train, y_test = train_test_split(X,Y,train_size=0.9,test_size=0.1) # 90% of data is used for training and remaining data for testing

# Convert lists into numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# reshape the values as we have only one input feature
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)


# Linear Regression model
clf_lr = LinearRegression()
clf_lr.fit(x_train,y_train)
y_pred_lr = clf_lr.predict(x_test)

# Support Vector Machine with a Radial Basis Function as kernel
clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf_svr.fit(x_train,y_train)
y_pred_svr = clf_svr.predict(x_test)

# Random Forest Regressor
clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(x_train,y_train)
y_pred_rf = clf_rf.predict(x_test)

# Gradient Boosting Regressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(x_train,y_train)
y_pred_gb = clf_gb.predict(x_test)


f,(ax1,ax2) = plt.subplots(1,2,figsize=(30,10))

# Linear Regression
ax1.scatter(range(len(y_test)),y_test,label='data')
ax1.plot(range(len(y_test)),y_pred_lr,color='green',label='LR model')
ax1.legend()

# Support Vector Machine
ax2.scatter(range(len(y_test)),y_test,label='data')
ax2.plot(range(len(y_test)),y_pred_svr,color='orange',label='SVM-RBF model')
ax2.legend()

f1,(ax3,ax4) = plt.subplots(1,2,figsize=(30,10))

# Random Forest Regressor
ax3.scatter(range(len(y_test)),y_test,label='data')
ax3.plot(range(len(y_test)),y_pred_rf,color='red',label='RF model')
ax3.legend()

# Gradient Boosting Regressor
ax4.scatter(range(len(y_test)),y_test,label='data')
ax4.plot(range(len(y_test)),y_pred_gb,color='black',label='GB model')
ax4.legend()

print("Accuracy of Linear Regerssion Model:",clf_lr.score(x_test,y_test))
print("Accuracy of SVM-RBF Model:",clf_svr.score(x_test,y_test))
print("Accuracy of Random Forest Model:",clf_rf.score(x_test,y_test))
print("Accuracy of Gradient Boosting Model:",clf_gb.score(x_test,y_test))

