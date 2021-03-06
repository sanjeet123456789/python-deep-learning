#recurrent Neural Network

#part-1 Data data Processing
import numpy an np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
training_set=pd.read_csv('Google_Stock_Price_Train.csv')
training_set=trining_set.iloc[:,1:2].values


#Featuring Scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
training_set=sc.fit_transform(training_set)

#getting the input and the outputs
x_train=training_set[0:1257]
y_train=training_set[1:1258]
print(x_train)
print(y_train)
print(training_set)

#reshapeing
x_train=np.reshape(x_train,(1257,1,1))

#Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Initialising the RNN
regressor=Sequential()

#Adding the input layer and the LSTM layer
regressor.add(LSTM(unit=4,activation='sigmoid',input_shape=(None,1)))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')

#Fitting the RNNto the Training set
regressor.fit(x_train,y_train,batch_size=32,epochs=200)


#part -3 - Making the predictions and visualising the results
#Getting the real stock price of 2017


test_set=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_set.iloc[:,1:2].values



#Getting the predicted stock price of 2017
inputs=real_stock_price
inputs=sc.transform(inputs)
print(inputs)

inputs=np.reshape(inputs,(20,1,1))
predicted_stock_price=regressor(inputs)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualising the results
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google stock Price')
plt.legend()
plt.show()

#Homework
#Getting the real stock price of 2012-2016
real_stock_price_train=pd.read_csv('Google_Stock_Price_Train.csv')
real_stock_price_train=real_stock_price_train.iloc[:,1:2].values
#Getting the predicted stock price of 2012 - 2016

predicted_stock_price_train=regressor.predict(x_train)
predicted_stock_price_train=sc.inverse_transform(predicted_stock_price_train)


#Visualising the result

plt.plot(real_stock_price_train,color='red',label='Real Google Stock Price')
plt.plot(Predicted_stock_price_train,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


#part- 4 Evaluating the RNN
import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price)




























