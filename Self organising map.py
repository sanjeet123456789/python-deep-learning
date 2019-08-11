#Self Organizing Map

#Importing the libries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing dataset
dataset=pd.read_csv('Credit_Card_Application.csv')

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values


#Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
X=sc.fit_transform(X)

#training the Som

from minisom import MiniSom
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

# Visualizing the results
from pyplot import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
marker=['o','s']
color=['r','g']
for i,x in enumerate(X):
	w=som.winner(X)
	plot(w[0] +0.5,
		w[1]+0.5,
		markers[y[i]],
		markeredgecolor=colors[y[i]] ,
		markerfacecolor=None,
		markersize=10,
		markeredgewidth=2

		)

show()

#Finding the fraud 
mappings = som.win_map(X) 
frauds=np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)
frauds= sc.inverse_transform(frauds)




#part- 2 Going from Unsupervised to supervised  Deep Learning
#Creating the matrix of feature
customer=dataset.iloc[:,1:].values
#Creating the dependent variable
is_fraud=np.zeros(len(dataset))
for i in range(len(dataset)):
	if dataset.iloc[i,0] in frauds:
		is_fraud[i] =1

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
customers=sc.fit_transform(customers)
# making the Ann
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
	#Adding the input and the first hidden layer
classifier.add(Dense(units=2,kernel_initializer='uniform',activation='relu',input_dim=15))

#Adding the output layer
classifier.add(Dense(units=1,kernal_initializer='uniform',activation='sigmoid'))

#Compiling the ANN to the Training set
classifier.fit(customers.is_fraud,batch_size=1,epochs=2)

## part-3 Making the prediction and evaluating the model
#Predicting the Test set result
y_pred=classifier.predict(customers)
y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)
ypred=y_pred[y_pred[:,1].argsort()]
print(y_pred)

























