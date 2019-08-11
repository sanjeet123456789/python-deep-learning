# dataset available on grouplens.org/dataset/movieslens 1000K and 1 M with meomory sieze value 5 and 6 M
#Boltzmann Machines

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch .nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autogrid import Variable


#importing the dataset
movies=pd.read_csv('ml-lm/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-lm/users.dat',sep='::',header='None',engine='python',encoding='latin-1')
rating=pd.read_csv('ml-lm/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

#Preparing the training set and the test set
training_set=pd.read_csv('ml-100k/ul.base',delimiter='\t')
training_set=np.array(training_set,dtype='int')
test_set=np.read_csv('ml-100k/ul.test',delimiter='\t')
test_set=np.array(test_set,dtype='int')


#Getting the number of user and movies
nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,0]),max(test_set[:,1])))


#Converting the data into and array in lines and movies in columns
def convert(data):
	new_data=[]
	for id_users in range(1,nb_users+1)
		id_movies = data[:,1][data[:,0]==id_users]
		id_rating= data[:,2][data[:,0]==id_users]
		ratings=np.zeros(nb_movies)
		ratings[id_movies-1]=id_ratings
		new_data.append(list(ratings))
	return new_data
training_set=convert(training_set)
test_set=convert(test_set)

#converting the data into torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_Set)
#converting the rating into binary 1 (liked) or 0 (Not Linked)
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1
test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1

#Creating the architecture of the Neural Network
class RBM():
	def __init__(self,nv,nh):
		self.W = torch.randn(nh,nv)
		self.a=torch.randn(1, nh)  #batch and bises 2 dimentions
		self.b=torch.rand(1,nv)	#biased
	def sample_h(self,x): #x correspond to given probabity of hidden of given v

		wx=torch.mm(x,self.W.t())
		activation= wx+self.a.expand_as(wx)
		p_h_given_v = torch.sigmoid(activation)
		return p_h_given_v,torch.bernoulli(p_h_given_v)

	def sample_v(self,y): #x correspond to given probabity of hidden of given v

		wy=torch.mm(y,self.W)
		activation= wy+self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		return p_v_given_h,torch.bernoulli(p_v_given_h)

	def train(self,v0,vk,ph0,phk):
		self.W += torch.mm(v0,t(), ph0)-torch.mm(vk.t(),phk)
		self.b += torch.sum((v0 - vk),0)#two dimension
		self.a += torch.sum((ph0 - phk),0)

nv = len(training_set[0])
nh = 100
batch_size=100
rbm= RBM(nv,nh)

#Training the RBM
nb_epoch=10
for epoch in range(1,nb_epoch+1):
	train_loss=0
	s=0.
	for id_user in range(0,nb_users - batch_size,batch_size):
		vk = training_set[id_user:id_user+batch_size]
		v0=training_set[id_user:id_user+batch_size]
		ph0,_ = rbm.sample_h(v0)
		for k in range(10):
			_,hk = rbm.sample_h(vk)
			_,vk = rbm.sample_v(hk)
			vk[v0<0] = v0[v0<0]
		phk,_=rbm.sample_h(vk)
		rbm.train(v0,vk,ph0,phk)
		train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
		s += 1.
	print('epoch:'+str(epoch)+' loss: '+str(train_loss/s))

	#Testing the RBM
test_loss=0
s=0.
for id_user in range(nb_users):
	v = training_set[id_user:id_user+1]
	vt=test_Set[id_user:id_user+1]
	if len(vt[vt>=0]) > 0:
		_,h = rbm.sample_h(v)
		_,v = rbm.sample_v(h)
	test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
	s += 1.
print('epoch:'+str(epoch)+' loss: '+str(test_loss/s))


