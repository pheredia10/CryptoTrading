import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM,TimeDistributed,Conv1D
from keras.utils import np_utils
from keras import backend as K
from utils import *
import pickle
import sys

price_change_mat,coin_dict=data_Preprocessing('binaryprices-9630m.p','orderedbtcpairs.txt') 
# price_change_mat=price_change_mat[1:10,1:10]
coin_ind=coin_dict['ETHBTC']
size=price_change_mat.shape
periods= size[1] #11664
num_coins=size[0] #96 #120


def create_dataset(dataset, look_back=1):
	dataX=[]
	# for n in range(num_coins):
		# row=[]
		# print(dataset.shape)
		# print(len(dataset))
	for i in range(dataset.shape[0]-look_back+1):

	
		a = dataset[i:(i+look_back), :]
		# row.append(a)
		dataX.append(a)
	

	# if n==2:
	# print('dataX',dataX)
	# print('jj')
	# sys.exit()
	
			# dataY.append(dataset[i + look_back, n])
	return np.array(dataX)

# y_train=np.zeros((periods,num_coins))
look_back=10
# X_train=np.transpose(price_change_mat)
X_train=create_dataset(np.transpose(price_change_mat),look_back)
X_train=X_train[:-1,:,coin_ind]
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1]))
print('ttt',X_train.shape)


y_train=np.transpose(price_change_mat[:,look_back:periods+1])  # Predicting next price
y_train=y_train[:,coin_ind]

# y_train=np.zeros((periods-look_back+1,num_coins))     # Prediting next best portfolio
# for t in range(look_back-1,periods-1):
# 	y_train[t-look_back+1,:]=bestStrategy(price_change_mat,t,num_coins)


# X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1]))
print('y size',y_train.shape)
# y_train = np.reshape(y_train, (y_train.shape[0],1,y_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0]))
print('y size',y_train.shape)

# y_train=np.repeat(y_train,look_back,axis=1)
print('y size',y_train.shape)
print(X_train.shape)
# testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

hidden_size=20#num_coins
# y_train=np.reshape(y_train,(y_train.shape[0],y_train.shape[1],1))


X_list=[]
y_list=[]
if True:
	new_t = 15
	for i in range(15):
		old_shape = X_train.shape
		X_list.append(np.reshape(X_train[i,:],(old_shape[1])))
		y_list.append(y_train[i]) 
		
		# X_train = X_train[:new_t,:]
		# y_train = y_train[:new_t]
		# old_shape = X_train.shape
	# X_train = X_train.reshape((old_shape[0],old_shape[1]))
	
X_train_size=X_train.shape
print('X_train size',X_train_size)
use_dropout=True
print('hidden size',hidden_size)

model = Sequential()
# model.add(Conv1D(5,48,input_shape=(look_back,X_train_size[2])))
# model.add(LSTM(hidden_size, return_sequences=True))

# model.add(LSTM(hidden_size, return_sequences=True,input_shape=(look_back,X_train_size[2])))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))


model.add(Dense(hidden_size,batch_input_shape=(10,10)))
model.add(Activation('relu'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))
# model.add(Dense(hidden_size))
# model.add(Activation('softmax'))

# model.add(LSTM(hidden_size, return_sequences=True))
# model.add(LSTM(hidden_size, return_sequences=True))

# model.add(LSTM(hidden_size, return_sequences=True,input_shape=(1,X_train_size[2])))
# model.add(LSTM(hidden_size, return_sequences=True,input_shape=(1,X_train_size[2])))
# model.add(LSTM(hidden_size, return_sequences=True,input_shape=(X_train_size[1],X_train_size[2])))
# if use_dropout:
#     model.add(Dropout(0.5))

# model.add(TimeDistributed(Dense(units=1)))
model.add((Dense(units=1)))

# model.add(Dense(units=1)) #num_coins

print('y',y_train.shape)

# y_train=y_train[:,1:5]
# print('y',y_train)
print('x_list',X_list)
print('y_list',y_list)
# sys.exit()
model.compile(loss='mse', optimizer='Adam',metrics=['mae'])
model.fit(X_list, y_list, epochs=10000, batch_size=32)
#X_train = X_train.reshape(old_shape[:-1])


np.random.seed(1)
X_train = np.random.random((10,10))
y_train = np.random.random(10)

from sklearn.neural_network import MLPRegressor
mlp_clf = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100),max_iter=1000)
mlp_clf.fit(X_train,y_train)		
print('mlp score')
print (mlp_clf.score(X_train,y_train))


