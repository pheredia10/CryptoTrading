from sklearn import svm
from sklearn import decomposition
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def create_dataset(dataset, look_back=1):
	dataX=[]
	
	for i in range(dataset.shape[0]-look_back+1):

	
		a = dataset[i:(i+look_back), :]
		dataX.append(a)
	return np.array(dataX)


OPENTIME,OPEN,HIGH,LOW,CLOSE,VOLUME,CLOSTIME,QUOTEVOLUME,NTRADES,TAKERBUYBASEVOLUME,TAKERBUYQUOTEVOLUME,IGNORE=range(12) 
K_all=pickle.load(open('p/rkline-30m.p','rb'))
coin_index=4


size=K_all.shape



features=[HIGH,LOW,CLOSE,QUOTEVOLUME,NTRADES,TAKERBUYQUOTEVOLUME]
lookback=10
X=np.transpose(K_all[coin_index,:-1,features])
X=create_dataset(X,lookback)
samples=X.shape[0]
feature_num=X.shape[1]*X.shape[2]
X=np.reshape(X,(samples,feature_num))
print('X',X.shape)

scaling=False
if scaling:
	for n in range(X.shape[1]):
		
		X[:,n]=X[:,n]/np.max(X[:,n])

print(X)

# pca1=decomposition.PCA(n_components=10)
# X=pca1.fit_transform(X)


color_list=[]
# print('samples',samples)
for i in range(samples):
	# if K_all[coin_index,i+1,CLOSE]>K_all[coin_index,i,CLOSE]:
	if K_all[coin_index,i+1,CLOSE]>1:
		color_list.append(matplotlib.colors.to_rgb('b'))
	else:
		color_list.append(matplotlib.colors.to_rgb('r'))

# print(color_list)
# plt.scatter(X[:,0],X[:,1],c=color_list,s=1)
# plt.show()

y=np.zeros((samples))
for i in range(samples):
	# if K_all[coin_index,i+1,CLOSE]>K_all[coin_index,i,CLOSE]:
	if K_all[coin_index,i+1,CLOSE]>1:
		y[i]=1
	else:
		y[i]=-1

svm1=svm.SVC(C=9000,gamma=.003) # ,  kernel='poly',degree=5
svm1.fit(X,y)

K_all_test=pickle.load(open('p/rkline-30mtest.p','rb'))

X_test=np.transpose(K_all_test[coin_index,:-1,features])
X_test=create_dataset(X_test,lookback)
samples_test=X_test.shape[0]
features_test=X_test.shape[1]*X_test.shape[2]
X_test=np.reshape(X_test,(samples_test,features_test))
print('X_test',X_test.shape)
# X_test=K_all_test[coin_index,:-1,:-1]

# if scaling:

# 	for n in range(X_test.shape[1]):
# 		X_test[:,n]=X_test[:,n]/np.max(X_test[:,n])

size_test=K_all_test.shape
y_test=np.zeros(samples_test)

# for i in range(samples_test):
# 	# if K_all[coin_index,i+1,CLOSE]>K_all[coin_index,i,CLOSE]:
# 	if K_all_test[coin_index,i+1,CLOSE]>1:
# 		color_list.append(matplotlib.colors.to_rgb('b'))
# 	else:
# 		color_list.append(matplotlib.colors.to_rgb('r'))


# for i in range(size_test[1]-1):
# 	if K_all_test[coin_index,i+1,OPEN]>K_all_test[coin_index,i,OPEN]:
# 		y_test[i]=1
# 	else:
# 		y_test[i]=-1

for i in range(samples_test):
	# if K_all[coin_index,i+1,CLOSE]>K_all[coin_index,i,CLOSE]:
	if K_all_test[coin_index,i+1,CLOSE]>1:
		y_test[i]=1
	else:
		y_test[i]=-1


# print(X_test.shape,y_test.shape)


print(svm1.score(X,y))

print(svm1.score(X_test,y_test))




