import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import math

from sklearn import svm
from sklearn import decomposition
import matplotlib
import datetime

# Look into setting a system of exposure between strategies: Ex- uniform rebalance and PMR2


def data_Preprocessing(pricesfile,pairsfile):
	price_change_mat=pickle.load(open(pricesfile,'rb'))
	pairs=open(pairsfile)
	coin_dict={}
	
	for count, line in enumerate(pairs):
		
		coin_dict[line.strip()]=count
		
	return price_change_mat[:,:-2],coin_dict
	

def PAMR(price_change_vec,portfolio_vec,t):
	m=num_coins+1
	
	x_bar=np.dot(price_change_vec,np.ones((m)))/m
	epsilon=0.9

	if np.absolute(np.sum(price_change_vec-np.ones((m))))>0:
		if t==1:
			# tau_right=(.9-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			portfolio_next=np.ones((m))/m
		else:
			tau_right=(np.dot(portfolio_vec,price_change_vec)-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			if tau_right>0:
				tau_t=tau_right
			else:
				tau_t=0


			portfolio_next=portfolio_vec -tau_t*(price_change_vec-x_bar*np.ones((m)))

			for i in range(np.size(portfolio_next)):
				if portfolio_next[i]<0:
					portfolio_next[i]=0
			portfolio_next=portfolio_next/np.sum(portfolio_next)


			
	else:
		portfolio_next=portfolio_vec


	return portfolio_next

def PAMR1(price_change_vec,portfolio_vec,t):
	m=num_coins+1
	C=50
	
	x_bar=np.dot(price_change_vec,np.ones((m)))/m
	epsilon=0.9

	if np.absolute(np.sum(price_change_vec-np.ones((m))))>0:
		if t==1:
			# tau_right=(.9-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			portfolio_next=np.ones((m))/m
		else:
			tau_inner=(np.dot(portfolio_vec,price_change_vec)-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			if tau_inner<C:
				tau_right=tau_inner
			else:
				tau_right=C
			if tau_right>0:
				tau_t=tau_right
			else:
				tau_t=0


			portfolio_next=portfolio_vec -tau_t*(price_change_vec-x_bar*np.ones((m)))

			for i in range(np.size(portfolio_next)):
				if portfolio_next[i]<0:
					portfolio_next[i]=0
			portfolio_next=portfolio_next/np.sum(portfolio_next)


			
	else:
		portfolio_next=portfolio_vec


	return portfolio_next


def PAMR2(price_change_vec,portfolio_vec,t,start,C,epsilon):
	m=num_coins
	# C=.1 #1.0  50 
	x_bar=np.dot(price_change_vec,np.ones((m)))/m

	def projection_simplex_sort(v, z=1):
		    n_features = v.shape[0]
		    u = np.sort(v)[::-1]
		    cssv = np.cumsum(u) - z
		    ind = np.arange(n_features) + 1
		    cond = u - cssv / ind > 0
		    rho = ind[cond][-1]
		    theta = cssv[cond][-1] / float(rho)
		    w = np.maximum(v - theta, 0)
		    return w
	
	if t==start:
			# tau_right=(.9-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			portfolio_next=np.ones((m))/m
			return portfolio_next
	if np.absolute(np.sum(price_change_vec-np.ones((m))))>0:
		
		
		
		
		if True:

			tau_right=(np.dot(portfolio_vec,price_change_vec)-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2 +(1/(2*C)))

			if tau_right>0:
				tau_t=tau_right
			else:
				tau_t=0

			commission_hack=False
			margin_cutoff=True
			if commission_hack: 
				init_guess=portfolio_vec-tau_t*(price_change_vec-x_bar*np.ones((m)))
				for i in range(np.size(init_guess)):
					if init_guess[i]<0:
						init_guess[i]=0
				init_guess=init_guess/np.sum(init_guess)
				commission=.5
				diff_portfolio=np.sum(np.absolute(init_guess-portfolio_vec))
				mu=commission*diff_portfolio
				print('mu virtual',mu)


				portfolio_next=portfolio_vec+(init_guess-portfolio_vec)*(1-mu)
			elif margin_cutoff:
				portfolio_next=np.zeros(m)
				# margin=.006
				#margin=.019
				stan_dev=np.std(price_change_vec)
				# mean=np.mean(price_change_vec)
				for i in range(m):
					# if np.abs(price_change_vec[i]-x_bar)< margin:

					if np.abs(price_change_vec[i]-x_bar)/stan_dev<2:
						portfolio_next[i]=portfolio_vec[i]
					else:
						portfolio_next[i]=portfolio_vec[i]-tau_t*(price_change_vec[i]-x_bar)

			else:
				portfolio_next=portfolio_vec-tau_t*(price_change_vec-x_bar*np.ones((m)))



			for i in range(np.size(portfolio_next)):
				if portfolio_next[i]<0:
					portfolio_next[i]=0


			portfolio_next=portfolio_next/np.sum(portfolio_next)

			
	else:
		portfolio_next=portfolio_vec


	return portfolio_next




def OLMAR2(price_change_mat,portfolio_vec,t,start):
	w=5
	price_change_sum=1
	for i in range(w):
		

		prod_price_change=np.ones([num_coins])
		for j in range(i+1):
			# print('t:'+str(t),'j1:'+str(j),prod_price_change[0])
			# print('price',price_change_mat[0,t-j])
			prod_price_change*=price_change_mat[:,t-j]
			# print('t:'+str(t),'j2:'+str(j),prod_price_change[0])
			
		# print('t:'+str(t),'j3:'+str(j),prod_price_change[0])
		price_change_sum+=1/prod_price_change
		# print('t:'+str(t),'i:'+str(i),price_change_sum[0])
	price_change_vec=(1/w)*price_change_sum
	m=num_coins
	C=.01 #1.0  50 
	x_bar=np.dot(price_change_vec,np.ones((m)))/m
	epsilon=.81  #.9809  #.71 

	def projection_simplex_sort(v, z=1):
		    n_features = v.shape[0]
		    u = np.sort(v)[::-1]
		    cssv = np.cumsum(u) - z
		    ind = np.arange(n_features) + 1
		    cond = u - cssv / ind > 0
		    rho = ind[cond][-1]
		    theta = cssv[cond][-1] / float(rho)
		    w = np.maximum(v - theta, 0)
		    return w
	
	if t==start:
			# tau_right=(.9-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
			portfolio_next=np.ones((m))/m
			return portfolio_next
	if np.absolute(np.sum(price_change_vec-np.ones((m))))>0:
		
		
		
		
		if True:

			tau_right=(np.dot(portfolio_vec,price_change_vec)-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2 +(1/(2*C)))

			if tau_right>0:
				tau_t=tau_right
			else:
				tau_t=0
			
			mag = np.sum(np.abs(price_change_vec-x_bar))*0.001

			new_vec = np.random.random(m)-1
			new_vec = new_vec/np.sum(new_vec)
			new_vec*=mag

			# print(new_vec,'\n', price_change_vec-x_bar*np.ones((m)))

			#portfolio_next=portfolio_vec -tau_t*new_vec
			portfolio_next=portfolio_vec-tau_t*(price_change_vec-x_bar*np.ones((m)))

			for i in range(np.size(portfolio_next)):
				if portfolio_next[i]<0:
					portfolio_next[i]=0

			portfolio_next=portfolio_next/np.sum(portfolio_next)

			
	else:
		portfolio_next=portfolio_vec


	return portfolio_next



def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def PAMR_quadprog(price_change_vec,portfolio_vec,t,start):
	P=np.eye(numcoins)
	P=np.vstack([P,C])



	if t==start:
				# tau_right=(.9-epsilon)/((np.linalg.norm(price_change_vec-x_bar*np.ones((m))))**2)
				portfolio_next=np.ones((m))/m
				return portfolio_next
	x_aug=quadprog_solve_qp(P,q,G,h)
	portfolio_next=x_aug[:numcoins-1]+portfolio_vec


	return portfolio_next





def create_dataset(dataset, look_back=1):
		dataX=[]
		
		for i in range(dataset.shape[0]-look_back):

		
			a = dataset[i:(i+look_back+1), :]
			dataX.append(a)
		return np.array(dataX)



def svmPreTrain(X,y):

	
	OPENTIME,OPEN,HIGH,LOW,CLOSE,VOLUME,CLOSTIME,QUOTEVOLUME,NTRADES,TAKERBUYBASEVOLUME,TAKERBUYQUOTEVOLUME,IGNORE=range(12) 
	svm1=svm.SVC(C=800000,gamma=.0003) # ,  kernel='poly',degree=5
	svm1.fit(X,y)
	print(svm1.score(X,y))

	return svm1


def svmPortfolio(svm_list,X_list,t):
	coins=len(svm_list)
	portfolio_next=np.zeros(num_coins)
	
	for i in range(coins):
		X=X_list[i][t,:]
		X=X.reshape(1,-1)
	
		if svm_list[i].predict(X)==1:
			portfolio_next[i]=1
	portfolio_next=portfolio_next/coins
	
	return portfolio_next


def svmWeeklyTrain(svm1,X,y_train):

	svm1.fit(X,y_train)
	return 0

# t=151 ,152
# price_change_mat,coin_dict=data_Preprocessing('binaryprices-9630m.p','orderedbtcpairs.txt')  #'binaryprices-30m.p'

OPENTIME,OPEN,HIGH,LOW,CLOSE,VOLUME,CLOSTIME,QUOTEVOLUME,NTRADES,TAKERBUYBASEVOLUME,TAKERBUYQUOTEVOLUME,IGNORE=range(12) 
features=[HIGH,LOW,CLOSE,QUOTEVOLUME,NTRADES,TAKERBUYQUOTEVOLUME]
K_all=pickle.load(open('p/rkline-30m.p','rb'))

btcpairs=open('btcpairs-jun-start.txt','r').readlines()
btcpairs=[btcpair.strip() for btcpair in btcpairs]
btcpairs=np.array(btcpairs)



opensellbuy=pickle.load(open('opensellbuy-30m.p','rb'))
opensell=pickle.load(open('opensell-30m.p','rb'))

price_change_mat=opensell



size=price_change_mat.shape
periods= size[1] #11664
num_coins=size[0] #96 #120
coin_indeces=range(1)

lookback=5
svmOn=False
if svmOn:
	X_list=[]
	y_list=[]
	for i in coin_indeces:
		X=np.transpose(K_all[i,:-1,features])
		X=create_dataset(X,lookback)
		samples=X.shape[0]
		feature_num=X.shape[1]*X.shape[2]
		X=np.reshape(X,(samples,feature_num))
		X_list.append(X)

		y=np.zeros((samples))
		for j in range(samples):
			# if K_all[coin_index,i+1,CLOSE]>K_all[coin_index,i,CLOSE]:
			if K_all[i,j+lookback+1,CLOSE]>1.0003:
				y[j]=1
			else:
				y[j]=0
		y_list.append(y)

bit_coin=False
if bit_coin:
	num_coins=num_coins+1
	price_change_mat=np.vstack([price_change_mat,np.ones((size[1]))]) #Adding riskless asset



# plt.plot(range(6000,periods+1),price_change_mat[35,5999:periods])
# plt.show()
# return 0


january_startstep=4416
month=48*31
start_vec=[0,january_startstep,6000]
start=0 #month #start_vec[0] 

if svmOn:
	svm_list=[]
	for i in coin_indeces:

		svm_list.append(svmPreTrain(X_list[i][0:start-lookback,:],y_list[i][0:start-lookback]))
	print('Validation Accuracy',svm_list[0].score(X_list[0][start:,:],y_list[0][start:]))


values_list=[]
max_val=0
bestC_list=[]
bestEps_list=[]
best_lastvalue_list=[]

#best params: C=1.4 , epsilon=0.79
for n in  [1.4]:  #np.arange(.1,10,.1):
	
	for j in [.79]: #np.arange(.1,1,.01):
		price_change_count=np.ones((num_coins))
		
		  #6000#january_startstep
		end= size[1]-1 #start+48*20  
		# price_change_mat=data_Preprocessing('orderedbtcpairs.txt','prices-30m.dat')



		# portfolio=np.zeros((num_coins+1,periods))
		portfolio=np.zeros((num_coins,periods))
		print('C',n)
		print('epsilon',j)
		print('numcoins',num_coins)

		portfolio[num_coins-1,start]=1
		cap_change_rate=np.ones((periods))
		mu=np.zeros((periods))
		portfolio_value=np.ones((periods))
		# price_change_mat[:,start]=np.ones(num_coins+1)
		price_change_mat[:,start]=np.ones(num_coins)
		# running_price_rate=np.zeros(num_coins+1)
		running_price_rate=np.zeros(num_coins)

		drop_count=np.zeros((num_coins))
		drop_count_mat=np.zeros((num_coins,periods))
		train_count=0
		for t in range(start,end):
			memory=48*7
		

			if t<memory:
					running_price_rate=(running_price_rate+price_change_mat[:,t])/(t-start+1)
			if t>=memory:
				running_price_rate=np.mean(price_change_mat[:,t-memory:t],axis=0) 

			prev_portfolio_vec=portfolio[:,t-1]
			price_change_vec=price_change_mat[:,t]
			price_change_next = price_change_mat[:,t+1]


			portfolio[:,t]=PAMR2(price_change_vec,prev_portfolio_vec,t,start,n,j)
			
			# portfolio[:,t]=OLMAR2(price_change_mat,prev_portfolio_vec,t,start)
			


			portfolio_vec=portfolio[:,t]
			


			if t>start:
				# diff_portfolio=((np.multiply(portfolio[:,t-1],price_change_returns[:,t])/np.sum(np.multiply(portfolio[:,t-1],price_change_returns[:,t])))-portfolio[:,t])
				diff_portfolio=(portfolio[:,t]-(np.multiply(portfolio[:,t-1],price_change_mat[:,t])/np.sum(np.multiply(portfolio[:,t-1],price_change_mat[:,t]))))
				# print('diff',diff_portfolio)
			else:
				# Shouldn't this be zero
				diff_portfolio=portfolio[:,t]
			C=0.0005
			mu[t]=C*np.sum(np.absolute(diff_portfolio))



			slippage=True
			passive_portfolio=np.multiply(portfolio[:,t-1],price_change_mat[:,t])
			if np.sum(portfolio_vec)!=0:
				#btc_position = 1 - np.sum(protfolio_vec)
				# cap_change_rate[t+1]=np.dot(portfolio_vec,price_change_returns[:,t+1])
				if slippage:
					cap_change_rate[t+1]=0
					for i in range(num_coins):
						if diff_portfolio[i]>0:
							
							cap_change_rate[t+1]+=passive_portfolio[i]*price_change_mat[i,t+1]+diff_portfolio[i]*opensellbuy[i,t+1]
						else:
							cap_change_rate[t+1]+=portfolio_vec[i]*price_change_mat[i,t+1]
						
					
				else:	
					cap_change_rate[t+1]=np.dot(portfolio_vec,price_change_mat[:,t+1])
			else:
				cap_change_rate[t+1]=1

			
		
			if t>start:
				# print(portfolio_value[t-1]*cap_change_rate[t]*(1-mu[t]))
				if np.sum(portfolio_vec)==0:
					portfolio_value[t+1]=portfolio_value[t]
				else:
					portfolio_value[t+1]=portfolio_value[t]*cap_change_rate[t+1]*(1-mu[t])


			week=7*48
			day=48
			if svmOn:
				if t%(week)==0 and False:
				
					train_count+=1
					for i in coin_indeces:
						# svmWeeklyTrain(svm_list[i],i,X_list[i][t-lookback-week:t-lookback,:])
						svmWeeklyTrain(svm_list[i],X_list[i][t-lookback-week:t-lookback,:],y_list[i][t-lookback-week:t-lookback])
				
			if t<periods-1:
				pass
				
		if False:
			periodsperday=48
			portfolio_value=portfolio_value[start:start+30*periodsperday]
			T=len(portfolio_value)
			print ('new time T',T)
			N_days=T//periodsperday
			dailyreturns=np.zeros(N_days)

			for day in range(N_days):
				if not  portfolio_value[day*periodsperday]==0:
					dailyreturns[day]=portfolio_value[(day+1)*periodsperday]/portfolio_value[day*periodsperday]
				else:
					dailyreturns[day]=1


			mean=np.mean(dailyreturns-1)
			std=np.std(dailyreturns-1)

			print ('newsharpe',mean,std,mean/std)	

			print('max price change',np.max(price_change_mat))
			print(portfolio_value[end-1])
			plt.figure(1)
			plt.plot(portfolio_value[periodsperday-1:periods:periodsperday])
			
			plt.figure(2)
			plt.plot(range(30*periodsperday-2),portfolio_value[:])
		


		periodsperday=48
		portfolio_value_day=portfolio_value[periodsperday-1:size[1]:periodsperday]
	
		if portfolio_value[-1]>max_val:
			max_val= portfolio_value[-1]
			best_values=portfolio_value
			best_params=[n,j]
			bestC_list.append(n)
			bestEps_list.append(j)
			best_lastvalue_list.append(max_val)

		print('current best params',best_params)
		print('current best final value',max_val)
		


	
print('max last value', max_val)
print('best params',best_params)
plt.plot(best_values)
plt.show()

