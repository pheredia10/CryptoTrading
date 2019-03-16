from rl.core import Env

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

OPENTIME,OPEN,HIGH,LOW,CLOSE,VOLUME,CLOSTIME,QUOTEVOLUME,NTRADES,TAKERBUYBASEVOLUME,TAKERBUYQUOTEVOLUME,IGNORE=range(12) 
BUYPERCENT,HIGHLOW,OPENBUY,OPENSELL,OPENSELLBUY,CLOSEBUY,CLOSESELL,CLOSESELLBUY,NTRADES2,TRADESAVG,TRADESCV,VWPAVG,VWPCV,PRICEAVG,PRICECV,VOLAVG,VOLCV,TAKERSELLAVG,TAKERSELLCV,TIMESPACEAVG,TIMESPACECV,TIMEAVG,TIMECV=range(12,12+23)
N_feature = 28
class CryptoEnv(Env):

	
	


	def __init__(self,start,end):

		
		# self.features=[HIGH,LOW,CLOSE,QUOTEVOLUME,NTRADES,TAKERBUYQUOTEVOLUME]
		# K_all=pickle.load(open('p/rkline-30m.p','rb'))

		self.weighted_avg_features=True

		K_all,R_all=pickle.load(open('dat/derivedall-30m.p','rb'))

		K_all=K_all[:,start:end,:]
		R_all=R_all[:,start:end,:]
		size=K_all[:,:,OPENSELLBUY].shape
		self.periods= size[1] #11664
		self.tot_coins=size[0]


		if True:
			period=30*60
			K_all[:,:,TIMEAVG]/=(1000*period)
			K_all[:,:,TIMESPACEAVG]/=(1000)
			R_all[np.isinf(R_all[:,:,HIGHLOW])]=1.0

		standardize=True
		self.stand_R_all=np.zeros((size[0],size[1],35))
		if standardize:
			for i in range(35):
				# mean_k=np.mean(K_all[:,:,i],axis=1)
				# mean_k=np.reshape(mean_k,(size[0],1))
				# mean_k=np.repeat(mean_k,size[1],axis=1)
				# print('mean_k',mean_k.shape)
				# std_k=np.std(K_all[:,:,i],axis=1)
				# std_k=np.reshape(std_k,(size[0],1))
				# std_k=np.repeat(std_k,size[1],axis=1)

				mean_r=np.mean(R_all[:,:,i],axis=1)
				mean_r=np.reshape(mean_r,(size[0],1))
				mean_r=np.repeat(mean_r,size[1],axis=1)
				std_r=np.std(R_all[:,:,i],axis=1)
				std_r=np.reshape(std_r,(size[0],1))
				std_r=np.repeat(std_r,size[1],axis=1)			

				# stand_K_all[:,:,i]=(K_all[:,:,i]-mean_k)/std_k
				self.stand_R_all[:,:,i]=(R_all[:,:,i]-mean_r)/std_r




			# for j in K_all[:,:,0].shape[0]:
			# 	K_all[j,:,i]=K_all[j,:,i]-
			


		# btcpairs=open('dat/btcpairs-jun-start.txt','r').readlines()
		# btcpairs=[btcpair.strip() for btcpair in btcpairs]
		# self.btcpairs=np.array(btcpairs)

		# self.opensellbuy=pickle.load(open('dat/opensellbuy-30m.p','rb'))

		# # self.opensellbuy=self.opensellbuy[0,:]
		# opensell=pickle.load(open('dat/opensell-30m.p','rb'))


		# dims = opensell.shape
		# opensell = opensell.reshape(dims[0]*dim[1])
		# opensell=opensell[0,:]
		# opensell = opensell[0].reshape(1,-1)


		self.opensellbuy=K_all[:,:,OPENSELLBUY]
		self.K_all=K_all
		self.R_all=R_all

		opensell=R_all[:,:,OPENSELL]

		self.price_change_mat=opensell

		# size=self.price_change_mat.shape
		
		self.num_coins=5#size[0] #96 #120
		if self.weighted_avg_features:
			self.dim_state=self.num_coins+N_feature+1
		else:
			self.dim_state=self.num_coins*(N_feature+1)+1
		self.C=.00075
		self.t=0
		self.mu=np.zeros(self.periods)
		self.cap_change_rate=np.ones((self.periods))
		self.portfolio_value=np.ones((self.periods))
		self.viewer=False
		self.done=False
		self.state_mem=np.zeros((self.num_coins,self.periods))
		self.counter=0
		# self.observation_space=np.zeros((1,self.num_coins))
		# self.action_space=np.zeros((1,self.num_coins))



	def step(self,action):
		# print('SHAAAAAAAAAAAAAAAAAAAAAAAAPE',self.state.shape)

		# passive_portfolio=np.zeros(num_coins)
		# print('action',action)

		def bestStrategy(price_change_next):
			num_coins=self.num_coins
			portfolio_next=np.zeros((self.tot_coins))
			num_selected=0
			for i in range(num_coins):
				if price_change_next[i] >1: 
					portfolio_next[i]=1
					# print(i)

					num_selected+=1

			if num_selected>0:
				portfolio_next/=num_selected
			# else:
			# 	portfolio_next[0]=1/price_change_mat[0,t+1]
				
			return portfolio_next


		prev_portfolio_vec=np.zeros(self.tot_coins)
		if self.t>0:
			
			prev_portfolio_vec[:self.num_coins]=self.state_mem[:,self.t-1]
		
			
		# print('prev_portfolio',prev_portfolio_vec)
		

		# prev_portfolio_vec=self.state

		price_change_vec= self.price_change_mat[:,self.t]
		price_change_next =self. price_change_mat[:,self.t+1]
		# print('pre_portfolio',self.t,prev_portfolio_vec)
		# print('price',price_change_vec)

	
		

		# if self.t>0:
		if np.sum(prev_portfolio_vec)!=0:

			
			passive_portfolio=(np.multiply(prev_portfolio_vec,price_change_vec)/np.sum(np.multiply(prev_portfolio_vec,price_change_vec)))
		else:
			passive_portfolio=np.zeros(self.tot_coins)
		# print('action',action)
		action_vec=np.zeros(self.tot_coins)
		action_vec[:self.num_coins]=action
		portfolio_temp=action_vec  #(passive_portfolio+action_vec)
		# print('passive',passive_portfolio)
		# print('action',action)
		# print('portfolio_temp',portfolio_temp)
		portfolio_vec=np.array([max(0,x) for x in portfolio_temp])
		if np.sum(portfolio_vec)!=0:

			portfolio_vec=portfolio_vec/np.sum(portfolio_vec)
		else:
			self.counter+=1


		# else:
	
			# portfolio_vec[:self.num_coins]=1/(-action)
			# portfolio_vec=portfolio_vec/np.sum(portfolio_vec)

		# print('update_portfolio',portfolio_vec)


		
		# self.state=portfolio_vec
		# if self.t==3:
		# 	plt.plot(self.portfolio_value)
		# 	plt.show()

		# if self.t>0 and np.sum(prev_portfolio_vec)==0:
		# 	sys.exit()
		


		# if np.isnan(action)[0]:
		# 	print('t,action',self.t,action)
		# 	sys.exit()
		

		if self.t>0:
			
			diff_portfolio=(portfolio_vec- passive_portfolio)
		else:
			# Shouldn't this be zero
			
			diff_portfolio=np.zeros(self.tot_coins)
		
		self.mu[self.t]=self.C*np.sum(np.absolute(diff_portfolio))
		# print('mu',self.mu[self.t])


		slippage=True
		  #np.multiply(portfolio[:,t-1],price_change_mat[:,t])
		if np.sum(portfolio_vec)!=0:
		
			if slippage:
				self.cap_change_rate[self.t+1]=0
				for i in range(self.num_coins):
					if diff_portfolio[i]>0:
						
						self.cap_change_rate[self.t+1]+=passive_portfolio[i]*price_change_next[i]+diff_portfolio[i]*self.opensellbuy[i,self.t+1]
					else:
						self.cap_change_rate[self.t+1]+=portfolio_vec[i]*price_change_next[i]
					
				
			else:	
				self.cap_change_rate[self.t+1]=np.dot(portfolio_vec,price_change_next)
		else:
			self.cap_change_rate[self.t+1]=1

		# print('max opensell',np.max(self.price_change_mat))
		
	
		if self.t>0:

			# if np.sum(portfolio_vec)==0:
			# 	self.portfolio_value[self.t+1]=self.portfolio_value[self.t]
			# else:
			# 	self.portfolio_value[self.t+1]=self.portfolio_value[self.t]*self.cap_change_rate[self.t+1]*(1-self.mu[self.t])
			self.portfolio_value[self.t+1]=self.portfolio_value[self.t]*self.cap_change_rate[self.t+1]*(1-self.mu[self.t])

		# print('t, cap_change',self.cap_change_rate[self.t])
		# print('portfolio_vec',portfolio_vec)
		# print('price_change_next',price_change_next)
		# print('action',action)

		reward= 10000*(self.portfolio_value[self.t+1]/self.portfolio_value[self.t]-1) #-10*np.sum(np.abs(action)) #self.portfolio_value[self.t+1]/self.portfolio_value[self.t]
		
		# best_portfolio=bestStrategy(price_change_next)
		# reward= -10*np.linalg.norm(portfolio_vec-best_portfolio)**2 #-1*np.sum(np.abs(action))

		# print('reward2',reward)
		if self.portfolio_value[self.t+1]==0:
			self.done=True
		# print('t',self.t)
		# print('periods',self.periods)
		# if True and  self.t==self.periods-4:
		# 	plt.plot(self.portfolio_value)
		# 	plt.show()

		# print('portfolio_value',self.portfolio_value[self.t])
		# print('state',self.state)


		###################  Fix this #########################
		featureindices = [HIGH,LOW,CLOSE,NTRADES,QUOTEVOLUME] +list(range(12,12+23))
		featureindices = np.array(featureindices)
		# featureindices = np.array([CLOSE])

		if N_feature > 0:
			# print('hey',self.R_all[:self.num_coins,self.t,featureindices[:N_feature]].reshape(-1,1))
			# print('here',np.concatenate( (portfolio_vec[:self.num_coins].reshape(self.num_coins,1),self.R_all[:self.num_coins,self.t,featureindices[:N_feature]].reshape(-1,1)),axis=0))

			# self.state[:self.num_coins*(N_feature+1)]=np.concatenate( (portfolio_vec[:self.num_coins].reshape(self.num_coins,1),self.R_all[:self.num_coins,self.t,featureindices[:N_feature]].reshape(-1,1)),axis=0).reshape(-1)
			if self.weighted_avg_features:
				weights=(np.repeat(portfolio_vec[:self.num_coins].reshape((self.num_coins,1)),N_feature,1))
				# print(weights.shape)
				# print('fff',self.stand_R_all[:self.num_coins,self.t,featureindices[:N_feature]].shape)
				weighted_avgs=(np.sum((weights*self.stand_R_all[:self.num_coins,self.t,featureindices[:N_feature]]),0)).reshape(-1)
				self.state[:self.num_coins+N_feature]=np.concatenate((portfolio_vec[:self.num_coins],weighted_avgs))
			else:		

				self.state[:self.num_coins*(N_feature+1)]=np.concatenate( (portfolio_vec[:self.num_coins].reshape(self.num_coins,1),self.stand_R_all[:self.num_coins,self.t,featureindices[:N_feature]].reshape(-1,1)),axis=0).reshape(-1)


		else:
			self.state[:self.num_coins] = portfolio_vec[:self.num_coins].reshape(self.num_coins,1).reshape(-1)
		
			# self.state = portfolio_vec
		# print('SHAAAAAAAAAAAAAAAAAAAAAAAAPE',self.state.shape)
		# if self.t>5:
		# 	sys.exit()
		self.state[-1]=self.portfolio_value[self.t+1]

		self.state_mem[:,self.t]=self.state[:self.num_coins]

		# print('update state',self.state)

		self.t+=1
		# print(self.counter)
		# print('action',action)
	
		return self.state, reward, self.done,{}

	def reset(self):


		
		# if self.weighted_avg_features:

		# 	self.state= np.zeros((self.num_coins+N_feature+1),dtype=np.float64)
		# else:
		# 	self.state= np.zeros((self.num_coins*(N_feature+1)+1),dtype=np.float64)

		self.state=np.zeros(self.dim_state,dtype=np.float64)
		# self.state= np.zeros(self.num_coins*(N_feature+1))
		self.t=0
		self.mu=np.zeros(self.periods)
		self.cap_change_rate=np.ones((self.periods))
		self.portfolio_value=np.ones((self.periods))
		self.viewer=False
		self.done=False
		self.state_mem=np.zeros((self.num_coins,self.periods))
		return self.state

	def render(self,mode='human'):
		# if self.t==self.periods-50:
		# 	plt.plot(self.portfolio_value)
		# 	plt.show(block=False)
		
		# self.viewer=True
		pass
	

	def close(self):
		# if self.viewer:
		# 	plt.close()
		# 	self.viewer=False
		pass
		
	

	def seed(self,seed):
		pass
	def configure(self):
		pass



class CryptoEnv2(Env):

	
	


	def __init__(self):

		
		# self.features=[HIGH,LOW,CLOSE,QUOTEVOLUME,NTRADES,TAKERBUYQUOTEVOLUME]
		# K_all=pickle.load(open('p/rkline-30m.p','rb'))

		K_all,R_all=pickle.load(open('dat/derivedall-30m.p','rb'))

		if True:
			period=30*60
			K_all[:,:,TIMEAVG]/=(1000*period)
			K_all[:,:,TIMESPACEAVG]/=(1000)
			R_all[np.isinf(R_all[:,:,HIGHLOW])]=1.0
			


		# btcpairs=open('dat/btcpairs-jun-start.txt','r').readlines()
		# btcpairs=[btcpair.strip() for btcpair in btcpairs]
		# self.btcpairs=np.array(btcpairs)

		# self.opensellbuy=pickle.load(open('dat/opensellbuy-30m.p','rb'))

		# # self.opensellbuy=self.opensellbuy[0,:]
		# opensell=pickle.load(open('dat/opensell-30m.p','rb'))


		# dims = opensell.shape
		# opensell = opensell.reshape(dims[0]*dim[1])
		# opensell=opensell[0,:]
		# opensell = opensell[0].reshape(1,-1)


		self.opensellbuy=K_all[:,:,OPENSELLBUY]
		self.K_all=K_all
		self.R_all=R_all

		opensell=R_all[:,:,OPENSELL]

		self.price_change_mat=opensell

		size=self.price_change_mat.shape
		self.periods= size[1] #11664
		self.num_coins=1#size[0] #96 #120
		self.tot_coins=size[0]
		self.dim_state=1#self.num_coins*(N_feature)
		self.C=.0005
		self.t=0
		self.mu=np.zeros(self.periods)
		self.cap_change_rate=np.ones((self.periods))
		self.portfolio_value=np.ones((self.periods))
		self.viewer=False
		self.done=False
		self.portfolio_mem=np.zeros((self.num_coins,self.periods))
		self.counter=0
		# self.observation_space=np.zeros((1,self.num_coins))
		# self.action_space=np.zeros((1,self.num_coins))



	def step(self,action):
		# print('SHAAAAAAAAAAAAAAAAAAAAAAAAPE',self.state.shape)

		# passive_portfolio=np.zeros(num_coins)
		# print('action',action)

		def bestStrategy(price_change_next):
			num_coins=self.num_coins
			portfolio_next=np.zeros((self.tot_coins))
			num_selected=0
			for i in range(num_coins):
				if price_change_next[i] >1: 
					portfolio_next[i]=1
					# print(i)

					num_selected+=1

			if num_selected>0:
				portfolio_next/=num_selected
			# else:
			# 	portfolio_next[0]=1/price_change_mat[0,t+1]
				
			return portfolio_next


		prev_portfolio_vec=np.zeros(self.tot_coins)
		if self.t>0:
			
			prev_portfolio_vec[:self.num_coins]=self.portfolio_mem[:,self.t-1]
		
			
		# print('prev_portfolio',prev_portfolio_vec)
		

		# prev_portfolio_vec=self.state

		price_change_vec= self.price_change_mat[:,self.t]
		price_change_next =self. price_change_mat[:,self.t+1]
		# print('pre_portfolio',self.t,prev_portfolio_vec)
		# print('price',price_change_vec)

	
		

		# if self.t>0:
		if np.sum(prev_portfolio_vec)!=0:

			
			passive_portfolio=(np.multiply(prev_portfolio_vec,price_change_vec)/np.sum(np.multiply(prev_portfolio_vec,price_change_vec)))
		else:
			passive_portfolio=np.zeros(self.tot_coins)
		# print('action',action)
		action_vec=np.zeros(self.tot_coins)
		action_vec[:self.num_coins]=action
		portfolio_temp= action_vec  #(passive_portfolio+action_vec)
		# print('passive',passive_portfolio)
		# print('action',action)
		# print('portfolio_temp',portfolio_temp)
		portfolio_vec=np.array([max(0,x) for x in portfolio_temp])
		if np.sum(portfolio_vec)!=0:

			portfolio_vec=portfolio_vec/np.sum(portfolio_vec)
		else:
			self.counter+=1


		# else:
	
			# portfolio_vec[:self.num_coins]=1/(-action)
			# portfolio_vec=portfolio_vec/np.sum(portfolio_vec)


		

		if self.t>0:
			
			diff_portfolio=(portfolio_vec- passive_portfolio)
		else:
			# Shouldn't this be zero
			
			diff_portfolio=np.zeros(self.tot_coins)
		
		self.mu[self.t]=self.C*np.sum(np.absolute(diff_portfolio))
		# print('mu',self.mu[self.t])


		slippage=True
		  #np.multiply(portfolio[:,t-1],price_change_mat[:,t])
		if np.sum(portfolio_vec)!=0:
		
			if slippage:
				self.cap_change_rate[self.t+1]=0
				for i in range(self.num_coins):
					if diff_portfolio[i]>0:
						
						self.cap_change_rate[self.t+1]+=passive_portfolio[i]*price_change_next[i]+diff_portfolio[i]*self.opensellbuy[i,self.t+1]
					else:
						self.cap_change_rate[self.t+1]+=portfolio_vec[i]*price_change_next[i]
					
				
			else:	
				self.cap_change_rate[self.t+1]=np.dot(portfolio_vec,price_change_next)
		else:
			self.cap_change_rate[self.t+1]=1

		# print('max opensell',np.max(self.price_change_mat))
		
	
		if self.t>0:

			# if np.sum(portfolio_vec)==0:
			# 	self.portfolio_value[self.t+1]=self.portfolio_value[self.t]
			# else:
			# 	self.portfolio_value[self.t+1]=self.portfolio_value[self.t]*self.cap_change_rate[self.t+1]*(1-self.mu[self.t])
			self.portfolio_value[self.t+1]=self.portfolio_value[self.t]*self.cap_change_rate[self.t+1]*(1-self.mu[self.t])

		# print('t, cap_change',self.cap_change_rate[self.t])
		# print('portfolio_vec',portfolio_vec)
		# print('price_change_next',price_change_next)
		# print('action',action)

		reward= 10000*(self.portfolio_value[self.t+1]/self.portfolio_value[self.t]-1) #-10*np.sum(np.abs(action)) #self.portfolio_value[self.t+1]/self.portfolio_value[self.t]
		
		# best_portfolio=bestStrategy(price_change_next)
		# reward= -10*np.linalg.norm(portfolio_vec-best_portfolio)**2 #-1*np.sum(np.abs(action))

		# print('reward2',reward)
		if self.portfolio_value[self.t+1]==0:
			self.done=True
		# print('t',self.t)
		# print('periods',self.periods)
		if True and  self.t==self.periods-4:
			plt.plot(self.portfolio_value)
			plt.show()

		# print('portfolio_value',self.portfolio_value[self.t])
		# print('state',self.state)



		

		self.state=self.portfolio_value[self.t+1]
			# self.state = portfolio_vec
		# print('SHAAAAAAAAAAAAAAAAAAAAAAAAPE',self.state.shape)
		# if self.t>5:
		# 	sys.exit()

		self.portfolio_mem[:,self.t]=portfolio_vec[:self.num_coins]
		# print('update state',self.state)

		self.t+=1
		# print(self.counter)
		# print('action',action)
	
		return self.state, reward, self.done,{}

	def reset(self):
		self.state=1 #np.zeros(self.num_coins*(N_feature+1),dtype=np.float64)
		# self.state= np.zeros(self.num_coins*(N_feature+1))
		self.t=0
		self.mu=np.zeros(self.periods)
		self.cap_change_rate=np.ones((self.periods))
		self.portfolio_value=np.ones((self.periods))
		self.viewer=False
		self.done=False
		self.portfolio_mem=np.zeros((self.num_coins,self.periods))
		return self.state

	def render(self,mode='human'):
		# if self.t==self.periods-50:
		# 	plt.plot(self.portfolio_value)
		# 	plt.show(block=False)
		
		# self.viewer=True
		pass
	

	def close(self):
		# if self.viewer:
		# 	plt.close()
		# 	self.viewer=False
		pass
		
	

	def seed(self,seed):
		pass
	def configure(self):
		pass