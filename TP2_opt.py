import matplotlib as mpl
from matplotlib import cm
from math import sqrt
from math import exp, expm1
from numpy import random
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


np.random.seed()
def g(x,a):
	return exp(-1*a*x)

def g2(x,a1, a2):
	return x**a1* exp(-1*a2*x)

def vg2(x,a1, a2):
	size_x =np.size(x)
	y=np.zeros(size_x)
	for i in range(0,size_x):
		y[i] =x[i]**a1* exp(-1*a2*x[i])
	return (y)	
	
def vg(x,a):
	size_x =np.size(x)
	y=np.zeros(size_x)
	for i in range(0,size_x):
		current_x = x[i]
		y[i] = exp(-1*a*current_x)
	return (y)	


def random_data_set(x,a,b) : # B est l'amplitude du bruit

	y=np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		current_x = x[i]
		y[i] = g(current_x, a) + b*np.random.normal(0,1, 1)	
	return (y)	

def random_data_set2(x,a1,a2,b) : # B est l'amplitude du bruit

	y=np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		y[i] = g(x[i], a1, a2) + b*np.random.normal(0,1, 1)	
	return (y)	

def cost_fucntion(x,y,a):
	# size of x,y vectors are equal
	g_xa = vg(x,a)
	if np.size(y) == np.size(g_xa):
		SCE = 0.5*(y - g_xa)**2		
		f = sum(SCE)
	else :
		print('Dimension error')	
	return f	

def stop (cond, k, nb_iter, g, current_grad , fk , fkk):
	''' stop is a function use both by the gradient descend method and 
	Newton method. We have fuour stop criteria such as :

	0 => a maximum number of iterations (k_max)
	1 => a minimal  norm to reach (g_min)
	2 => union of 0 and 1 conditions
	3 => at each step we assure that f(x_k+1)< f(x_k)

	This function return impossible if the first argument is different from this list.'''

	if cond ==0 :
		c_stop = nb_iter < k
	elif cond ==1  :
		c_stop = current_grad < g
	elif cond==2 :
		c_stop =  nb_iter < k and current_grad < g
	elif cond==3 :
		c_stop = fkk < fk
			
	else:
		return 'impossible'	
	return c_stop				
	

def grad (x,y,a):
	g_xa = vg(x,a)
	if np.size(y) == np.size(g_xa):
		G = sum((y-g_xa)*(x*g_xa))
	else :
		print('Dimension error')	
	return G


def grad2 (x,y,a1,a2):
	df_da1 = -1 * sum(y -( x**a1*exp(-a2*x)))	* a1*x**(a1-1)*exp(-a2*x)
	df_da2 = -1 * sum(y -( x**a1*exp(-a2*x)))	* -a2*x**(a1)*exp(-a2*x)
	return df_da1 , df_da2
def derivative_2 (x,a):	
	g_xa = vg(x,a)
	return sum((-1*x*g_xa)**2)

def iter_LM(Last_L, x,y, Last_A , last_F):
	L = Last_L*10
	dLM = -1 * (grad(x,y,Last_A )/ (derivative_2(x ,Last_A)+ L ))
	next_f = cost_fucntion(x,y, Last_A + dLM)
	
	if next_f >  last_F :
		Last_L = L
		last_F = next_f
		

		return iter_LM(Last_L  , x,y, Last_A , last_F)
	else :
		# a_k+1
		LL = Last_L/10
		dLM = -1 * (grad(x,y,Last_A )/ (derivative_2(x ,Last_A)+ LL ))
		AA =  Last_A + dLM 
		return AA, LL 	


def LM (x,y,a,l,cond, k, g):
	f0 = cost_fucntion(x,y,a)
	f00 = cost_fucntion(x,y,a) -1 # atificial initiation for cond 3
	nb_iter =0
	L =[l]
	G =[grad(x,y,a)]
	F= [f0]
	A = [a]
	c_stop = stop(cond,k,nb_iter,g,G[-1],  f0, f00)	
	# def stop (cond, k, nb_iter, g, current_grad , fk , fkk):
	while c_stop==True:
		GG= abs(grad(x,y,A[-1] ))
		dLM = -1 * (grad(x,y,A[-1] )/ (derivative_2(x ,A[-1])+ L[-1] ))
		next_f = cost_fucntion(x,y, A[-1] + dLM)
		if next_f <  F[-1] :
			AA =  A[-1] + dLM # a_k+1
			LL = L[-1]/10
		else :
			sol =iter_LM(L[-1], x,y,A[-1], next_f)
			AA = sol[0]
			LL =  sol[1]
			# LL = L[-1]*10
			# while next_f <  F[-1]:
			# 	dLM = -1 * (grad(x,y,Last_A )/ (derivative_2(x ,Last_A)+ LL ))
			# 	next_f = cost_fucntion(x,y, Last_A + dLM)
			# 	LL = LL*10
			# AA = A[-1] +dLM
			# LL = LL /10
		L.append(LL)
		G.append(GG)
		F.append(next_f)
		A.append(AA)
		nb_iter +=1
		c_stop = stop(cond,k,nb_iter,g,G[-1], F[-1], next_f)	
			
	return L, G, F, A



###############################################################################################
#								MAIN														  #	
###############################################################################################

Which_question  = int(input('Which question ?	'))
if Which_question==1:
	print('''Gradient descent method : \n
	 		The gradient descent method was easy to implement, and could be use in all cases.
	 		This algorithm return in each case the minimum of a function according the steepest 
	 		descent, and a step 'a'. The choice of this step 'a' could be seen as a first drawback 
	 		of the method even if this could be optimmise. Then for some function poorly conditionated
	 		function, the algorithm could be extremly long.\n
	 		\n
	 		Newtwon method : \n
	 		This method is faster than gradient descent algorithm, so in fewer iterations it reachs the 
	 		minimum.Nevertheless this is true only if the  Hessian matrix on which is based the algorithm 
	 		is defined positve. Otherwise Newton method doesn't distinguish differences between saddle local
	 		minimum, or even local maximum.''')

if Which_question==2:	
	print('function g  returns the result of g(x)= e^(-ax) with x=0 and a=1		:',g(0,1))

if Which_question==3:	
	print('''The function random_data_set allow to generate a randomized data set such that y = g(a,x)+bN(0,1).
		  Users have to enter as parameters : \n
		   x -> a vector of x values\n
		   a -> such that g(x) = e^(-ax)
		   b -> factor allowing to add noise in data		''')
if Which_question==4: 
	print('''Representation of a data set with the following parameters :\n
			 a = 2 \n
			 x in [0,3] by 0.01
			 b = 0.01''')
	X= np.arange(0,3+0.01,0.01)
	y=random_data_set(X,2,0.01)
	fig = plt.figure() 
	plt.scatter(X, y)
	plt.plot(X,vg(X,2), c = 'Green')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

if Which_question==5: 
	x= np.arange(0,3+0.01,0.01)
	y=random_data_set(x,2,0.01)
	print(''' For the random data set such as  y = g(2, x) + 0.01 * N(0,1)
		and x=[0,3] by 0.1.  we calculate the cost function, and a=2. The result is  :''' ,
		 cost_fucntion(x,y,2))

if Which_question==6:
	x= np.arange(0,3+0.01,0.01)
	y=random_data_set(x,2,0.01)
	print(''' For the random data set such as  y = g(2, x) + 0.01 * N(0,1)
		and x=[0,3] by 0.1, and a=2 .  we calculate the gradadient according to the function named grad.
	    The result is  :''' , grad(x,y,2))

if Which_question==7:
	x= np.arange(0,3+0.01,0.01)
	print(''' For  x=[0,3] by 0.1 and a =2.  we calculate the second order derivative of the function g 
		   The result is  :''' , derivative_2(x,2))

if Which_question==8:
	x= np.arange(0,3+0.01,0.01)
	y=random_data_set(x,2,3)
	#def LM (x,y,a,l,c_stop, k, g):
	LMf1 =LM (x,y,0.5,0.001, 0, 10, 1)
	y_fit = vg(x,LMf1[3][-1])
	print(LMf1[2][0])
	print(LMf1[2][-1])
	print(LMf1[3][-1])
	fig = plt.figure() 
	plt.scatter(x, y)
	plt.plot(x,vg(x,2), c='black')
	plt.plot(x, y_fit, c='red')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.ylim(-0.5,1.5)
	plt.show()
	

if Which_question==9:
	x= np.arange(0,3+0.01,0.01)
	y=random_data_set(x,2,0.01)
	print(y[0:10])
	#def LM (x,y,a,l,c_stop, k, g):
	LMf1 =LM (x,y,1.5,0.001, 2, 10, 0.001)
	dic = {
    'lambda': LMf1[0],
    'norm gradient':LMf1[1],
 	'f(a_k)':LMf1[2],
 	'a ':LMf1[3],    	
	}
	df = pd.DataFrame(dic)
	print(df)


if Which_question==10 :
	x= np.arange(0,3+0.01,0.1)
	
	# B1 = list(np.arange(0, 1.2, 0.2))
	# B2 = list(np.arange(0, 6, 1))
	# B = B1 + B2[2:]
	B = list(np.arange(0, 2, 0.01))
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	for i in B :
		y=random_data_set(x,2,i)
		S =LM (x,y,1.5,0.001, 2, 10, 0.001)
		L_end.append (S[0][-1])
		G_end.append (S[1][-1])
		F_end.append(S[2][-1])
		A_end.append(S[3][-1])
		B_end.append(i)
	dic = {
	'B' : B_end,
    'lambda': L_end,
    'norm gradient':G_end,
 	'f(a_k)':F_end,
 	'a ':A_end    	
	}
	df = pd.DataFrame(dic)
	print(df)

	plt.figure(1)
	plt.subplot(221)
	plt.plot(B_end, L_end, 'o-')
	plt.ylabel('lambda')

	plt.subplot(222)	
	plt.plot(B_end, G_end, 'o-')
	plt.ylabel('Norm gradient')

	plt.subplot(223)	
	plt.plot(B_end, F_end, 'o-')
	plt.ylabel('f(a_k)')

	plt.subplot(224)	
	plt.plot(B_end, A_end, 'o-')
	plt.ylabel('a')
	plt.xlabel('b')

	plt.show()

if Which_question==11 :
	x= np.arange(0,3+0.01,0.01)
	
	# B1 = list(np.arange(0, 1.2, 0.2))
	# B2 = list(np.arange(0, 6, 1))
	# B = B1 + B2[2:]
	B = list(np.arange(0, 2, 0.1))
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	fig, axes = plt.subplots(nrows=10, ncols=2,  sharex=True, sharey=True)
	for i, ax in enumerate(axes.flatten()):

		y=random_data_set(x,2,i)
		S =LM (x,y,1.5,0.001, 2, 10, 0.001)
		L_end.append (S[0][-1])
		G_end.append (S[1][-1])
		F_end.append(S[2][-1])
		A_end.append(S[3][-1])
		B_end.append(i)
		y_fit = vg(x,S[3][-1])
	# 	plt.subplot(sub+c)
		ax.scatter(x, y, s=0.6 )
		ax.plot(x,vg(x,2), c='black')
		ax.plot(x, y_fit, c='red')
		ax.set_title(B[i])

	plt.ylim(-0.5,2)
	plt.xlim(0,3)
	
	plt.show()
	print(len(L_end))
	dic = {
	'B' : B_end,
    'lambda': L_end,
    'norm gradient':G_end,
 	'f(a_k)':F_end,
 	'a ':A_end    	
	}
	df = pd.DataFrame(dic)
	#print(df)

	


