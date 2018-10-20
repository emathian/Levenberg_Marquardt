# -*- coding: utf-8 -*-
import matplotlib as mpl
from matplotlib import cm
from math import sqrt
from math import exp, expm1 , log
from numpy import random
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


np.random.seed()

def cost_fucntion(x,y,a):
	# size of x,y vectors are equal
	g_xa = g(x,a)
	if np.size(y) == np.size(g_xa):
		f = 0.5 * sum((y - g_xa)**2)	
	else :
		print('Dimension error')	
	return f	

def cost_fucntion2(x,y,a1, a2):
	# size of x,y vectors are equal
	g_xa = g2(x,a1, a2)
	if np.size(y) == np.size(g_xa):
		f = 0.5 * sum((y - g_xa)**2)
	else :
		print('Dimension error')	
	return f	



def derivative_2 (x,a,l):	
		return sum((-x*np.exp(-1*a*x))**2) *(1+l)


def derivative_2_f2(x,y,a1,a2, l):	

	HLM = np.zeros((2, 2))
	HLM[0][0] = sum ((np.log(x)*x**a1 * np.exp(-a2*x))**2) *(1+l)
	HLM[0][1] = sum ((np.log(x)*x**a1 * np.exp(-a2*x))*(-x**(a1 +1 )*np.exp(-a2*x)) )
	HLM[1][0] = sum((np.log(x)*x**a1 * np.exp(-a2*x))*(-x**(a1 +1 )*np.exp(-a2*x)) )
	HLM[1][1] =sum((-x**(a1 +1 )*np.exp(-a2*x))**2) * (1+ l)

	return HLM



def g(x,a):
	y  = np.exp(-1*a*x)
	return y	


def g2(x,a1, a2):
	y = x**a1* exp(-1*a2*x)
	return y


def grad (x,y,a):
	G = sum((y- np.exp(-1*a*x))*(x*np.exp(-1*a*x)))
	return G


def grad2 (x,y,a1,a2):

	df_da1 = -1 * sum((y -( x**a1*np.exp(-a2*x)))	* np.log(x)* x**a1 * np.exp(-a2 *x) )
	df_da2 = -1 * sum((y -( x**a1*np.exp(-a2*x)))	* -x**(a1+1)*np.exp(-a2*x) )
	norm_grad = sqrt(df_da1 **2  + df_da2 **2  )

	return df_da1 , df_da2 , norm_grad


def LM (x,y,a,l,cond, k, g):
	f0 = cost_fucntion(x,y,a)
	f00 = cost_fucntion(x,y,a) -1 # atificial initiation for cond 3
	nb_iter =0
	L =[l]
	G =[abs(grad(x,y,a))]
	F= [f0]
	A = [a]
	c_stop = stop(cond,k,nb_iter,g,G[-1],  f0, f00)	
	# def stop (cond, k, nb_iter, g, current_grad , fk , fkk):
	while c_stop==True:
		GG= abs(grad(x,y,A[-1] ) )
		dLM = -1 * (grad(x,y,A[-1] )/ (derivative_2(x ,A[-1], L[-1] )) )
		next_f = cost_fucntion(x,y, A[-1] + dLM)
		if next_f <  F[-1] :
			AA =  A[-1] + dLM # a_k+1
			LL = L[-1]/10
			
		else :
			AA  = A[-1]
			LL = L[-1]*10
			
		L.append(LL)
		G.append(GG)
		F.append(next_f)
		A.append(AA)
		nb_iter +=1
		c_stop = stop(cond,k,nb_iter,g,G[-1], F[-1], next_f)	
			
	return L, G, F, A

def LM2 (x,y,a1, a2,l,cond, k, g):
	f0 = cost_fucntion2(x,y,a1,a2)
	f00 = f0 -1 # atificial initiation for cond 3
	nb_iter =0
	L =[l]
	G =[ grad2(x,y,a1,a2)[2]   ]
	F= [f0]
	A1 = [a1]
	A2 = [a2]
	c_stop = stop(cond,k,nb_iter,g,G[-1],  f0, f00)	

	while c_stop==True:
		GG= grad2(x,y,A1[-1],A2[-1])[2] 

		v_grad = np.zeros((2,1))
		v_grad[0][0] =  -1*(grad2(x,y,A1[-1],A2[-1])[0])
		v_grad[1][0] =  -1*(grad2(x,y,A1[-1],A2[-1])[1])
		dLM = np.dot( np.linalg.inv( derivative_2_f2(x,y,A1[-1], A2[-1] , L[-1]) ) , v_grad )
		
		next_f = cost_fucntion2(x,y, A1[-1] + dLM[0] , A2[-1] + dLM[1])
		if next_f < F[-1] :  ######## A REFLECHIR
			AA1 =  A1[-1] + dLM[0] # a_k+1
			AA2 = A2[-1] +dLM[1]
			LL = L[-1]/10
		else :
			AA1  = A1[-1]
			AA2  = A2[-1]
			LL = L[-1]*10
			
		L.append(LL)
		G.append(GG)
		F.append(next_f)
		A1.append(AA1)
		A2.append(AA2)
		nb_iter +=1
		c_stop = stop(cond,k,nb_iter,g,G[-1], F[-1], next_f)	
			
	return L, G, F, A1 , A2 	


def random_data_set(x,a,b) : # B est l'amplitude du bruit

	y=np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		current_x = x[i]
		y[i] = g(current_x, a) + b*np.random.normal(0,1, 1)	
	return (y)	

def random_data_set2(x,a1,a2,b) : # B est l'amplitude du bruit

	y=np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		y[i] = g2(x[i], a1, a2) + b*np.random.normal(0,1, 1)	
	return (y)	

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
		c_stop = current_grad > g
	elif cond==2 :
		c_stop =  nb_iter < k and current_grad > g
	elif cond==3 :
		c_stop = fkk < fk
			
	else:
		return 'impossible'	
	return c_stop				
	



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
	X= np.arange(0,3+0.01,0.01)
	print(g(0,X))
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
	y=random_data_set(X,2,0.1)
	fig = plt.figure() 
	plt.scatter(X, y)
	plt.plot(X,g(X,2), c = 'black', linewidth=2)
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
	y_fit = g(x,LMf1[3][-1])
	print(LMf1[2][0])
	print(LMf1[2][-1])
	print(LMf1[3][-1])
	fig = plt.figure() 
	plt.scatter(x, y)
	plt.plot(x,g(x,2), c='black')
	plt.plot(x, y_fit, c='red')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.ylim(-0.5,1.5)
	plt.show()
	

if Which_question==9:
	x= np.arange(0,3+0.01,0.01)


	B = [0.01]
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	c =0
	C = 0.2
	for i in B :
		col = (0, C, 0.6) 
		C += 0.2

		y=random_data_set(x,2,i)
		S =LM (x,y,1.5,0.001, 0 , 50, 0.001)
		L_end.append (S[0])
		G_end.append (S[1])
		F_end.append(S[2])
		A_end.append(S[3])

		fig = plt.figure(c) 
		
		fig.suptitle(i, fontsize=16)
 
		plt.subplot(121)
		plt.plot(range(51), S[0], c=col)
		plt.xlabel('k')
		plt.ylabel('lambda')

		plt.subplot(122)
		plt.plot(range(51), S[1], c=col)
		plt.yscale('log')
		plt.xlabel('k')
		plt.ylabel('|g|')

		c+= 1
	
	
	fig = plt.figure(1) 

	plt.plot(range(51), F_end[0],label='b=0.01' ,c=(0, 0.2, 0.6) )
	plt.yscale('log')
	plt.legend() #adds a legend
	mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	


	y_fit = g(x,S[3][-1])
	
	fig = plt.figure(2) 
	plt.scatter(x, y)
	plt.plot(x,g(x,2), c='black')
	plt.plot(x, y_fit, c='red')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.ylim(-0.5,1.5)
	plt.show()
	
	print(S[3])
		


if Which_question==10 :
	x= np.arange(0,3+0.01,0.1)
	
	# B1 = list(np.arange(0, 1.2, 0.2))
	# B2 = list(np.arange(0, 6, 1))
	# B = B1 + B2[2:]
	B = list(np.arange(0, 0.3, 0.01))
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	for i in B :
		y=random_data_set(x,2,i)
		S =LM (x,y,1.5,0.001, 2, 30, 0.001)
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






	B = [0.01, 0.1 , 0.5,1]
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	c =0
	C = 0.2
	for i in B :
		col = (0, C, 0.6) 
		C += 0.2

		y=random_data_set(x,2,i)
		S =LM (x,y,1.5,0.001, 0 , 30, 0.001)
		L_end.append (S[0])
		G_end.append (S[1])
		F_end.append(S[2])
		A_end.append(S[3])

		fig = plt.figure(c) 
		
		fig.suptitle(i, fontsize=16)
 
		plt.subplot(121)
		plt.plot(range(31), S[0], c=col)
		
		plt.xlabel('k')
		plt.ylabel('lambda')

		plt.subplot(122)
		plt.plot(range(31), S[1], c=col)
		plt.yscale('log')
		plt.xlabel('k')
		plt.ylabel('|g|')
	

		
		
		c+= 1
	
	
	fig = plt.figure(5) 

	plt.plot(range(31), F_end[0],label='b=0.01' ,c=(0, 0.2, 0.6) )
	plt.plot(range(31), F_end[1], label='b=0.1' ,c=(0, 0.4, 0.6) )
	plt.plot(range(31), F_end[2], label='b=0.5' ,c=(0, 0.6, 0.6) )
	plt.plot(range(31), F_end[3], label='b=1' ,c=(0, 0.8, 0.6) )
	plt.legend() #adds a legend
	mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size
	plt.show()
	print(F_end)
	print(A_end)



	print(LMf1[0])
	print(len(LMf1[0]))
	print(range(62))



	fig = plt.figure(1) 
	plt.subplot(131)
	plt.plot(range(61), LMf1[0])
	#plt.ylim(-0.5,2)
	plt.xlabel('k')
	plt.ylabel('lambda')

	plt.subplot(132)
	plt.plot(range(61), log(LMf1[1]))
	#plt.ylim(0,0.002)
	plt.xlabel('k')
	plt.ylabel('|g|')


	plt.subplot(133)
	plt.plot(range(61), log(LMf1[2]))
	#plt.ylim(0,0.002)
	plt.xlabel('k')
	plt.ylabel('f')

	plt.show()


if Which_question==11 :
	x= np.arange(0,3+0.01,0.01)
	
	# B1 = list(np.arange(0, 1.2, 0.2))
	# B2 = list(np.arange(0, 6, 1))
	# B = B1 + B2[2:]
	B = list(np.arange(0, 10, 0.5))
	L_end  =[]
	G_end = []
	F_end = []
	A_end =[]
	B_end = []
	fig, axes = plt.subplots(nrows=len(B)/2, ncols=2,  sharex=True, sharey=True)
	for i, ax in enumerate(axes.flatten()):

		y=random_data_set(x,2,B[i])
		S =LM (x,y,1.5,0.001, 0, 10000, 0.01)
		L_end.append (S[0][-1])
		G_end.append (S[1][-1])
		F_end.append(S[2][-1])
		A_end.append(S[3][-1])
		B_end.append(B[i])
		y_fit = g(x,S[3][-1])
	# 	plt.subplot(sub+c)
		ax.scatter(x, y, s=0.6 )
		ax.plot(x,g(x,2), c='black')
		ax.plot(x, y_fit, c='red')
		ax.set_title(B[i])

	plt.ylim(-0.2,2)
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
	print(df)

if Which_question==12 :
	print (' function g  returns the result of g(x)= x**a1 e**(-a2 x) with x=1, a1=1  and a2=1 		:', g2(1, 1, 1))
	print ( ' We initialize a data set according the function random_data_set2(x,a1,a2,b) such taht y_i = g2 (A, x) + b N(0,1)')
	print( ' cost_fucntion2  return the function f such that f(a) = 0.5 sum ((yi -xi**a1 e^(-a2 xi) )**2)' )
	


	x= np.arange(0,5+0.01,0.01)
	y=random_data_set2(x, 2, 3, 0.01 )
	fig = plt.figure() 
	plt.scatter(x, y)
	plt.plot(x,g2(x,2,3), c = 'red')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()



if Which_question==13 :
	x= np.arange(0,5+0.01,0.01)
	y=random_data_set2(x, 2, 3, 0.01 )
	a1 = 2
	a2 = 3
	print (' We clculate the gradient of function 2 with a1 = 2, a2 = 3 , x  in (0,5)	: \n', 
		   '- df / da1   = ' ,grad2(x,y,a1,a2)[0] , '\n',
		   '- df / da2   = ' ,grad2(x,y,a1,a2)[1] , '\n',
		   '- |g|   = ' ,grad2(x,y,a1,a2)[2] , '\n')

if Which_question==14 :

	# def derivative_2_f2(x,y,a1,a2, l):	
	print ('''Test of  derivative_2_f2  with:  \n
		       -  x in (0,5) \n
		       -  y generated with b = 0.01 \n
		       -  a1  = 2 \n
		       -  a2  = 3 \n
		       -  l = 0.001  \n ''')


	x= np.arange(0,5+0.01,0.01)
	y= random_data_set2(x, 2, 3, 0.01 )
	
	a1 = 2
	a2 = 3
	l = 0.001
	print (derivative_2_f2(x,y,a1,a2, l))

if Which_question==15 :
	x= np.arange(0.5,5+0.01,0.01)
	y=random_data_set2(x, 2, 3, 0.01 )
	a1 = 1.5
	a2 = 1.5
	l = 0.00001
	
	sol = LM2(x,y,a1,a2 ,l, 1 , 30 ,0.001)
	s = pd.DataFrame()
	s["Lambda"] = sol[0]
	s["Norm Grad "] = sol[1]
	s["Cost function F"] = sol[2]
	s["a1 "] = sol[3]
	s["a2 "] = sol[4]
	print(s)

	fig = plt.figure() 
	plt.scatter(x, y)
	plt.plot(x,g2(x,2,3),label = 'Theoric curve' ,c = 'black',  linewidth=3.0)
	plt.plot(x,g2(x,sol[3][-1],sol[4][-1]), label = 'learning curve', c = 'red')
	plt.legend()
	mpl.rcParams['legend.fontsize'] = 10 
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()


if Which_question==16 :
	pass

# LM2 (x,y,a1, a2,l,cond, k, g)	return L, G, F, A1 , A2 	

