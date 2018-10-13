import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt
from math import exp, expm1
from numpy import random
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import decimal


def g(x,a):
	return exp(-1*a*x)

def vg(x,a, size_x):
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

def cost_fucntion(x,y,a):
	# size of x,y vectors are equal
	g_xa = vg(x,a, np.siez(x))
	if np.size(ylabel) == np.size(g_xa):
		SCE = 0.5*((y - g_xa)**2) 
		print(SCE)	
	else :
		print('Dimension error')		
	return f

def grad (x,y,a):
	g = np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		g[i] = (y[i] - exp(-a*x[i]))(x[i]*exp(-a*x[i]))
	return g

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
		pritn('''Representation of a data set with the following parameters :\n
			 a = 2 \n
			 x in [0,3] by 0.01
			 b = 0.01''')
	X= np.arange(0,3+0.01,0.01)
	y=random_data_set(X,2,0.01)
	fig = plt.figure() 
	plt.scatter(X, y)
	plt.plot(X,vg(X,2, np.size(X)), c = 'Green')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

if Which_question==5: 
	x= np.arange(0,3+0.01,0.01)
	y=random_data_set(X,2,0.01)
	cost_fucntion(x,y,2)