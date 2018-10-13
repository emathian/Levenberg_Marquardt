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
	f = np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		f[i] = 0.5*(y[i]-exp(-a*x[i]))^2
	return f

def grad (x,y,a):
	g = np.zeros(np.size(x))
	for i in range(0,np.size(x)):
		g[i] = (y[i] - exp(-a*x[i]))(x[i]*exp(-a*x[i]))
	return g


X= np.arange(0,3+0.01,0.01)
y=random_data_set(X,2,0.01)
print(type(y))
fig = plt.figure() 
plt.scatter(X, y)
plt.plot(X,vg(X,2, np.size(X)), c = 'Green')
plt.show()


print(grad(X,y,2))
print('ok')