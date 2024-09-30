# -*- coding: utf-8 -*-
'''
Created on Saturday April 6 01:41:26 2024

@author: Frank A. Segui Gonzalez, frank.segui1@upr.edu
@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

INGE 4035
Asignación 6

Parte 1:
Programa que genera un modelo para hacer una aproximación de precio de casas 
a base de una data sin normalizar de precios respecto a sus cantidad de cuartos y área.
Se generan adicionalmente unas gráficas para visualiazar los impactos de cada feature.

Última actualizacion 4/8/2024 
'''
import numpy as np
import matplotlib.pyplot as plt

filename = "house.txt"

TS = np.loadtxt(filename, delimiter=',')

y = TS[:,2]

x = TS[:,:2]

m,nf = np.shape(x)

w = np.zeros((1,nf))

b = 0

alpha = 0.0000002

niter = 1000000

wv = np.zeros((2,niter), dtype = float).T

bv = np.zeros(niter, dtype = float)

MSEv = np.zeros(niter, dtype = float)

ym = w@x.T + b

MSE = np.sum((y-ym)**2)/m

MSEv[0] = MSE

epsilon = 1e-5

for k in range(1,niter):
    w = w - alpha*(2/m)*(ym-y)@(x)
    b = b - alpha*(2/m)*np.sum((ym-y)*1)
    ym = b + w@x.T
    MSEv[k] = np.sum((y-ym)**2)/m
    wv[k,:] = w
    bv[k] = b
    tolc = (MSEv[k-1]-MSEv[k])/MSEv[k]
    
    if tolc<0:
        print("Error is increasing at %i iterations" %k)
        break
    elif tolc<epsilon:
        print("Target tolerance reached at %i iterations" %k)
        break
else:
    print("Target tolerance was not reached within maximun of iterations")
        
wv = wv[:k+1,:]
bv = bv[:k+1]
MSEv = MSEv[:k+1]

linealIDX = np.argsort(ym[0,:])

plt.close('all')
plt.figure(figsize = (20,15))
plt.subplot(121)
plt.title("USD ($) VS Area (ft^2)", size = 35, weight = 'bold')
plt.plot(x[:,0], y, 'o', color = "dimgray", label = "Data Points")
plt.plot(x[:,0], ym[0,:], "-o", color = "red", label = 'Linear Regression')
plt.xlabel("area (ft^2)", fontsize = 18, weight = "bold")
plt.ylabel("USD ($)", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()	


plt.subplot(122)
plt.title("USD ($) VS Rooms", size = 35, weight = 'bold')
plt.plot(x[:,1], y, 'o', color = "dimgray", label = "Data Points")
plt.plot(x[:,1], ym[0,:], "o", color = "red", label = 'Linear Regression')
plt.xlabel("Rooms", fontsize = 18, weight = "bold")
plt.ylabel("USD ($)", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()	


plt.figure()
plt.plot(y[linealIDX], 'o', color = "dimgray", label = "Data Points")
plt.plot(ym[0,linealIDX], "-", color = "red", label = 'Linear Regression')
plt.legend(fontsize=25)
plt.tight_layout()	



plt.figure(figsize = (20,15))
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],y,'o',color = 'navy',mfc='white')
ax.plot3D(x[:,0],x[:,1],ym[0,:],'o',color = 'red')
ax.set_xlabel('Areas (ft^2)', weight = "bold", fontsize = 18); ax.set_ylabel('Rooms', weight = "bold", fontsize = 18); ax.set_zlabel('USD ($)', weight = "bold", fontsize = 18)


plt.figure(figsize = (20,15))
plt.title("Cost VS Iterations", size = 35, weight = 'bold')
plt.semilogy(MSEv, '-o', color = "red", label = f"Learning Curve = {alpha:.7f}")
plt.xlabel("Iterations", fontsize = 18, weight = "bold")
plt.ylabel("Cost", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()


plt.figure(figsize = (20,15))
plt.subplot(131)
plt.title("Cost VS W1", size = 35, weight = 'bold')
plt.semilogy(wv[:,0],MSEv, '-o', color = "red")
plt.xlabel("W1", fontsize = 18, weight = "bold")
plt.ylabel("Cost", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()
plt.subplot(132)
plt.title("Cost VS W2", size = 35, weight = 'bold')
plt.semilogy(wv[:,1],MSEv, '-o', color = "red")
plt.xlabel("W2", fontsize = 18, weight = "bold")
plt.ylabel("Cost", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()
plt.subplot(133)
plt.title("Cost VS b", size = 35, weight = 'bold')
plt.semilogy(bv,MSEv, '-o', color = "red")
plt.xlabel("b", fontsize = 18, weight = "bold")
plt.ylabel("Cost", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()

predict = lambda area, rooms: w[0,0]*area + w[0,1]*rooms + b

v = predict(3330,3)

print(round(v))








    