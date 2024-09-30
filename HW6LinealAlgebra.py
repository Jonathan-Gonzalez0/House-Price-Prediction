'''
Created on Saturday April 6 01:41:26 2024

@author: Frank A. Segui Gonzalez, frank.segui1@upr.edu
@author: Jonathan Gonzalez Rodriguez, jonathan.gonzalez57@upr.edu

INGE 4035
Asignación 6

Parte 1:
Programa que genera un modelo para hacer una aproximación de precio de casas 
a base de una data de precios respecto a sus cantidad de cuartos y área utilizando 
ecuaciones normales de algebra lineal.
Se generan adicionalmente unas gráficas para visualizar los impactos de cada feature.

Última actualizacion 4/8/2024 
'''

import numpy as np
import matplotlib.pyplot as plt

filename = "house.txt"

TS = np.loadtxt(filename, delimiter=',')

x = TS[:,:2]

y = TS[:,2]

m,nf = np.shape(x)

A = np.vstack((np.ones(m),x[:,0],x[:,1])).T

w = np.linalg.pinv(A)@y

b = w[0]

w1 = w[1]

w2 = w[2]

myFun = w2*x[:,1] + w1*x[:,0] + b

linealIDX = np.argsort(myFun)

R2 = 1-np.sum((y-myFun)**2)/np.sum((y-np.average(y))**2)

plt.close('all')
plt.figure(figsize = (20,15))
plt.subplot(121)
plt.title("USD ($) VS Area (ft^2)", size = 35, weight = 'bold')
plt.plot(x[:,0], y, 'o', color = "dimgray", label = "Data Points")
plt.plot(x[:,0], myFun, "o", color = "red", label = 'Linear Regression, R2 = 0.73')
plt.xlabel("area (ft^2)", fontsize = 18, weight = "bold")
plt.ylabel("USD ($)", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()	


plt.subplot(122)
plt.title("USD ($) VS Rooms", size = 35, weight = 'bold')
plt.plot(x[:,1], y, 'o', color = "dimgray", label = "Data Points")
plt.plot(x[:,1], myFun, "o", color = "red", label = 'Linear Regression, R2 = 0.73')
plt.xlabel("Rooms", fontsize = 18, weight = "bold")
plt.ylabel("USD ($)", fontsize =18, weight = "bold")
plt.legend(fontsize=25)
plt.tight_layout()	

plt.figure()
plt.plot(y[linealIDX], 'o', color = "dimgray", label = "Data Points")
plt.plot(myFun[linealIDX], "-", color = "red", label = 'Linear Regression')
plt.legend(fontsize=25)
plt.tight_layout()

plt.figure(figsize = (20,15))
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],y,'o',color = 'navy',mfc='white')
ax.plot3D(x[:,0],x[:,1],myFun,'o',color = 'red')
ax.set_xlabel('Areas (ft^2)', weight = "bold", fontsize = 18); ax.set_ylabel('Rooms', weight = "bold", fontsize = 18); ax.set_zlabel('USD ($)', weight = "bold", fontsize = 18)

plt.figure(figsize = (20,15))
ax = plt.axes(projection='3d')
ax.plot3D(x[:,0],x[:,1],y,'o',color = 'navy',mfc='white')
ax.set_xlabel('Areas (ft^2)', weight = "bold", fontsize = 18); ax.set_ylabel('Rooms', weight = "bold", fontsize = 18); ax.set_zlabel('USD ($)', weight = "bold", fontsize = 18)


predict = lambda area, rooms: w1*area + w2*rooms + b

v = predict(3330,3)

print(round(v))