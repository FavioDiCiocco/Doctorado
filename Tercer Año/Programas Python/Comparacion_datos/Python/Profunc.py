#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.spatial.distance import jensenshannon
from pathlib import Path
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

#####################################################################################

# Armo el gráfico de las regiones del espacio de parámetros Beta-Kappa

def normal_pdf(x, mu, sigma):
    """
    Compute the probability density function (PDF) of the normal distribution.

    Parameters:
    - x (float or array-like): Input values.
    - mu (float): Mean of the normal distribution.
    - sigma (float): Standard deviation of the normal distribution.

    Returns:
    - pdf (float or array-like): Probability density function values corresponding to input x.
    """
    # Calculate the exponent term
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    
    # Calculate the normalization constant
    normalization = 1 / (sigma * np.sqrt(2 * np.pi))
    
    # Compute the PDF
    pdf = normalization * np.exp(exponent)
    
    return pdf


# Example usage:


#-------------------------------------------------------------------------
"""
# Armada la función que producel las pdf, veo de calcular la distancia
# jensen-shannon de dos distribuciones a medida que aumento la distancia
# entre ellas

mu = -5
sigma = 1
x_values = np.linspace(-10,10,10000)


pdf_fijo = normal_pdf(x_values, mu, sigma)  # Calculate PDF values


X = np.arange(-5,5,0.1)
Y = np.zeros(X.shape[0])

for i,mu_2 in enumerate(X):
    
    pdf_variable = normal_pdf(x_values, mu_2, sigma)
    Y[i] = jensenshannon(pdf_fijo,pdf_variable)


X = np.arange(0,9000,100)
Y = np.zeros(X.shape[0])

for i,cantidad in enumerate(X):
    
    pdf_variable = np.zeros(pdf_fijo.shape[0])
    pdf_variable[cantidad::] = pdf_fijo[cantidad::]
    Y[i] = jensenshannon(pdf_fijo,pdf_variable)


# Plot the PDF
plt.plot(X-mu, Y)
plt.xlabel(r'$\Delta \mu$')
plt.ylabel('Distancia')
plt.title('Distancia Jensen-Shannon entre dos distribuciones normales')
plt.grid()
plt.show()
"""

#####################################################################################

# Tomo una matriz y la roto. Repito, roto la matriz como quien gira la cara de un
# cubo Rubik, no estoy rotando el objeto que la matriz representa.
"""
def Rotar_matriz(M):
    
    # Primero miro el tamaño de la matriz que recibí
    n = M.shape[0]
    
    # Armo la matriz P que voy a returnear
    P = np.zeros(M.shape)
    
    # Giro el anillo más externo. Lo hago todo de una.
    for i in range(n):
        P[i,n-1] = M[0,i]
        P[n-1,n-1-i] = M[i,n-1]
        P[n-1-i,0] = M[n-1,n-1-i]
        P[0,i] = M[n-1-i,0]
        
    # Recursivamente mando la parte interna de la matriz M a resolverse
    # con esta misma función.
    if n > 3:
        P[1:n-1,1:n-1] = Rotar_matriz(M[1:n-1,1:n-1])
    elif n == 3:
        P[1:n-1,1:n-1] = M[1:n-1,1:n-1]
    
    return P

A = np.reshape(np.arange(25),(5,5))


print(A)
A = Rotar_matriz(A)
print(A)
A = Rotar_matriz(A)
print(A)
"""

#####################################################################################

# Create some example data
np.random.seed(0)
x = np.random.randint(0, 8, size=1000)  # Random integers between 0 and 7
y = np.random.randint(0, 8, size=1000)  # Random integers between 0 and 7

# Exclude the value '3' from the data
exclude_value = 3
mask = (x != exclude_value) & (y != exclude_value)
filtered_x = x[mask]
filtered_y = y[mask]

# Define bins excluding the value '3'
bins_x = np.arange(0, 8)
bins_x = bins_x[bins_x != exclude_value] - 0.5
bins_x = np.append(bins_x, exclude_value + 0.5)
bins_x = np.sort(bins_x)

bins_y = np.arange(0, 8)
bins_y = bins_y[bins_y != exclude_value] - 0.5
bins_y = np.append(bins_y, exclude_value + 0.5)
bins_y = np.sort(bins_y)

# Create the 2D histogram
plt.figure(figsize=(8, 6))
hist2d, xedges, yedges, im = plt.hist2d(filtered_x, filtered_y, bins=[bins_x, bins_y], cmap='inferno')

# Add a colorbar
plt.colorbar(im, label='Counts')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Histogram with Excluded Bin')

# Customize tick labels to exclude the value '3'
x_ticks = np.arange(0, 8)
x_ticks = x_ticks[x_ticks != 3]
y_ticks = np.arange(0, 8)
y_ticks = y_ticks[y_ticks != 3]

plt.xticks(x_ticks)
plt.yticks(y_ticks)

# Show the plot
plt.show()


func.Tiempo(t0)
