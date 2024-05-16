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

"""
X = np.arange(0,9000,100)
Y = np.zeros(X.shape[0])

for i,cantidad in enumerate(X):
    
    pdf_variable = np.zeros(pdf_fijo.shape[0])
    pdf_variable[cantidad::] = pdf_fijo[cantidad::]
    Y[i] = jensenshannon(pdf_fijo,pdf_variable)
"""

# Plot the PDF
plt.plot(X-mu, Y)
plt.xlabel(r'$\Delta \mu$')
plt.ylabel('Distancia')
plt.title('Distancia Jensen-Shannon entre dos distribuciones normales')
plt.grid()
plt.show()



func.Tiempo(t0)
