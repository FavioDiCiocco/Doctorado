#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 10:32:08 2023

@author: favio
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.optimize import fsolve
from pathlib import Path
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

###################################################################################################

# Quiero graficar la función logística y una recta. Esto lo quiero hacer para el poster. La
# idea es graficar las dos curvas y marcar los puntos donde se cruzan.

def Logistica(X,kappa,alfa,epsilon):
    return kappa*(1/(1+np.exp(-alfa*X+epsilon)))

# Defino la ecuación dinámica que voy a usar para hallar los puntos fijos del sistema.

def Ecuacion_dinamica(x,K,A,Cdelta,Eps):
    return -x+K*(1/(1+np.exp(-A*(1+Cdelta)*x+Eps)))

# Armo mi región de valores de x para graficar

inicio = 0
final = 1

X = np.linspace(inicio,final,2000)

# Defino los parámetros para que la logística corte a la recta

Kappa = 1
Alfa = 6
Epsilon = 3

# Calculo los valores de Y de la logística y de la recta

Yl = Logistica(X,Kappa,Alfa,Epsilon)
Yr = X

# Calculo los puntos fijos de mi sistema

x0 = 0

raices = np.zeros(3)
indice = 0

while x0 < Kappa:
    
    resultado = fsolve(Ecuacion_dinamica,x0,args=(Kappa,Alfa,0,Epsilon))[0]
    
    Condicion_raiz = np.isclose(Ecuacion_dinamica(resultado,Kappa,Alfa,0,Epsilon),0,atol=1e-06)
    
    if not(np.isclose(raices,np.ones(3)*resultado).any()) and Condicion_raiz:
        
        raices[indice] = resultado
        indice += 1
    
    x0 += 0.1


# Armo el gráfico

plt.rcParams.update({'font.size': 32})
plt.figure("Dinamica1",figsize=(20,15))

# Grafico las curvas de mis términos

plt.plot(X, Yl, label = r"$\kappa \frac{1}{1+e^{-\alpha x + \epsilon}} $",color = "tab:red", linewidth=7)
plt.plot(X,Yr, label= r"x", color = "tab:gray", linewidth = 7)

# Modifico los ticks de los ejes para que se vean mejor los 
# puntos fijos

paso = 0.1
plt.xticks(np.arange(inicio,final+paso,paso))
plt.yticks(np.arange(inicio,final+paso,paso))
plt.grid()

# Grafico los puntos de los cruces entre la curva y la recta
plt.plot(raices,raices,"ok", label = "Puntos fijos",markersize=24)

# Defino los últimos detalles del sistema
plt.legend()
plt.xlabel("Interés")
plt.ylabel("Términos dinámicos")

# Guardo mi archivo
direccion_guardado = Path("../../../Imagenes/Complemento_Poster/Dinamica1.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("Dinamica1")

#-----------------------------------------------------------------------------------------

# Armo el segundo gráfico, donde la logística cruza a la recta en un sólo punto.

# Armo mi región de valores de x para graficar

inicio = 0
final = 1

X = np.linspace(inicio,final,2000)

# Defino los parámetros para que la logística corte una sola vez a la recta

Kappa = 1
Alfa = 3
Epsilon = 2

# Calculo los valores de Y de la logística y de la recta

Yl = Logistica(X,Kappa,Alfa,Epsilon)
Yr = X

# Calculo los puntos fijos de mi sistema

x0 = 0

raices = np.zeros(1)
indice = 0

while x0 < Kappa:
    
    resultado = fsolve(Ecuacion_dinamica,x0,args=(Kappa,Alfa,0,Epsilon))[0]
    
    Condicion_raiz = np.isclose(Ecuacion_dinamica(resultado,Kappa,Alfa,0,Epsilon),0,atol=1e-06)
    
    if not(np.isclose(raices,np.ones(1)*resultado).any()) and Condicion_raiz:
        
        raices[indice] = resultado
        indice += 1
    
    x0 += 0.1


# Armo el gráfico

plt.rcParams.update({'font.size': 32})
plt.figure("Dinamica2",figsize=(20,15))

# Grafico las curvas de mis términos

plt.plot(X, Yl, label = r"$\kappa \frac{1}{1+e^{-\alpha x + \epsilon}} $",color = "tab:red", linewidth=7)
plt.plot(X,Yr, label= r"x", color = "tab:gray", linewidth = 7)

# Modifico los ticks de los ejes para que se vean mejor los 
# puntos fijos

paso = 0.1
plt.xticks(np.arange(inicio,final+paso,paso))
plt.yticks(np.arange(inicio,final+paso,paso))
plt.grid()

# Grafico los puntos de los cruces entre la curva y la recta
plt.plot(raices,raices,"ok", label = "Punto fijo",markersize=24)

# Defino los últimos detalles del sistema
plt.legend()
plt.xlabel("Interés")
plt.ylabel("Términos dinámicos")

# Guardo mi archivo
direccion_guardado = Path("../../../Imagenes/Complemento_Poster/Dinamica2.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("Dinamica2")

###################################################################################################

# Ahora lo siguiente es armar el gráfico 3D con los cortes a la derecha. Mi idea es que haya
# un corte en alfa y otro en Epsilon.

# Defino las funciones que uso para calcular los puntos críticos y los Kappa

def Derivada_kappa(x,alfa,epsilon):
    return np.exp(alfa*x-epsilon)+1-alfa*x

def Kappa(x,alfa,epsilon):
    return x*( 1 + np.exp(-alfa*x +epsilon) )


# Primero armo el gráfico, colocando a la izquierda el gráfico 3D y a la derecha los cortes

plt.rcParams.update({'font.size': 45})
plt.figure("3D y cortes",figsize=(60,40))
ax3D = plt.subplot2grid((2,2),(0,0),rowspan=2,projection="3d")
axcorte1 = plt.subplot2grid((2,2),(0,1))
axcorte2 = plt.subplot2grid((2,2),(1,1))

#-------------------------------------------------------------------------------------------
# Armo el gráfico 3D
# Preparo mis variables para graficar

Epsilons = np.linspace(2,5,50)
Alfas = np.linspace(0.5,5,50)
XX,YY = np.meshgrid(Epsilons,Alfas)

# Armos las dos matrices que formarán las superficies de Kappa que voy a graficar.

Kappas_min = np.zeros(XX.shape)
Kappas_max = np.zeros(XX.shape)

# Calculo los Kappa y armo las matrices

for fila,alfa in enumerate(Alfas):
    for columna,epsilon in enumerate(Epsilons):
        
        # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
        x_min = epsilon/alfa
        
        # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
        raiz_min = fsolve(Derivada_kappa,x_min-3,args=(alfa,epsilon))
        raiz_max = fsolve(Derivada_kappa,x_min+3,args=(alfa,epsilon))
        
        # Asigno los valores de los Kappa a mis matrices
        Kappas_min[fila,columna] = Kappa(raiz_max, alfa, epsilon)
        Kappas_max[fila,columna] = Kappa(raiz_min, alfa, epsilon) 
        
# Ya tengo mis tres matrices, ahora puedo armar el gráfico de mis superficies.

# Hago unos primeros ajustes generales al gráfico

ax3D.set_xlabel(r"$\epsilon$",labelpad = 30)
ax3D.set_ylabel(r"$\alpha$",labelpad = 30)
ax3D.set_zlabel(r"$\kappa$", labelpad = 50)


# Grafico las dos superficies que encierran la región de tres puntos fijos

surf1 = ax3D.plot_surface(XX,YY,Kappas_min, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

surf2 = ax3D.plot_surface(XX,YY,Kappas_max, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

#-------------------------------------------------------------------------------------------

# Armo el gráfico del corte en alfa
# Armo mis variables a graficar

Epsilons = np.linspace(2,5,50)
Alfa = 3

# Armo mi array donde pondré los valores de Kappa, tanto los máx como los mín
Kappas_min = np.zeros(Epsilons.shape[0])
Kappas_max = np.zeros(Epsilons.shape[0])

for indice,epsilon in enumerate(Epsilons):
    
    # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
    x_min = epsilon/Alfa
    
    # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
    raiz_min= fsolve(Derivada_kappa,x_min-3,args=(Alfa,epsilon))[0]
    raiz_max = fsolve(Derivada_kappa,x_min+3,args=(Alfa,epsilon))[0]
    
    # Asigno los valores de los Kappa a mis matrices
    Kappas_min[indice] = Kappa(raiz_max, Alfa, epsilon)
    Kappas_max[indice] = Kappa(raiz_min, Alfa, epsilon)
    
# Ahora que tengo la curva hecha, la grafico y guardo el gráfico


axcorte1.set_xlabel(r"$\epsilon$")
axcorte1.set_ylabel(r"$\kappa$")
axcorte1.set_title(r"Corte en $\alpha$ = 3")
axcorte1.plot(Epsilons,Kappas_min,"--g",label=r"$\alpha$ = {}".format(Alfa), linewidth = 8)
axcorte1.plot(Epsilons,Kappas_max,"--g", linewidth = 8)
axcorte1.grid()
# axcorte1.legend()


#-------------------------------------------------------------------------------------------

# Hago el plotteo de las curvas de Kapppa
    
Alfas = np.linspace(0.5,5,80)
epsilon = 3
Kappa_min = np.zeros(Alfas.shape[0])
Kappa_max = np.zeros(Alfas.shape[0])

for indice,alfa in enumerate(Alfas):
    
    # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
    x_min = epsilon/alfa
    
    # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
    raiz_min = fsolve(Derivada_kappa,x_min-3,args=(alfa,epsilon))
    raiz_max = fsolve(Derivada_kappa,x_min+3,args=(alfa,epsilon))
    
    # Asigno los valores de los Kappa a mis matrices
    Kappa_min[indice] = Kappa(raiz_max, alfa, epsilon)
    Kappa_max[indice] = Kappa(raiz_min, alfa, epsilon)
    
# Ahora que tengo las curvas, las grafico

axcorte2.plot(Alfas,Kappa_min,"--g",label=r"$\epsilon = 3$",linewidth=8)
axcorte2.plot(Alfas,Kappa_max,"--g",linewidth=8)
axcorte2.set_xlabel(r"$\alpha$")
axcorte2.set_ylabel(r"$\kappa$")
axcorte2.set_title(r"Corte en $\epsilon$ = 3")
axcorte2.grid()
# axcorte2.legend()

#-------------------------------------------------------------------------------------------

# Guardo mi gráfico

direccion_guardado = Path("../../../Imagenes/Complemento_Poster/3Dycortes.png")
plt.savefig(direccion_guardado ,bbox_inches = "tight")
plt.close("3D y cortes")



func.Tiempo(t0)