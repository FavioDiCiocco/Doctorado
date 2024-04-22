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
from pathlib import Path
import os
import math
import time
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

########################################################################################

T=2 # Defino el número de tópicos
Etapa = Path("Func_sign") # Defino el nombre de la etapa del trabajo en la que estoy

# Defino las carpetas que voy a recorrer. Tiene más sentido definir esto a mano.
Carpetas = ["2D"]

for carp in Carpetas:
    
    # Transformo estas cosas en paths. Espero que acostumbrarme a esto valga la pena
    Direccion = Path("../{}".format(carp))
    carpeta = Path(carp)
    
    # Recorro las carpetas con datos
    CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]
    
    # Me armo una lista con los nombres de todos los archivos con datos.
    # Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos
    
    Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]
"""
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.array([5,10,12.5,15,20])
    Arr_param_y = np.array([0.2,0.5,0.7,1,1.5,1.7,2])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20)
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                 Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
            
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                if repeticion < 20:
                        
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    # Levanto los datos del archivo
                    Datos = ldata(path / nombre)
                    
                    # Leo los datos de las Opiniones Finales
                    Opifinales = np.array(Datos[5][:-1:], dtype="float")
                    
                    # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                
                    # direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                    #                                                                                                                                  ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,repeticion))
                    direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,
                                                                                                ID_param_x,PARAM_X,ID_param_y,PARAM_Y,repeticion))
                    
                    # Armo mi gráfico, lo guardo y lo cierro
                    
                    indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
                    estado = int(Frecuencias[indice])
                    
                    Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
                               "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
                               "Polarización 1D y Consenso con anchura",
                               "Polarización Ideológica con anchura", "Transición con anchura",
                               "Polarización Descorrelacionada con anchura"]
                    
                    plt.rcParams.update({'font.size': 44})
                    plt.figure(figsize=(28,21))
                    _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                                             range=[[-PARAM_X,PARAM_X],[-PARAM_X,PARAM_X]],density=True,
                                             cmap=cmap)
                    plt.xlabel(r"$x_i^1$")
                    plt.ylabel(r"$x_i^2$")
                    plt.title('Histograma 2D, {}={:.2f}_{}={:.2f}\n{}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y,Nombres[estado]))
                    plt.colorbar(im, label='Frecuencias')
                    plt.savefig(direccion_guardado ,bbox_inches = "tight")
                    plt.close()

"""

func.Tiempo(t0)
