# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:00 2022

@author: Favio
"""

# Este archivo es para definir funciones

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import jensenshannon
import pandas as pd
import numpy as np
import time
import math
from pathlib import Path

##################################################################################
##################################################################################

# FUNCIONES GENERALES

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
#--------------------------------------------------------------------------------
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo(t0):
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))

#--------------------------------------------------------------------------------

# Esta es la función que uso por excelencia para levantar datos de archivos. Lo
# bueno es que lee archivos de forma general, no necesita que sean csv o cosas así
def ldata(archive):
    with open(archive) as f:
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data

#--------------------------------------------------------------------------------    
    
def data_processor(x):
    if(isinstance(x, int)):
        return x
    elif(isinstance(x, float)):
        return int(x)
    elif(isinstance(x, str)):
        return int(x[0]) if(x[0]!="-" and int(x[0])<9) else 0
    elif(x.isnan()):
        return 0
    else:
        print("Error, no se ha identificado el tipo: {}".format(type(x)))

##################################################################################
##################################################################################

# FUNCIONES GRAFICADORAS

##################################################################################
##################################################################################

# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la varianza de las opiniones como métrica.

def Mapa_Colores_Tiempo_convergencia(DF,path,carpeta,
                                    SIM_param_x,SIM_param_y,
                                    ID_param_extra_1):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((3,XX.shape[0],XX.shape[1]))
    
    #--------------------------------------------------------------------------------
    
    for EXTRAS in Arr_EXTRAS:
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
        
            # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
            Tiempos = np.zeros(len(archivos))
            
            #-----------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Pasos simulados
                # Semilla
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                
                # Leo los datos de las Opiniones Finales
                Tiempos[repeticion] = int(Datos[7][0])
                
            #------------------------------------------------------------------------------------------
            # Con los "tiempos" de las simulaciones calculo la fracción de estados que llegaron hasta el final
            # de la simulación, así la variación de esos tiempos de simulación
            
            ZZ[0,Arr_param_y.shape[0]-1-fila,columna] = np.count_nonzero(Tiempos==200000) / Tiempos.shape[0]
            ZZ[1,Arr_param_y.shape[0]-1-fila,columna] = np.mean(Tiempos/200000)
            ZZ[2,Arr_param_y.shape[0]-1-fila,columna] = np.var(Tiempos/200000)
        
    #--------------------------------------------------------------------------------
    
        # Una vez que tengo los ZZ completos, armo mis mapas de colores
        
        #--------------------------------------------------------------------------------
        # Fracción de estados oscilantes
        direccion_guardado = Path("../../../Imagenes/{}/Fraccion Oscilantes_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("FraccionOscilantes",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ[0,:,:],shading="nearest", cmap = "magma")
        plt.colorbar()
        plt.title("Fraccion de estados Oscilantes")
    
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("FraccionOscilantes")
        
        #--------------------------------------------------------------------------------
        
        # Promedio de cantidad de pasos de simulación
        direccion_guardado = Path("../../../Imagenes/{}/Promedio Pasos_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("PromedioPasos",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ[1,:,:],shading="nearest", cmap = "viridis")
        plt.colorbar()
        plt.title("Promedio normalizado de pasos de simulación")
    
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("PromedioPasos")
        
        #--------------------------------------------------------------------------------
        
        # Varianza de cantidad de pasos de simulación
        direccion_guardado = Path("../../../Imagenes/{}/Varianza Pasos_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("VarianzaPasos",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ[2,:,:],shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.title("Varianza normalizada de pasos de simulación")
    
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("VarianzaPasos")


#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Graf_Histograma_opiniones_2D(DF,path,carpeta,bins,cmap,
                                 ID_param_x,ID_param_y,ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_x = Arr_param_x[(Arr_param_x >= 0) & (Arr_param_x <= 0.04)]
    
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_param_y = Arr_param_y[(Arr_param_y >= 0.4) & (Arr_param_y <= 0.8)]
    
    
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
    Dic_Total = Diccionario_metricas(DF,path, 20, 20)
    
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
                if repeticion < 10:
                
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    # Levanto los datos del archivo
                    Datos = ldata(path / nombre)
                    
                    # Leo los datos de las Opiniones Finales
                    Opifinales = np.array(Datos[5][:-1:], dtype="float")
                    Opifinales = (Opifinales/EXTRAS)*bins[-1]
                    X_0 = Opifinales[0::T]
                    Y_0 = Opifinales[1::T]
                    
                    # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                
                    # direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                                # ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,repeticion))
                    
                    direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,
                                                                                                ID_param_x,PARAM_X,ID_param_y,PARAM_Y,repeticion))
                    
                    indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
                    estado = int(Frecuencias[indice])
                    
                    Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
                               "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
                               "Polarización 1D y Consenso con anchura",
                               "Polarización Ideológica con anchura", "Transición con anchura",
                               "Polarización Descorrelacionada con anchura"]
                    
                    X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    
                    # Armo mi gráfico, lo guardo y lo cierro
                    
                    plt.rcParams.update({'font.size': 32})
                    plt.figure(figsize=(20,15))
                    # _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                    #                          range=[[-EXTRAS,EXTRAS],[-EXTRAS,EXTRAS]],density=True,
                    #                          cmap=cmap)
                    _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
                    plt.xlabel(r"$x_i^1$")
                    plt.ylabel(r"$x_i^2$")
                    plt.title('Histograma 2D, {}={:.2f}_{}={:.2f}\n{}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y,Nombres[estado]))
                    plt.colorbar(im, label='Fracción')
                    plt.savefig(direccion_guardado ,bbox_inches = "tight")
                    plt.close()

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Mapa_Colores_Traza_Covarianza(DF,path,carpeta,SIM_param_x,SIM_param_y,
                                  ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF["Extra"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Extra"]==EXTRAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        Covarianzas = np.zeros(archivos.shape[0])
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.zeros((T,AGENTES))
            
            # Normalizo mis datos usando el valor de Kappa
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/EXTRAS
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
            
            M_cov = np.cov(Opifinales)
            Covarianzas[repeticion] = np.trace(M_cov)/T
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Covarianzas)
            
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Varianzas_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Traza_Covarianza",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Varianzas en Espacio de Parametros")
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Traza_Covarianza")
    


#-----------------------------------------------------------------------------------------------
    
# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Mapa_Colores_Covarianzas(DF,path,carpeta,SIM_param_x,SIM_param_y,
                             ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF["Extra"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Extra"]==EXTRAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        Covarianzas = np.zeros(archivos.shape[0])
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.zeros((T,AGENTES))
            
            # Normalizo mis datos usando el valor de Kappa
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/EXTRAS
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
            
            M_cov = np.cov(Opifinales)
            Covarianzas[repeticion] = M_cov[0,1]
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Covarianzas)
            
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Covarianzas_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Covarianzas",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Covarianza en Espacio de Parametros")
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Covarianzas")
    

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Diccionario_metricas(DF, path, Nx, Ny):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    Salida = dict()
    for EXTRAS in Arr_EXTRAS:
        Salida[EXTRAS] = dict()
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            Varianza_X = np.zeros(archivos.shape[0])
            Varianza_Y = np.zeros(archivos.shape[0])
            Covarianza = np.zeros(archivos.shape[0])
            Promedios = np.zeros(archivos.shape[0])
            Entropia = np.zeros(archivos.shape[0])
            Identidad = np.zeros(archivos.shape[0], dtype=int)
            
            for indice,nombre in enumerate(archivos):
                
        
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
        
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
        
                # Leo los datos de las Opiniones Finales
                Opifinales = np.zeros((T,AGENTES))
        
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                Identidad[indice] = repeticion
        
                M_cov = np.cov(Opifinales, bias = True)
                Varianza_X[indice] = M_cov[0,0]
                Varianza_Y[indice] = M_cov[1,1]
                Covarianza[indice] = M_cov[0,1]
                Promedios[indice] = np.linalg.norm(np.array(Datos[5][:-1:], dtype="float"),ord=1) / np.array(Datos[5][:-1:], dtype="float").shape[0]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales/EXTRAS
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,Nx, Ny,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                Entropia[indice] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
                
            #----------------------------------------------------------------------------------------------------------------------
            
            # Mis datos no están ordenados, pero con esto los ordeno según el
            # valor de la simulación. Primero inicializo el vector que tiene los índices
            # de cada simulación en sus elementos. El elemento 0 tiene la ubicación
            # de la simulación cero en los demás vectores.
            Ubicacion = np.zeros(max(Identidad)+1,dtype = int)
            
            # Para cada elemento en el vector de identidad, le busco su indice en el
            # vector y coloco ese índice en el vector de Ubicacion en la posición
            # del elemento observado
            for i in np.unique(Identidad):
                indice = np.where(Identidad == i)[0][0]
                Ubicacion[i] = indice
            
            # Ahora tengo que remover las simulaciones faltantes. Armo un vector
            # que tenga sólamente los índices de las simulaciones faltantes
            Faltantes = np.arange(max(Identidad)+1)
            Faltantes = np.delete(Faltantes,Identidad)
            
            # Borro esas simulaciones de mi vector de Ubicacion
            Ubicacion = np.delete(Ubicacion,Faltantes)
                
                
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            if PARAM_Y not in Salida[EXTRAS][PARAM_X].keys():
                Salida[EXTRAS][PARAM_X][PARAM_Y] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Entropia"] = Entropia[Ubicacion] / np.log2(Nx*Ny)
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"] = Varianza_X[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"] = Varianza_Y[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"] = Covarianza[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Promedios"] = Promedios[Ubicacion]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] = np.unique(Identidad)
            
    return Salida

#-----------------------------------------------------------------------------------------------

def Identificacion_Estados(Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
    
    Resultados = np.zeros(len(Entropia))
    
    for i,ent,sx,sy,cov,prom in zip(np.arange(len(Entropia)),
                                    Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
        
        # Reviso la entropía y separo en casos con y sin anchura
        
        if ent <= 0.3:
            
            # Estos son casos sin anchura
            
            if sx < 0.5 and sy < 0.5:
                
                # Caso de un sólo extremo
                
                # Consenso neutral
                if prom < 0.1:
                    Resultados[i] = 0
                
                # Consenso radicalizado
                else:
                    Resultados[i] = 1
                    
            
            # Casos de dos extremos
            elif sx >= 0.5 and sy < 0.5:
                # Dos extremos horizontal
                Resultados[i] = 2
            elif sx < 0.5 and sy >= 0.5:
                # Dos extremos vertical
                Resultados[i] = 2
                
            else:
                if np.abs(cov) > 0.6:
                    # Dos extremos ideológico
                    Resultados[i] = 3
                elif np.abs(cov) < 0.3:
                    # Cuatro extremos
                    Resultados[i] = 5
                else:
                    # Estados de Transición
                    Resultados[i] = 4
        
        else:
            
            # Estos son los casos con anchura
            
            # Casos de dos extremos
            if sx >= 0.3 and sy < 0.3:
                # Dos extremos horizontal
                Resultados[i] = 6
            elif sx < 0.3 and sy >= 0.3:
                # Dos extremos vertical
                Resultados[i] = 6
            
            else:
                # Polarización
                # Polarización ideológica
                if np.abs(cov) >= 0.25:
                    Resultados[i] = 7
                
                # Transición con anchura
                elif np.abs(cov) >= 0.15 and np.abs(cov) < 0.25:
                    Resultados[i] = 8
                
                # Polarización descorrelacionada
                else:
                    Resultados[i] = 9
                
    return Resultados


#-----------------------------------------------------------------------------------------------
# Esta función arma todos los mapas de colores de frecuencias de los estados finales.    

def Mapas_Colores_FEF(DF,path,carpeta,SIM_param_x,SIM_param_y,
                      ID_param_extra_1):
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF["Extra"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]

    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    # Voy a armar 11 mapas de colores
    ZZ = np.zeros((10,XX.shape[0],XX.shape[1]))
    
    #--------------------------------------------------------------------------------
    
    # Diccionario con la entropía, Sigma_x y Sigma_y de todas las simulaciones
    # para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20,20)
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
                
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        for grafico in range(10):
            ZZ[grafico,(Arr_param_y.shape[0]-1)-fila,columna] = np.count_nonzero(Frecuencias == grafico)/Frecuencias.shape[0]
            
    #--------------------------------------------------------------------------------
    
    Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
               "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
               "Polarización 1D y Consenso con anchura",
               "Polarización Ideológica con anchura", "Transición con anchura",
               "Polarización Descorrelacionada con anchura"]
    
    for grafico in range(10):
        # Una vez que tengo el ZZ completo, armo mi mapa de colores
        direccion_guardado = Path("../../../Imagenes/{}/Fracción estados finales {}_{}={}.png".format(carpeta,Nombres[grafico],ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("FEF",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ[grafico],shading="nearest", cmap = "plasma")
        plt.colorbar()
        plt.title("Fracción de estados de {}".format(Nombres[grafico]))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("FEF")

#-----------------------------------------------------------------------------------------------

def Clasificacion(Array, Nx, Ny,T):
    
    # Recibo un array de opiniones que van entre [-1,1]. Le sumo 1
    # para que las opiniones vayan entre [0,2].
    Array = Array+1
    
    # Divido mi espacio de tópicos 2D en cuadrados. Defino el ancho
    # de esos cuadrados.
    ancho_x = 2/Nx
    ancho_y = 2/Ny
    
    # Armo un array de tuplas que indiquen "fila" y "columna" en la cuál
    # cae cada opinión.
    Ubicaciones = np.array([(math.floor(x/ancho_x),math.floor(y/ancho_y)) for x,y in zip(Array[0::T],Array[1::T])])
    
    # Ahora me armo mi array de distribución, que cuenta cuántas opiniones tengo
    # por cada cajita.
    Distribucion = np.zeros((Nx*Ny))
    for opinion in Ubicaciones:
        # Tomo mínimos para que no intente ir a una cajita no existente. Tendría un problema
        # si algún agente tiene opinión máxima en algún tópico.
        fila = min(opinion[0],Nx-1)
        columna = min(opinion[1],Ny-1)
        Distribucion[fila*Nx+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Calculo_Entropia(DF,path,Nx,Ny):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    
    Salida = dict()
    for EXTRAS in Arr_EXTRAS:
        Salida[EXTRAS] = dict()
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            entropias = np.zeros(archivos.shape[0])
            
            for nombre in archivos:
        
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
        
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
        
                # Leo los datos de las Opiniones Finales
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales / EXTRAS
                
                # La nueva función de Entropía que armé normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
        
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
        
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,Nx,Ny,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                entropias[repeticion] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
        
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y] = entropias/np.log2(Nx*Ny)
    
    return Salida

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la entropía como métrica.

def Mapa_Colores_Entropia_opiniones(DF,path,carpeta,SIM_param_x,SIM_param_y,
                                    ID_param_extra_1):
    
    # Defino los arrays de parámetros diferentes    
    EXTRAS = int(np.unique(DF["Extra"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((2,XX.shape[0],XX.shape[1]))
    
    #--------------------------------------------------------------------------------
    
    # Calculo la entropía de mis distribuciones
    
    Entropias = Calculo_Entropia(DF, path, 20, 20)
    
    #--------------------------------------------------------------------------------
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        #------------------------------------------------------------------------------------------
        # Armo mi matriz con los valores de entropía y con los valores de la varianza
        
        ZZ[0,(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Entropias[EXTRAS][PARAM_X][PARAM_Y])
        ZZ[1,(Arr_param_y.shape[0]-1)-fila,columna] = np.var(Entropias[EXTRAS][PARAM_X][PARAM_Y])
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Entropia EP_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Entropia Opiniones",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ[0],shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.title("Entropía de opiniones en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Entropia Opiniones")
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Varianza Entropia EP_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Varianza Entropia",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ[1],shading="nearest", cmap = "magma")
    plt.colorbar()
    plt.title("Varianza de Entropía en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Varianza Entropia")

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la varianza de las opiniones como métrica.

def Mapa_Colores_Pol_vs_Oscil(DF,path,carpeta,T,SIM_param_x,SIM_param_y,
                              ID_param_extra_1):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.ones(XX.shape)*(-1)
    
    #--------------------------------------------------------------------------------
    
    for EXTRAS in Arr_EXTRAS:
        
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
        
            # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
            Pasos = np.zeros(len(archivos))
            Varianzas = np.zeros(len(archivos))
            
            #-----------------------------------------------------------------------------------------
            
            for indice,nombre in enumerate(archivos):
                
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Pasos simulados
                # Semilla
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                # repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                
                # Leo los datos de las Opiniones Finales
                Opifinales = np.zeros((T,AGENTES))
                
                # Normalizo mis datos usando el valor de Kappa
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/EXTRAS
                
                
                Pasos[indice] = int(Datos[7][0])
                M_cov = np.cov(Opifinales)
                Varianzas[indice] = np.trace(M_cov) / T
                
                
            #------------------------------------------------------------------------------------------
            # Con los "tiempos" de las simulaciones calculo la fracción de estados que llegaron hasta el final
            # de la simulación, así la variación de esos tiempos de simulación
            
            Varianzas_oscil = Varianzas[Pasos == 200000]
            
            if (Pasos == 200000).any():
                ZZ[Arr_param_y.shape[0]-1-fila,columna] = np.count_nonzero(Varianzas_oscil > 0.1) / np.count_nonzero(Pasos == 200000)
        
        #--------------------------------------------------------------------------------
    
        # Una vez que tengo los ZZ completos, armo mis mapas de colores
        
        #--------------------------------------------------------------------------------
        # Fracción de estados oscilantes
        direccion_guardado = Path("../../../Imagenes/{}/Pol_vs_Oscil_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("PolOscil",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "magma")
        plt.colorbar()
        plt.title("Fraccion de estados Polarizados respecto de Oscilantes")
    
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("PolOscil")
        

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Histogramas_Multiples(DF,path,carpeta,T,ID_param_x,ID_param_y,
                          ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.array([0.6])
    Arr_param_y = np.array([0.2,0.4,0.6,0.8])
    
    # Defino la cantidad de filas y columnas que voy a graficar
    Filas = 10
    Columnas = T*2
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------

            plt.rcParams.update({'font.size': 28})
            plt.figure(figsize=(50,42))
            plots = [[plt.subplot(Filas, Columnas, i*Columnas + j + 1) for j in range(Columnas)] for i in range(Filas)]
            
            for nombre in archivos:

                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])                
                if repeticion < Filas*2:
                
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    # Levanto los datos del archivo
                    Datos = ldata(path / nombre)
                    
                    # Leo los datos de las Opiniones Finales y las normalizo
                    Opifinales = np.array(Datos[5][:-1:], dtype="float")
                    Opifinales =  Opifinales / EXTRAS
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                    # Armo mi gráfico, lo guardo y lo cierro
                    
                    fila = repeticion % Filas
                    salto = math.floor(repeticion / Filas)
                    
                    for topico in range(T):
                        plots[fila][topico+salto*T].hist(Opifinales[topico::T], bins=np.linspace(-1, 1, 21), density=True, color='tab:blue')
                        plots[fila][topico+salto*T].set_xlim(-1, 1)  # Set x-axis limits
            
            # Le pongo nombres a los ejes más externos
            for i, row in enumerate(plots):
                for j, subplot in enumerate(row):
                    if j == 0:  # First column, set y label
                        subplot.set_ylabel('Densidad')
                    if i == Filas - 1:  # Last row, set x label
                        subplot.set_xlabel("Opiniones")# r"$x_i$")
                        
            # Set titles for each column
            column_titles = ['Histogramas tópico 0', 'Histogramas Tópico 1', 'Histogramas tópico 0', 'Histogramas Tópico 1']
            for j, title in enumerate(column_titles):
                plt.subplot(Filas, Columnas, j + 1)  # Go to first subplot in the column
                plt.title(title, fontsize=35)  # Set title for the column
                
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Set main title for the entire figure
            plt.suptitle(r'Histogramas en $\beta$ = {}'.format(PARAM_Y), fontsize=40)
            
            direccion_guardado = Path("../../../Imagenes/{}/Conjunto Histogramas_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                      ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS))
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Graf_Histogramas_Promedio(DF,path,carpeta,bins,cmap,
                              ID_param_x,ID_param_y,ID_param_extra_1):

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            OpiTotales = np.empty(0)
            
            for nombre in archivos:
            
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
                OpiTotales = np.concatenate((OpiTotales,Opifinales),axis=0)
                
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            direccion_guardado = Path("../../../Imagenes/{}/Hists_prom_N={:.0f}_{}={:.2f}_{}={:.2f}.png".format(carpeta,AGENTES,
                                                                                        ID_param_x,PARAM_X,ID_param_y,PARAM_Y))
            
            # Armo mi gráfico
            plt.rcParams.update({'font.size': 32})
            fig, axs = plt.subplots(1, 3, figsize=(54, 21))
            
            # Armo el gráfico 2D primero
            _, _, _, im = axs[0].hist2d(OpiTotales[0::T], OpiTotales[1::T], bins=bins,
                                     range=[[-EXTRAS,EXTRAS],[-EXTRAS,EXTRAS]],density=True,
                                     cmap=cmap)
            axs[0].set_xlabel(r"$x_i^1$")
            axs[0].set_ylabel(r"$x_i^2$")
            axs[0].set_title('Histograma 2D Promediado')
            cb = plt.colorbar(im, ax=axs[0])
            cb.set_label("Fracción")
            
            # Armo el gráfico del histograma del tópico 0
            axs[1].hist(OpiTotales[0::T], bins=np.linspace(-10, 10, 21), density=True,color='tab:blue')
            axs[1].set_xlim(-10, 10)  # Set x-axis limits
            axs[1].set_xlabel("Opiniones")
            axs[1].set_ylabel("Fracción")
            axs[1].set_title('Tópico 0')
            
            # Armo el gráfico del histograma del tópico 1
            axs[2].hist(OpiTotales[1::T], bins=np.linspace(-10, 10, 21), density=True,color='tab:blue')
            axs[2].set_xlim(-10, 10)  # Set x-axis limits
            axs[2].set_xlabel("Opiniones")
            axs[2].set_ylabel("Fracción")
            axs[2].set_title('Tópico 1')
            
            # Título de la figura total
            fig.suptitle('{}={:.2f}_{}={:.2f}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y))
            plt.tight_layout()
            
            # Lo guardo y lo cierro
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()


#-----------------------------------------------------------------------------------------------

# Con la siguiente función me leo los datos de la ANES

def Leer_Datos_ANES(filename,año):
    
    # Leo los datos del archivo que tiene formato dta
    df_raw_data = pd.read_stata(filename)
    
    # Después tendría que armar una separación en casos a partir del nombre
    # de la ANES, no de un número externo
    if año == 2020:
        # Brief description of the codes
        dict_labels = {'V201200':'Liberal-Conservative self Placement', 'V201225x':'Voting as duty or choice','V201231x':'Party Identity',
                       'V201246':'Spending & Services', 'V201249':'Defense Spending', 'V201252':'Gov-private Medical Insurance',
                       'V201255':'Guaranteed job Income', 'V201258':'Gov Assistance to Blacks', 'V201262':'Environment-Business Tradeoff',
                       'V201342x':'Abortion Rights Supreme Court', 'V201345x':'Death Penalty','V201356x':'Vote by mail',
                       'V201362x':'Allowing Felons to vote', 'V201372x':'Helpful-Harmful if Pres didnt have to worry about Congress',
                       'V201375x':'Restricting Journalist access', 'V201382x':'Corruption increased or decreased since Trump',
                       'V201386x':'House impeachment decision', 'V201405x':'Require employers to offer paid leave to parents',
                       'V201408x':'Allow to refuse service to same sex couples', 'V201411x':'Transgender Policy', 'V201420x':'Birthright Citizenship',
                       'V201423x':'Should children brought illegally be sent back','V201426x':'Wall on border with Mexico',
                       'V201429':'Best way to deal with Urban Unrest','V201605x':'Political Violence compared to 4 years ago',
                       'V202236x':'Allowing refugees to come to US','V202239x':'Effect of Illegal inmigration on crime rate',
                       'V202242x':'Providing path to citizenship','V202245x':'Returning unauthorized immigrants to native country',
                       'V202248x':'Separating children from detained immigrants','V202255x':'Less or more Government',
                       'V202256':'Good for society to have more government regulation',
                       'V202259x':'Government trying to reduce income inequality','V202276x':'People in rural areas get more/less from Govt.',
                       'V202279x':'People in rural areas have too much/too little influence','V202282x':'People in rural areas get too much/too little respect',
                       'V202286x':'Easier/Harder for working mother to bond with child','V202290x':'Better/Worse if man works and woman takes care of home',
                       'V202320x':'Economic Mobility compared to 20 years ago','V202328x':'Obamacare','V202331x':'Vaccines in Schools',
                       'V202336x':'Regulation on Greenhouse Emissions','V202341x':'Background checks for guns purchases',
                       'V202344x':'Banning "Assault-style" Rifles','V202347x':'Government buy back of "Assault-Style" Rifles',
                       'V202350x':'Government action about opiod drug addiction','V202361x':'Free trade agreements with other countries',
                       'V202376x':'Federal program giving 12K a year to citizens','V202380x':'Government spending to help pay for health care',
                       'V202383x':'Health benefits of vaccination outweigh risks','V202390x':'Trasgender people serve in military',
                       'V202490x':'Government treats whites or blacks better','V202493x':'Police treats whites or blacks better',
                       'V202542':'Use Facebook','V202544':'Use Twitter'}

        labels = list(dict_labels.keys())

        labels_pre = list()
        labels_post = list()

        for label in labels:
            if label[3] == "1":
                labels_pre.append(label)
            elif label[3] == "2":
                labels_post.append(label)
                
        # Primer Filtro

        labels_politicos = ['V201200','V201231x','V201372x','V201386x','V201408x',
                            'V201411x','V201420x','V201426x','V202255x','V202328x','V202336x']

        labels_apoliticos = ['V201429','V202320x','V202331x','V202341x','V202344x',
                             'V202350x','V202383x']

        labels_dudosos = ['V201225x','V201246','V201249','V201252','V201255','V201258',
                          'V201262','V202242x','V202248x']

        labels_filtrados = labels_politicos + labels_apoliticos + labels_dudosos
        
        # Preparo el dataframe que voy a sacar
        df_data_aux = df_raw_data[labels]
        df_data = pd.DataFrame()

        for code in labels_filtrados:
            df_data[code] = df_data_aux[code].apply(data_processor)
        
        # Asigno los pesos
        df_data[['V200010a','V200010b']] = df_raw_data[['V200010a','V200010b']]

    return df_data, dict_labels

#-----------------------------------------------------------------------------------------------
    
# Armo dos mapas de colores de DJS. Uno de los mapas de colores considera que la distribución
# no tiene el punto en el centro en que ambos agentes opinan neutro. El otro mapa de colores
# no tiene los agentes que hayan opinado neutro en ninguna de las preguntas.
# En ambos casos estoy considerando que ambas preguntas tienen 7 respuestas. Voy a tener que ir
# viendo cómo resolver si tienen 6 respuestas.

def Mapas_Colores_DJS(DF_datos,DF_Anes, dict_labels,path,carpeta,Dic_ANES,bins,
                      ID_param_x,SIM_param_x,ID_param_y,SIM_param_y):
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    frac_agente_ind = 1/AGENTES
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ_centro = np.zeros((XX.shape[0],XX.shape[1],100))
    ZZ_cruz = np.zeros((XX.shape[0],XX.shape[1],100))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo mis distribuciones sacando el centro, es decir, saco a los agentes que respondieron neutro
    # en ambas encuestas y también extraigo la distribución sacando la cruz, que sería sacar a los agentes que
    # respondieron neutro en alguna encuesta.
    
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    
    # Distribución de encuesta sin el Centro
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) | (df_aux[Dic_ANES["code_2"]] != 4)] # Sólo saca el centro
    hist2d_centro, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Distribución de encuesta sin la Cruz
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) & (df_aux[Dic_ANES["code_2"]] != 4)] # Saca la cruz
    hist2d_cruz, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Armo las distribuciones finalmente
    
    Distr_Enc_Centro = np.reshape(hist2d_centro,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
    Distr_Enc_Centro = np.delete(Distr_Enc_Centro,24) # Saco el elemento del centro que tiene un cero
    
    Distr_Enc_Cruz = np.reshape(hist2d_cruz,(hist2d_cruz.shape[0]*hist2d_cruz.shape[1],1))
    Ind_nulos = np.array([3,10,17,21,22,23,24,25,26,27,31,38,45])
    Distr_Enc_Cruz = np.delete(Distr_Enc_Cruz,Ind_nulos)
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        Dist_previa_centro = np.zeros(4)
        Dist_previa_cruz = np.zeros(4)
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales y me armo una distribución en forma de matriz de 7x7
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Orig = Clasificacion(Opifinales,hist2d_centro.shape[0],hist2d_centro.shape[1],T)
            
            #-----------------------------------------------------------------------------------------
            
            for rotacion in range(4):
                
                # Tomo la distribución original, le doy forma de matriz, la roto y luego la plancho
                Distr_Orig = np.reshape(Distr_Orig, hist2d_centro.shape)
                Distr_Orig = Rotar_matriz(Distr_Orig)
                Distr_Orig = np.reshape(Distr_Orig,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
                
                # A partir de la distribución original voy a fabricarme dos distribuciones de simulaciones, una sin el
                # punto central, otra sin la cruz.
                
                # Primero armo la que no tiene el centro removiendo el punto central
                Distr_Sim_centro = np.delete(Distr_Orig,24)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_centro) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_centro))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_centro)
                        Distr_Sim_centro[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_centro == 0)
                ubic = np.argmax(Distr_Sim_centro)
                Distr_Sim_centro[Distr_Sim_centro == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_centro[ubic] -= frac_agente_ind*restar
                
                Dist_previa_centro[rotacion] = jensenshannon(Distr_Enc_Centro,Distr_Sim_centro)
                
                
                # Segundo armo la que no tiene la cruz removiendo los puntos en la cruz de la distribución
                Distr_Sim_cruz = np.delete(Distr_Orig, Ind_nulos)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_cruz) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_cruz))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_cruz)
                        Distr_Sim_cruz[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_cruz == 0)
                ubic = np.argmax(Distr_Sim_cruz)
                Distr_Sim_cruz[Distr_Sim_cruz == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_cruz[ubic] -= frac_agente_ind*restar
                
                Dist_previa_cruz[rotacion] = jensenshannon(Distr_Enc_Cruz,Distr_Sim_cruz)
                
            #-----------------------------------------------------------------------------------------
            
            # Una vez que calcule las 4 distancias habiendo rotado 4 veces la distribución,
            # lo que me queda es guardar eso en las matrices ZZ correspondientes.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            ZZ_centro[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(Dist_previa_centro)
            ZZ_cruz[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(Dist_previa_cruz)
    
    #--------------------------------------------------------------------------------
    
    # Antes de ordenar mis matrices, voy a obtener la info del gráfico que más se parece
    # y del décimo que más se parece se parece a lo que estoy queriendo comparar.
    
    iMin_centro = np.unravel_index(np.argmin(ZZ_centro),ZZ_centro.shape)
    iMin_cruz = np.unravel_index(np.argmin(ZZ_cruz),ZZ_cruz.shape)
    
    # Hallo el décimo que más se parece a la distribución. Arranco con el que no tiene centro
    
    flattened_array = ZZ_centro.flatten()
    sorted_indices = np.argsort(flattened_array)
    tenth_element_flat_index = sorted_indices[9]
    iMax_centro = np.unravel_index(tenth_element_flat_index, ZZ_centro.shape)
    
    flattened_array = ZZ_cruz.flatten()
    sorted_indices = np.argsort(flattened_array)
    tenth_element_flat_index = sorted_indices[9]
    iMax_cruz = np.unravel_index(tenth_element_flat_index, ZZ_cruz.shape)
    
    #--------------------------------------------------------------------------------
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin centro
    direccion_guardado = Path("../../../Imagenes/{}/Sin Centro/DistanciaJS_{}vs{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"]))
    
    # Organizo las matrices ZZ según su similaridad
    ZZ_centro = np.sort(ZZ_centro)
    ZZ_cruz = np.sort(ZZ_cruz)
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon Centro",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,np.mean(ZZ_centro,axis=2),shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.title("Distancia Jensen-Shannon sin centro\n {} vs {}".format(dict_labels[Dic_ANES["code_2"]],dict_labels[Dic_ANES["code_1"]]))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon Centro")
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin cruz
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"]))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon Cruz",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,np.mean(ZZ_cruz,axis=2),shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.title("Distancia Jensen-Shannon sin cruz\n {} vs {}".format(dict_labels[Dic_ANES["code_2"]],dict_labels[Dic_ANES["code_1"]]))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon Cruz")
    
    # Y ahora me armo los rankings
    
    for i in range(3):
        direccion_guardado = Path("../../../Imagenes/{}/Sin Centro/DistanciaJS_{}vs{}_r{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"],i))
    
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Distancia Jensen-Shannon Centro",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh

        
        plt.pcolormesh(XX,YY,np.mean(ZZ_centro[:,:,0:10+i*30],axis=2),shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.title("Distancia Jensen-Shannon sin centro {} simulaciones\n {} vs {}".format(10+i*30,dict_labels[Dic_ANES["code_2"]],dict_labels[Dic_ANES["code_1"]]))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Ranking Distancia Jensen-Shannon Centro")
        
    
    for i in range(3):
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}_r{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"],i))
    
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Distancia Jensen-Shannon Cruz",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh

        
        plt.pcolormesh(XX,YY,np.mean(ZZ_cruz[:,:,0:10+i*30],axis=2),shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.title("Distancia Jensen-Shannon sin cruz {} simulaciones\n {} vs {}".format(10+i*30,dict_labels[Dic_ANES["code_2"]],dict_labels[Dic_ANES["code_1"]]))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Ranking Distancia Jensen-Shannon Cruz")
        
    #-------------------------------------------------------------------------------------------------
    
    # Hago los gráficos de histograma 2D de las simulaciones que más se parecen y que menos se parecen
    # a mis distribuciones de las encuestas
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF_datos,path, 20, 20)
    
    # Armo listas de strings y números para mis archivos
    Lista_similaridad = ["min_distancia","max_distancia","min_distancia","max_distancia"]
    Lista_carpeta = ["Sin Centro","Sin Centro", "Sin Cruz", "Sin Cruz"]
    Valor_distancia = [np.min(ZZ_centro),np.max(ZZ_centro),np.min(ZZ_cruz),np.max(ZZ_cruz)]
    Vmin = np.array([np.argmin(Distr_Enc_Centro),np.argmin(Distr_Enc_Centro),np.argmin(Distr_Enc_Cruz),np.argmin(Distr_Enc_Cruz)])
    Vmax = np.array([np.argmax(Distr_Enc_Centro),np.argmax(Distr_Enc_Centro),np.argmax(Distr_Enc_Cruz),np.argmax(Distr_Enc_Cruz)])
    
    for m,tupla in enumerate([iMin_centro, iMax_centro, iMin_cruz, iMax_cruz]):
    
        PARAM_X = XX[tupla[0],tupla[1]]
        PARAM_Y = YY[tupla[0],tupla[1]]
        
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            if repeticion == tupla[2]:
            
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                
                # Leo los datos de las Opiniones Finales
                Opifinales = np.array(Datos[5][:-1:], dtype="float")
                Opifinales = (Opifinales/EXTRAS)*bins[-1]
                X_0 = Opifinales[0::T]
                Y_0 = Opifinales[1::T]
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                
                direccion_guardado = Path("../../../Imagenes/{}/Hist_2D_{}_{}vs{}.png".format(carpeta / Lista_carpeta[m],Lista_similaridad[m],Dic_ANES["code_2"],Dic_ANES["code_1"]))
                
                indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
                estado = int(Frecuencias[indice])
                
                Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
                           "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
                           "Polarización 1D y Consenso con anchura",
                           "Polarización Ideológica con anchura", "Transición con anchura",
                           "Polarización Descorrelacionada con anchura"]
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Tengo que armar los valores de X e Y que voy a graficar en función de si saco la cruz
                # o si saco el centro
                
                if Lista_carpeta[m] == "Sin Centro":
                    X = X_0[((X_0>bins[4]) | (X_0<bins[3])) | ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) | ((Y_0>bins[4]) | (Y_0<bins[3]))]
                elif Lista_carpeta[m] == "Sin Cruz":
                    X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                    Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Armo mi gráfico, lo guardo y lo cierro
                
                plt.rcParams.update({'font.size': 32})
                plt.figure(figsize=(28,21))
                _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="magma")
                plt.xlabel(r"$x_i^1$")
                plt.ylabel(r"$x_i^2$")
                # Set x-ticks and y-ticks from -10 to 10 using plt.xticks() and plt.yticks()
                # plt.xticks(np.arange(-10, 11, 1))
                # plt.yticks(np.arange(-10, 11, 1))
                plt.title('Distancia JS = {:.2f}, {}={:.2f}_{}={:.2f} \n {} \n {} vs {}'.format(Valor_distancia[m],ID_param_x,PARAM_X,ID_param_y,PARAM_Y,Nombres[estado],dict_labels[Dic_ANES["code_2"]],dict_labels[Dic_ANES["code_1"]]))
                cbar = plt.colorbar(im, label='Fracción')
                cbar.set_clim(Vmin[m], Vmax[m])
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close()
    
    

#-----------------------------------------------------------------------------------------------
    
# Realizo un ajuste de los parámetros Beta y Cos(delta)
    
def Ajuste_DJS(DF_datos,DF_Anes,path,carpeta,Dic_ANES,
               Cd_range,B_range):
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    frac_agente_ind = 1/AGENTES
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_x = Arr_param_x[(Arr_param_x >= Cd_range[0]) & (Arr_param_x <= Cd_range[1])]
    
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    Arr_param_y = Arr_param_y[(Arr_param_y >= B_range[0]) & (Arr_param_y <= B_range[1])]
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    # Armo el vector en el que voy a poner todas las distancias Jensen-Shannon que calcule
    YY_centro = np.reshape(np.array([]),(0,1))
    YY_cruz = np.reshape(np.array([]),(0,1))
    
    XX = np.reshape(np.array([]),(0,5))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo la distribución en hist2d
    
    # Extraigo mis distribuciones sacando el centro, es decir, saco a los agentes que respondieron neutro
    # en ambas encuestas y también extraigo la distribución sacando la cruz, que sería sacar a los agentes que
    # respondieron neutro en alguna encuesta.
    
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    
    # Distribución de encuesta sin el Centro
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) | (df_aux[Dic_ANES["code_2"]] != 4)] # Sólo saca el centro
    hist2d_centro, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Distribución de encuesta sin la Cruz
    
    df_filtered = df_aux[(df_aux[Dic_ANES["code_1"]] != 4) & (df_aux[Dic_ANES["code_2"]] != 4)] # Saca la cruz
    hist2d_cruz, xedges, yedges, im = plt.hist2d(x=df_filtered[Dic_ANES["code_1"]], y=df_filtered[Dic_ANES["code_2"]], weights=df_filtered[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
              bins=[np.arange(df_filtered[Dic_ANES["code_1"]].min()-0.5, df_filtered[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_filtered[Dic_ANES["code_2"]].min()-0.5, df_filtered[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    # Armo las distribuciones finalmente
    
    Distr_Enc_Centro = np.reshape(hist2d_centro,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
    Distr_Enc_Centro = np.delete(Distr_Enc_Centro,24) # Saco el elemento del centro que tiene un cero
    
    Distr_Enc_Cruz = np.reshape(hist2d_cruz,(hist2d_cruz.shape[0]*hist2d_cruz.shape[1],1))
    Ind_nulos = np.array([3,10,17,21,22,23,24,25,26,27,31,38,45])
    Distr_Enc_Cruz = np.delete(Distr_Enc_Cruz,Ind_nulos)
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        #-----------------------------------------------------------------------------------------
        
        Dist_previa_centro = np.zeros(4)
        Dist_previa_cruz = np.zeros(4)
        DistJS_centro = np.zeros((archivos.shape[0],1))
        DistJS_cruz = np.zeros((archivos.shape[0],1))
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales y me armo una distribución en forma de matriz de 7x7
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Orig = Clasificacion(Opifinales,hist2d_centro.shape[0],hist2d_centro.shape[1],T)
            
            #-----------------------------------------------------------------------------------------
            
            for rotacion in range(4):
                
                # Tomo la distribución original, le doy forma de matriz, la roto y luego la plancho
                Distr_Orig = np.reshape(Distr_Orig, hist2d_centro.shape)
                Distr_Orig = Rotar_matriz(Distr_Orig)
                Distr_Orig = np.reshape(Distr_Orig,(hist2d_centro.shape[0]*hist2d_centro.shape[1],1))
                
                # A partir de la distribución original voy a fabricarme dos distribuciones de simulaciones, una sin el
                # punto central, otra sin la cruz.
                
                # Primero armo la que no tiene el centro removiendo el punto central
                Distr_Sim_centro = np.delete(Distr_Orig,24)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_centro) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_centro))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_centro)
                        Distr_Sim_centro[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_centro == 0)
                ubic = np.argmax(Distr_Sim_centro)
                Distr_Sim_centro[Distr_Sim_centro == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_centro[ubic] -= frac_agente_ind*restar
                
                Dist_previa_centro[rotacion] = jensenshannon(Distr_Enc_Centro,Distr_Sim_centro)
                
                
                # Segundo armo la que no tiene la cruz removiendo los puntos en la cruz de la distribución
                Distr_Sim_cruz = np.delete(Distr_Orig, Ind_nulos)
                # Como removí parte de mi distribución, posiblemente ya no esté normalizada
                # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
                # de que otra vez esté normalizada
                if np.sum(Distr_Sim_cruz) != 1:
                    agentes_agregar = int((1-np.sum(Distr_Sim_cruz))/frac_agente_ind)
                    for i in range(agentes_agregar):
                        ubic_min = np.argmin(Distr_Sim_cruz)
                        Distr_Sim_cruz[ubic_min] += frac_agente_ind
                # Luego de volver a normalizar mi distribución, si quedaron lugares
                # sin agentes, los relleno
                restar = np.count_nonzero(Distr_Sim_cruz == 0)
                ubic = np.argmax(Distr_Sim_cruz)
                Distr_Sim_cruz[Distr_Sim_cruz == 0] = np.ones(restar)*frac_agente_ind
                Distr_Sim_cruz[ubic] -= frac_agente_ind*restar
                
                Dist_previa_cruz[rotacion] = jensenshannon(Distr_Enc_Cruz,Distr_Sim_cruz)
                
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            DistJS_centro[repeticion] = np.min(Dist_previa_centro)
            DistJS_cruz[repeticion] = np.min(Dist_previa_cruz)
            
        #------------------------------------------------------------------------------------------
        # Una vez que calculé todas las distancias Jensen-Shannon, concateno eso con mi
        # vector YY, así como concateno mi vector XX
        
        YY_centro = np.concatenate((YY_centro,DistJS_centro))
        YY_cruz = np.concatenate((YY_cruz,DistJS_cruz))
        
        Bcuad = np.ones((len(archivos),1))*PARAM_Y*PARAM_Y
        Blin = np.ones((len(archivos),1))*PARAM_Y
        CDcuad = np.ones((len(archivos),1))*PARAM_X*PARAM_X
        CDlin = np.ones((len(archivos),1))*PARAM_X
        Unos = np.ones((len(archivos),1))
        
        M = np.concatenate((Bcuad,Blin,CDcuad,CDlin,Unos),axis=1)
        XX = np.concatenate((XX,M))
        
    #------------------------------------------------------------------------------------------
    
    # Una vez que tengo armado los vectores XX e YY, ya puedo calcular mis coeficientes

    param_centro = np.matmul(np.linalg.inv(np.matmul(np.transpose(XX),XX)),np.matmul(np.transpose(XX),YY_centro))
    param_cruz = np.matmul(np.linalg.inv(np.matmul(np.transpose(XX),XX)),np.matmul(np.transpose(XX),YY_cruz))
    
    return param_centro,param_cruz

#-----------------------------------------------------------------------------------------------
    
# Ploteo el paraboloide que ajuste en un gráfico 3D

def plot_3d_surface(carpeta, Dic_ANES, func, params_centro, params_cruz, x_range, y_range,
                    SIM_param_x, SIM_param_y,x_samples=100, y_samples=100):
    
    """
    Plot a 3D surface for a given mathematical function.
    
    Parameters:
    - func: The mathematical function to plot. It should take two arguments (x and y) and return a value.
    - x_range: A tuple specifying the range of x values (min, max).
    - y_range: A tuple specifying the range of y values (min, max).
    - x_samples: Number of samples in the x range.
    - y_samples: Number of samples in the y range.
    """
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], x_samples)
    y = np.linspace(y_range[0], y_range[1], y_samples)
    X, Y = np.meshgrid(x, y)
    Z_centro = func(X, Y, params_centro)
    Z_cruz = func(X, Y, params_cruz)

    #--------------------------------------------------------------------------------------
    
    # Armo el gráfico del paraboloide para el caso sin el centro de la distribución
    
    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Paraboloide_ajustado_sin_centro_{}vs{}.png".format(carpeta/"Sin Centro",Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(48,36))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_centro, cmap='viridis')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    
    # Format axes ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Adjust tick parameters
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.tick_params(axis='z')
    
    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x),labelpad = 60)
    ax.set_ylabel(r"${}$".format(SIM_param_y),labelpad = 60)
    ax.set_zlabel('Distancia JS',labelpad = 60)
    ax.set_title('Paraboloide ajustada con distribuciones sin centro \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()
    
    #--------------------------------------------------------------------------------------
    
    # Armo el gráfico para el caso sin la cruz central de la distribución
    
    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Paraboloide_ajustado_sin_cruz_{}vs{}.png".format(carpeta/"Sin Cruz",Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(48,36))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z_cruz, cmap='viridis')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Format axes ticks to 2 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Adjust tick parameters
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.tick_params(axis='z')
    
    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x),labelpad = 60)
    ax.set_ylabel(r"${}$".format(SIM_param_y),labelpad = 60)
    ax.set_zlabel('Distancia JS',labelpad = 60)
    ax.set_title('Paraboloide ajustada con distribuciones sin cruz \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()

#-----------------------------------------------------------------------------------------------
    
# Ploteo el los puntos sobre los que hice el ajuste en un gráfico 3D
    
def plot_3d_scatter(DF_datos,DF_Anes, path, carpeta, Dic_ANES, x_range, y_range,
                    SIM_param_x, SIM_param_y):
    
    """
    Plot a 3D surface for a given mathematical function.
    
    Parameters:
    - func: The mathematical function to plot. It should take two arguments (x and y) and return a value.
    - x_range: A tuple specifying the range of x values (min, max).
    - y_range: A tuple specifying the range of y values (min, max).
    - x_samples: Number of samples in the x range.
    - y_samples: Number of samples in the y range.
    """
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_x = Arr_param_x[Arr_param_x > x_range[0]]
    Arr_param_x = Arr_param_x[Arr_param_x < x_range[1]]
    
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    Arr_param_y = Arr_param_y[Arr_param_y > y_range[0]]
    Arr_param_y = Arr_param_y[Arr_param_y < y_range[1]]
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros((XX.shape[0],XX.shape[1],100))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo la distribución en hist2d
    
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[Dic_ANES["code_1"]], y=df_aux[Dic_ANES["code_2"]], weights=df_aux[Dic_ANES["weights"]], vmin=0,
             bins=[np.arange(df_aux[Dic_ANES["code_1"]].min()-0.5, df_aux[Dic_ANES["code_1"]].max()+1.5, 1), np.arange(df_aux[Dic_ANES["code_2"]].min()-0.5, df_aux[Dic_ANES["code_2"]].max()+1.5, 1)])
    plt.close()
    
    Distr_Enc = np.reshape(hist2d,(hist2d.shape[0]*hist2d.shape[1],1))
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        DistJS = np.zeros(archivos.shape[0])
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Pasos simulados
            # Semilla
            # Matriz de Adyacencia
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.array(Datos[5][:-1], dtype="float")
            Opifinales = Opifinales / EXTRAS
            Distr_Sim = np.reshape(Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],T),(hist2d.shape[0]*hist2d.shape[1],1))
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            DistJS[repeticion] = jensenshannon(Distr_Enc,Distr_Sim)
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna,:] = DistJS

    # Create the plot
    direccion_guardado = Path("../../../Imagenes/{}/Scatter de DJS_{}vs{}.png".format(carpeta,Dic_ANES["code_2"],Dic_ANES["code_1"]))
    plt.rcParams.update({'font.size': 44})
    fig = plt.figure(figsize=(40,30))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(100):
        ax.scatter(XX,YY,ZZ[:,:,i], c="blue", marker="o")

    # Labels and title
    ax.set_xlabel(r"${}$".format(SIM_param_x))
    ax.set_ylabel(r"${}$".format(SIM_param_y))
    ax.set_zlabel('Distancia JS')
    ax.set_title('Distancias calculadas \n {} vs {}'.format(Dic_ANES["code_2"],Dic_ANES["code_1"]))

    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close()
    

#-----------------------------------------------------------------------------------------------

# Tomo una matriz y la roto. Repito, roto la matriz como quien gira la cara de un
# cubo Rubik, no estoy rotando el objeto que la matriz representa.

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
