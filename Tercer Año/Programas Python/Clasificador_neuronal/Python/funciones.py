# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:00 2022

@author: Favio
"""

# Este archivo es para definir funciones

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import time
import math
from scipy.optimize import fsolve
from pathlib import Path
from cycler import cycler

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

##################################################################################
##################################################################################

# FUNCIONES GRAFICADORAS

##################################################################################
##################################################################################

def Clasificacion(Array,N,T):
    
    # Recibo un array de opiniones que van entre [-1,1]. Le sumo 1
    # para que las opiniones vayan entre [0,2].
    Array = Array+1
    
    # Divido mi espacio de tópicos 2D en cuadrados. Defino el ancho
    # de esos cuadrados.
    ancho = 2/N
    
    # Armo un array de tuplas que indiquen "fila" y "columna" en la cuál
    # cae cada opinión.
    Ubicaciones = np.array([(math.floor(x/ancho),math.floor(y/ancho)) for x,y in zip(Array[0::T],Array[1::T])])
    
    # Ahora me armo mi array de distribución, que cuenta cuántas opiniones tengo
    # por cada cajita.
    Distribucion = np.zeros((N*N))
    for opinion in Ubicaciones:
        # Tomo mínimos para que no intente ir a una cajita no existente. Tendría un problema
        # si algún agente tiene opinión máxima en algún tópico.
        fila = min(opinion[1],N-1)
        columna = min(opinion[0],N-1)
        Distribucion[fila*N+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Diccionario_metricas(DF,path,N):
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

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
            Entropia = np.zeros(archivos.shape[0])
            
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
        
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico::T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
        
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
        
                M_cov = np.cov(Opifinales)
                Varianza_X[repeticion] = M_cov[0,0]
                Varianza_Y[repeticion] = M_cov[1,1]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales/EXTRAS
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,N,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                Entropia[repeticion] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
                
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            if PARAM_Y not in Salida[EXTRAS][PARAM_X].keys():
                Salida[EXTRAS][PARAM_X][PARAM_Y] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Entropia"] = Entropia/np.log2(N*N)
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"] = Varianza_X
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"] = Varianza_Y
            
    return Salida

#-----------------------------------------------------------------------------------------------

def Identificacion_Estados(ent, sx, sy, cov, prom):

    
    # Reviso la entropía y separo en casos con y sin anchura
    
    if ent <= 0.3:
        
        # Estos son casos sin anchura
        
        if sx < 0.1 and sy < 0.1:
            
            # Caso de un sólo extremo
            
            # Consenso neutral
            if prom < 0.1:
                return 0
            
            # Consenso radicalizado
            else:
                return 1
                
        
        # Casos de dos extremos
        elif sx >= 0.1 and sy < 0.1:
            # Dos extremos horizontal
            return 2
        elif sx < 0.1 and sy >= 0.1:
            # Dos extremos vertical
            return 3
            
        else:
            if ent < 0.18:
                # Dos extremos ideológico
                return 4
            elif ent < 0.23:
                # Estados de Transición
                return 5
            else:
                # Cuatro extremos
                return 6
    
    else:
        
        # Estos son los casos con anchura
        
        # Casos de dos extremos
        if sx >= 0.1 and sy < 0.1:
            # Dos extremos horizontal
            return 7
        elif sx < 0.1 and sy >= 0.1:
            # Dos extremos vertical
            return 8
        
        else:
            # Polarización
            # Polarización ideológica
            if np.abs(cov) >= 0.1:
                return 9
                
            # Polarización descorrelacionada
            else:
                return 10

        
#--------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Histogramas_2D_Clasificador(DF,path,carpeta,bins,cmap,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

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
    imagen=0
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
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
        
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico::T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                # Calculo la matriz de Covarianza.

                M_cov = np.cov(Opifinales)
                VarX = M_cov[0,0]
                VarY = M_cov[1,1]
                Covar = M_cov[0,1]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:], dtype="float")
                Opifinales = Opifinales/EXTRAS
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,bins,T)
                
                # Con esa distribución puedo directamente calcular la entropía y el promedio de las opiniones.
                Entropia = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1) / np.log2(bins*bins)
                Prom = np.linalg.norm(np.array(Datos[5][::], dtype="float"),ord=1) / np.array(Datos[5][::], dtype="float").shape[0]
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Ahora que tengo todas mis métricas, puedo clasificar el estado final de mi distribución
                estado = Identificacion_Estados(Entropia, VarX, VarY, Covar, Prom)
                
                # Vuelvo a construir mi array de Opifinales para que los gráficos salgan bien
                Opifinales = np.array(Datos[5][:], dtype="float")
                
                #----------------------------------------------------------------------------------------------------------------------------------
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                # repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                direccion_guardado = Path("../Imagenes/Estado_{}_Img_{}.png".format(estado, imagen))
                
                # Armo mi gráfico, lo guardo y lo cierro
                
                plt.rcParams.update({'font.size': 32})
                plt.figure(figsize=(1240/100, 950/100))
                _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                                         range=[[-EXTRAS,EXTRAS],[-EXTRAS,EXTRAS]],density=True,
                                         cmap=cmap)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(direccion_guardado ,dpi = 100)
                plt.close()
                
                # Avanzo el número de imagen, que avanza con cada nuevo archivo
                imagen+=1

#-----------------------------------------------------------------------------------------------
