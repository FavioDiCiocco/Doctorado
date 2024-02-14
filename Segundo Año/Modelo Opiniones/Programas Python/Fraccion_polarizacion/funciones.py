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

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando el valor medio de la opinión.

def Promedio_opiniones_vs_T(DF,path,carpeta,T,ID_param_x,ID_param_y):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes    
    KAPPAS = int(np.unique(DF["Kappas"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    for topico in range(T):
        
        #--------------------------------------------------------------------------------
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            
            
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Kappas"]==KAPPAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])      
    
            #------------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Evolución de Opiniones
                # n filas con opiniones del total de agentes
                # Semilla
                # Número de la semilla
                
                # Es decir, mi archivo tiene n+3 filas
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                
                # Me defino el array en el cual acumulo los datos de las opiniones de todas
                # mis iteraciones
                Opiniones = np.zeros((len(Datos)-3,len(Datos[1])-1))
                
                # Leo los datos de las Opiniones
                for fila in range(len(Datos)-3):
                    Opiniones[fila,:] = Datos[fila+1][:-1:]
                    Opiniones[fila,:] = Opiniones[fila,:]/KAPPAS
            
                #------------------------------------------------------------------------------------------
                
                # Armo el array que va a contener los promedios de cada estado
                # guardado del sistema
                Promedios = np.array([np.mean(Opiniones[:,topico::T],axis=1) for topico in range(T)])
        
                #--------------------------------------------------------------------------------
                
                # Con los promedios armos el gráfico de opiniones vs T para ambos tópicos
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                direccion_guardado = Path("../../../Imagenes/{}/PromediovsT_{}={}_{}={}_Iter={}.png".format(carpeta,ID_param_x,PARAM_X,
                                          ID_param_y,PARAM_Y,repeticion))
                
                plt.rcParams.update({'font.size': 24})
                plt.figure("PromediovsT",figsize=(20,15))
                plt.xlabel(r"Tiempo$(10^3)$")
                plt.ylabel("Promedio Opiniones")
                plt.grid()
                
                # Hago el ploteo de las curvas de opinión en función del tiempo
                
                Tiempo = np.arange(Opiniones.shape[0])+1
                
                for topico in range(T):
                    plt.plot(Tiempo,Promedios[topico,:],"--",linewidth=3,label ="Topico {}".format(topico))
                
                plt.legend()
                plt.title(r"Promedio de opiniones vs T")
                
                # Guardo la figura y la cierro
                plt.savefig(direccion_guardado , bbox_inches = "tight")
                plt.close("PromediovsT")

    
#--------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Graf_Histograma_opiniones_2D(DF,path,carpeta,bins,cmap,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    Arr_KAPPAS = np.unique(DF["Kappas"])
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
    
    for KAPPAS in Arr_KAPPAS:
        for PARAM_X,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Kappas"]==KAPPAS) & 
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
                Opifinales = np.array(Datos[5][::], dtype="float")
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                # if repeticion < 2 :
                direccion_guardado = Path("../../../Imagenes/{}/Histograma_opiniones_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                                                                                                                                 ID_param_y,PARAM_Y,ID_param_extra_1,KAPPAS,repeticion))
                
                # Armo mi gráfico, lo guardo y lo cierro
                
                plt.rcParams.update({'font.size': 32})
                plt.figure(figsize=(20,15))
                _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                                         range=[[-KAPPAS,KAPPAS],[-KAPPAS,KAPPAS]],density=True,
                                         cmap=cmap)
                plt.xlabel(r"$x_i^1$")
                plt.ylabel(r"$x_i^2$")
                plt.title('Histograma 2D, {}={:.2f}_{}={:.2f}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y))
                plt.colorbar(im, label='Frecuencias')
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close()

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Traza_Covarianza_vs_T(DF,path,carpeta,T,ID_param_x,ID_param_y):
    
   # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes    
    KAPPAS = int(np.unique(DF["Kappas"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    for topico in range(T):
        
        #--------------------------------------------------------------------------------
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            
            
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Kappas"]==KAPPAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])      
    
            #------------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Evolución de Opiniones
                # n filas con opiniones del total de agentes
                # Semilla
                # Número de la semilla
                
                # Es decir, mi archivo tiene n+3 filas
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                
                # Me defino el array en el cual acumulo los datos de las opiniones de todas
                # mis iteraciones
                Opiniones = np.zeros((len(Datos)-3,len(Datos[1])-1))
                
                # Leo los datos de las Opiniones
                for fila in range(len(Datos)-3):
                    Opiniones[fila,:] = Datos[fila+1][:-1:]
                    Opiniones[fila,:] = Opiniones[fila,:]/KAPPAS
            
                #------------------------------------------------------------------------------------------
                
                # Armo el array que va a contener las trazas de covarianza 
                # de cada estado guardado del sistema
                Covarianzas = np.array([np.trace(
                        np.cov(np.array([Opiniones[fila,topico::T] for topico in range(T)])
                                )
                        )/2 for fila in range(Opiniones.shape[0])])

                #--------------------------------------------------------------------------------
                
                # Con los promedios armos el gráfico de opiniones vs T para ambos tópicos
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                direccion_guardado = Path("../../../Imagenes/{}/TrazaCovvsT_{}={}_{}={}_Iter={}.png".format(carpeta,ID_param_x,PARAM_X,
                                          ID_param_y,PARAM_Y,repeticion))
                
                plt.rcParams.update({'font.size': 24})
                plt.figure("TrazacovvsT",figsize=(20,15))
                plt.xlabel(r"Tiempo$(10^3)$")
                plt.ylabel("Traza Covarianzas")
                plt.grid()
                plt.title(r"Suma Varianzas vs T")
                
                # Hago el ploteo de las curvas de opinión en función del tiempo
                
                Tiempo = np.arange(Opiniones.shape[0])+1
                
                plt.plot(Tiempo,Covarianzas,"--",linewidth=4)
                
                # Guardo la figura y la cierro
                plt.savefig(direccion_guardado , bbox_inches = "tight")
                plt.close("TrazacovvsT")


#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Fraccion_polarizados_vs_T(DF,path,carpeta):
    
   # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes    
    KAPPAS = int(np.unique(DF["Kappas"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Con los promedios armos el gráfico de opiniones vs T para ambos tópicos
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion_polarizados.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracPol",figsize=(20,15))
    
    #--------------------------------------------------------------------------------
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        
        
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Kappas"]==KAPPAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])      
        
        
        Tiempos_polarizados = np.zeros(len(archivos))
        
        #------------------------------------------------------------------------------------------
        
        for indice,nombre in enumerate(archivos):
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Evolución de Opiniones
            # n filas con opiniones del total de agentes
            # Semilla
            # Número de la semilla
            
            # Es decir, mi archivo tiene n+3 filas
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # El tiempo de polarización es n. Si n fuera cero,
            # fijo a mano el tiempo de polarización a 1
            Tiempos_polarizados[indice] = max(len(Datos)-3,1)

            #--------------------------------------------------------------------------------
        
        # Construyo el array de estados polarizados
        Fraccion_polarizados = np.zeros(100)
        for i in range(100):
            Fraccion_polarizados[i] = np.count_nonzero(Tiempos_polarizados > i)
        Fraccion_polarizados = Fraccion_polarizados/100
        # Hago el ploteo de las curvas de opinión en función del tiempo
        
        Tiempo = np.arange(Fraccion_polarizados.shape[0])+1
        plt.plot(Tiempo,Fraccion_polarizados,"--",linewidth=4,label="Beta = {}".format(PARAM_Y))
        
    # Guardo la figura y la cierro
    plt.xlabel(r"Tiempo$(10^3)$")
    plt.ylabel(r"$f_p$")
    plt.grid()
    plt.legend()
    plt.title("Fracción de estados polarizados")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracPol")
    
#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Fraccion_polarizados_vs_Y(DF,path,carpeta):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes    
    EXTRAS = int(np.unique(DF["Extra"]))
    PARAM_X = float(np.unique(DF["parametro_x"]))
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Defino el número de tópicos
    T=2
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(j,param_y) for j,param_y in enumerate(Arr_param_y)]
    
    # Con los promedios armos el gráfico de opiniones vs T para ambos tópicos
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion_polarizados.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracPol",figsize=(20,15))
    
    # Construyo el array de fracción estados polarizados
    Fraccion_polarizados = np.zeros(Arr_param_y.shape[0])
    
    #--------------------------------------------------------------------------------
    for i_y,PARAM_Y in Tupla_total:
        
        
        
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Extra"]==EXTRAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])      
        
        
        Estados_polarizados = np.zeros(len(archivos))
        
        #------------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Evolución de Opiniones
            # n filas con opiniones del total de agentes
            # Semilla
            # Número de la semilla
            
            # Es decir, mi archivo tiene n+3 filas
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.zeros((T,AGENTES))
    
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")
                Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
            
            # Esta función normaliza las Opiniones Finales usando la 
            # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
            # corregir a que EXTRAS sea PARAM_X o algo así
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agentes
    
            M_cov = np.cov(Opifinales)
            
            # Defino si el estado que observé está polarizado si la suma de sus
            # varianzas me da mayor a 0.1.
            if np.trace(M_cov)/2 > 0.1:
                Estados_polarizados[repeticion] = 1

            #--------------------------------------------------------------------------------
        
        Fraccion_polarizados[i_y] = np.sum(Estados_polarizados)/len(archivos)
    

    plt.plot(Arr_param_y,Fraccion_polarizados,"--",linewidth=4)
    
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$f_p$")
    plt.grid()
    # plt.legend()
    plt.title("Fracción de estados polarizados")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracPol")

#-----------------------------------------------------------------------------------------------

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
            Covarianza = np.zeros(archivos.shape[0])
            Promedios = np.zeros(archivos.shape[0])
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
                    Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
        
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
        
                M_cov = np.cov(Opifinales)
                Varianza_X[repeticion] = M_cov[0,0]
                Varianza_Y[repeticion] = M_cov[1,1]
                Covarianza[repeticion] = M_cov[0,1]
                Promedios[repeticion] = np.linalg.norm(np.array(Datos[5][:-1:], dtype="float"),ord=1) / np.array(Datos[5][:-1:], dtype="float").shape[0]
                
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
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"] = Covarianza
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Promedios"] = Promedios
            
    return Salida

#-----------------------------------------------------------------------------------------------

def Identificacion_Estados(Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
    
    Resultados = np.zeros(len(Entropia))
    
    for i,ent,sx,sy,cov,prom in zip(np.arange(len(Entropia)),
                                    Entropia, Sigma_X, Sigma_Y, Covarianza, Promedios):
        
        # Reviso la entropía y separo en casos con y sin anchura
        
        if ent <= 0.3:
            
            # Estos son casos sin anchura
            
            if sx < 0.1 and sy < 0.1:
                
                # Caso de un sólo extremo
                
                # Consenso neutral
                if prom < 0.1:
                    Resultados[i] = 0
                
                # Consenso radicalizado
                else:
                    Resultados[i] = 1
                    
            
            # Casos de dos extremos
            elif sx >= 0.1 and sy < 0.1:
                # Dos extremos horizontal
                Resultados[i] = 2
            elif sx < 0.1 and sy >= 0.1:
                # Dos extremos vertical
                Resultados[i] = 3
                
            else:
                if ent < 0.18:
                    # Dos extremos ideológico
                    Resultados[i] = 4
                elif ent < 0.23:
                    # Estados de Transición
                    Resultados[i] = 5
                else:
                    # Cuatro extremos
                    Resultados[i] = 6
        
        else:
            
            # Estos son los casos con anchura
            
            # Casos de dos extremos
            if sx >= 0.1 and sy < 0.1:
                # Dos extremos horizontal
                Resultados[i] = 7
            elif sx < 0.1 and sy >= 0.1:
                # Dos extremos vertical
                Resultados[i] = 8
            
            else:
                # Polarización
                # Polarización ideológica
                if np.abs(cov) >= 0.1:
                    Resultados[i] = 9
                    
                # Polarización descorrelacionada
                else:
                    Resultados[i] = 10
                
    return Resultados


#-----------------------------------------------------------------------------------------------

# Esta función arma las curvas de fracción de estados
# en función del parámetro Y. La fracción de estados
# se refiere a los estados numerados de 0 a 9.

def Fraccion_estados_vs_Y(DF,path,carpeta):
    
    # Defino los arrays de parámetros diferentes    
    EXTRAS = int(np.unique(DF["Extra"]))
    PARAM_X = float(np.unique(DF["parametro_x"]))
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(j,param_y) for j,param_y in enumerate(Arr_param_y)]
    
    # Construyo el array de fracción estados polarizados
    Fraccion_polarizados = np.zeros((11,Arr_param_y.shape[0]))
    
    #--------------------------------------------------------------------------------
    
    # Diccionario con la entropía, Sigma_x y Sigma_y de todas las simulaciones
    # para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20)
    
    for indice,PARAM_Y in Tupla_total:
                
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        for estado in range(11):
            Fraccion_polarizados[estado,indice] = np.count_nonzero(Frecuencias == estado)/Frecuencias.shape[0]

    for estado in range(11):
        # Con los promedios armos el gráfico de opiniones vs T para ambos tópicos
        direccion_guardado = Path("../../../Imagenes/{}/Fraccion_EF{}.png".format(carpeta,estado))
        plt.rcParams.update({'font.size': 24})
        plt.figure("FracEstado",figsize=(20,15))
        plt.plot(Arr_param_y,Fraccion_polarizados[estado],"--", label = "estado {}".format(estado),linewidth=4)
            
        # Guardo la figura y la cierro
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$f_p$")
        plt.grid()
        # plt.legend()
        plt.title("Fracción de estados {} finales".format(estado))
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("FracEstado")
        
#-----------------------------------------------------------------------------------------------

# Esta función arma las curvas de fracción de estados
# en función del parámetro Y. La fracción de estados
# se refiere a los estados numerados de 0 a 9.

def Fraccion_dominante_vs_Y(DF,path,carpeta):
    
    # Defino los arrays de parámetros diferentes    
    EXTRAS = int(np.unique(DF["Extra"]))
    PARAM_X = float(np.unique(DF["parametro_x"]))
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(j,param_y) for j,param_y in enumerate(Arr_param_y)]
    
    # Construyo el array de fracción estados polarizados
    Fraccion_polarizados = np.zeros((11,Arr_param_y.shape[0]))
    
    #--------------------------------------------------------------------------------
    
    # Diccionario con la entropía, Sigma_x y Sigma_y de todas las simulaciones
    # para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20)
    
    for indice,PARAM_Y in Tupla_total:
                
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        for estado in range(11):
            Fraccion_polarizados[estado,indice] = np.count_nonzero(Frecuencias == estado)/Frecuencias.shape[0]

    
    # Armo los gráficos de fracción de estados para mis cuatro conjuntos de estados.
    
    # Estados de Consenso 
    
    ConsNeut = Fraccion_polarizados[0]
    ConsRad = Fraccion_polarizados[1]
    
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion Consenso.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracCons",figsize=(20,15))
    plt.plot(Arr_param_y,ConsNeut,"--",color = "tab:blue", label = "Consenso Neutral",linewidth=4)
    plt.plot(Arr_param_y,ConsRad,"--",color = "tab:green", label = "Consenso Radicalizado",linewidth=4)
    plt.plot(Arr_param_y,ConsNeut + ConsRad,color = "tab:orange", label = "Consenso Total",linewidth=5)
        
        
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.grid()
    plt.legend()
    plt.title("Estados finales de Consenso")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracCons")
    
    
    # Estados de Polarización Unidimensional
    
    Pol_Uni_sA = Fraccion_polarizados[2]+Fraccion_polarizados[3]
    Pol_Uni_cA = Fraccion_polarizados[7]+Fraccion_polarizados[8]
    
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion Pol Uni.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracPolUni",figsize=(20,15))
    plt.plot(Arr_param_y,Pol_Uni_sA,"--",color = "tab:blue", label = "Polarización sin Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Uni_cA,"--",color = "tab:green", label = "Polarización con Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Uni_sA + Pol_Uni_cA,color = "tab:orange", label = "Polarización total",linewidth=5)
    
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.grid()
    plt.legend()
    plt.title("Estados finales de Polarización Unidimensional")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracPolUni")
    
    # Estados de Polarización Ideológica
    
    Pol_Id_sA = Fraccion_polarizados[4]
    Pol_Id_cA = Fraccion_polarizados[9]
    
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion Pol Id.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracPolId",figsize=(20,15))
    plt.plot(Arr_param_y,Pol_Id_sA,"--",color = "tab:blue", label = "Polarización sin Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Id_cA,"--",color = "tab:green", label = "Polarización con Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Id_sA + Pol_Id_cA,color = "tab:orange", label = "Polarización total",linewidth=5)
        
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.grid()
    plt.legend()
    plt.title("Estados finales de Polarización Ideológica")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracPolId")
    
    # Estados de Polarización Descorrelacionada
    
    Pol_Des_sA = Fraccion_polarizados[6]
    Pol_Des_cA = Fraccion_polarizados[10]
    
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion Pol Des.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracPolDes",figsize=(20,15))
    plt.plot(Arr_param_y,Pol_Des_sA,"--",color = "tab:blue", label ="Polarización sin Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Des_cA,"--",color = "tab:green", label ="Polarización con Anchura",linewidth=4)
    plt.plot(Arr_param_y,Pol_Des_sA + Pol_Des_cA,color = "tab:orange", label ="Polarización Total",linewidth=5)
        
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.grid()
    plt.legend()
    plt.title("Estados finales de Polarización Descorrelacionada")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracPolDes")
    
    # Estados de Transición
    
    Trans = Fraccion_polarizados[5]
    
    direccion_guardado = Path("../../../Imagenes/{}/Fraccion Trans.png".format(carpeta))
    plt.rcParams.update({'font.size': 24})
    plt.figure("FracTrans",figsize=(20,15))
    plt.plot(Arr_param_y,Trans,color = "tab:red",linewidth=4)
        
    # Guardo la figura y la cierro
    plt.xlabel(r"$\beta$")
    plt.grid()
    plt.title("Estados finales de Transición")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("FracTrans")
    
