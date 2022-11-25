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
        f = open(archive)
        data = []
        for line in f:
            col = line.split("\t")
            col = [x.strip() for x in col]
            data.append(col)
        return data 

#--------------------------------------------------------------------------------

##################################################################################
##################################################################################

# FUNCIONES DE ANÁLISIS Y CLASIFICACIÓN

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

def Indice_Color(vector,Divisiones):
    # Primero calculo el ángulo
    Vhor = [1,0] # Este vector representa la horizontal
    if np.linalg.norm(vector) != 0 :
        vector_unitario = vector/np.linalg.norm(vector)
        Producto_escalar = np.dot(Vhor,vector_unitario)
        Angulo = np.arccos(Producto_escalar)

        # Le hago ajuste considerando el cuadrante del vector
        if vector[1] < 0:
            Angulo = 2*math.pi-Angulo


        # Ahora calculo el valor de división entera y el Resto
        Delta = (2*math.pi)/Divisiones
        Dividendo = Angulo/Delta
        D = math.floor(Dividendo)
        R = (Dividendo - D) * Delta

        # Compruebo en qué casillero cae el ángulo y returneo el índice
        if R <= Delta/2:
            return D # En este caso el ángulo se encuentra entre (D*Delta-Delta/2,D*Delta+Delta/2]
        elif R > Delta/2:
            return (D+1)%Divisiones # En este caso el ángulo se encuentra entre ((D+1)*Delta-Delta/2,(D+1)*Delta+Delta/2]
    else:
        return 0;

#--------------------------------------------------------------------------------

# Acá lo preparo los colores que voy a usar para definir los puntos finales de las trayectorias de las opiniones

Divisiones = 144
color=cm.rainbow(np.linspace(0,1,Divisiones))

# Lo que hice acá es definir una ¿lista? que tiene en cada casillero los datos que definen un color.
# Tiene diferenciados 144 colores, es decir que tengo un color para cada región de 2.5 grados. Estas regiones
# las voy a distribuir centrándolas en cada ángulo que cada color representa. Por lo tanto,
# Los vectores que tengan ángulo entre -1.25º y 1.25º tienen el primer color. Los que tengan entre
# 1.25º y 3.75º tienen el segundo color. Y así. Por tanto yo tengo que hallar una fórmula que para
# cada ángulo le asigne el casillero que le corresponde en el vector de color. Luego, cuando grafique
# el punto, para el color le agrego un input que sea: c = color[n]

#---------------------------------------------------------------------------------------------------------

##################################################################################
##################################################################################

# FUNCIONES GRAFICADORAS

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_opi_vs_tiempo(DF,path,carpeta,T=2):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    arrayN = np.unique(DF["n"])
    arrayAlfa = np.unique(DF["alfa"])[::math.floor((len(np.unique(DF["alfa"]))-1)/2)]
    arrayUmbral = np.unique(DF["umbral"])[::math.floor((len(np.unique(DF["umbral"]))-1)/2)]
    
    Tupla_total = [(n,alfa,umbral) for n in arrayN
                   for alfa in arrayAlfa
                   for umbral in arrayUmbral]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    for AGENTES,ALFA,UMBRAL in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["umbral"]==UMBRAL) & 
                                    (DF["alfa"]==ALFA), "nombre"])

        #-----------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
            # Estos archivos tienen las opiniones de dos agentes.
            
            Datos = ldata(path / nombre)
            
            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
            
            for i,fila in enumerate(Datos[1:-1:]):
                Testigos[i] = fila[:-1]
            
            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
            direccion_guardado = Path("../../../Imagenes/{}/OpivsT_N={:.0f}_alfa={:.1f}_umbral={:.1f}_sim={}.png".format(carpeta,AGENTES,ALFA,UMBRAL,repeticion))
            
            if repeticion in [0,1]:
            
                plt.rcParams.update({'font.size': 32})
                plt.figure("Topico",figsize=(20,15))
                X = np.arange(Testigos.shape[0])*0.01
                for sujeto in range(int(Testigos.shape[1]/T)):
                    for topico in range(T):
                        plt.plot(X,Testigos[:,sujeto*T+topico], linewidth = 6)
                plt.xlabel("Tiempo")
                plt.ylabel("Tópico")
                plt.grid(alpha = 0.5)
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close("Topico")

        
#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la varianza de las opiniones como métrica.

def Mapa_Colores_Varianza_opiniones(DF,path,carpeta):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    arrayN = np.unique(DF["n"])
    arrayAlfa = np.unique(DF["alfa"])
    arrayUmbral = np.unique(DF["umbral"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,alfa,j,umbral) for n in arrayN
                   for i,alfa in enumerate(arrayAlfa)
                   for j,umbral in enumerate(arrayUmbral)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(arrayUmbral,np.flip(arrayAlfa))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    
    # Itero en los valores de mis parámetros alfa y umbral.
    for AGENTES,fila,ALFA,columna,UMBRAL in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["umbral"]==UMBRAL) & 
                                    (DF["alfa"]==ALFA), "nombre"])        

        #------------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Matriz de Adyacencia
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.concatenate((Opifinales, np.array(Datos[5][:-1:], dtype="float")), axis = None)
        
        #------------------------------------------------------------------------------------------
        # Con las opiniones finales de todas las simulaciones lo que hago es calcular la varianza
        # de la distribución de opiniones.
        ZZ[arrayAlfa.shape[0]-1-fila,columna] = np.var(Opifinales)
        
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Varianza Opiniones EP.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Varianza Opiniones",figsize=(20,15))
    plt.xlabel("umbral")
    plt.ylabel(r"$\alpha$")
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Varianza de opiniones en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Varianza Opiniones")


#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la entropía como métrica.

def Mapa_Colores_Entropia_opiniones(DF,path,carpeta):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    arrayN = np.unique(DF["n"])
    arrayAlfa = np.unique(DF["alfa"])
    arrayUmbral = np.unique(DF["umbral"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,alfa,j,umbral) for n in arrayN
                   for i,alfa in enumerate(arrayAlfa)
                   for j,umbral in enumerate(arrayUmbral)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(arrayUmbral,np.flip(arrayAlfa))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    for AGENTES,fila,ALFA,columna,UMBRAL in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas
        # mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["umbral"]==UMBRAL) & 
                                    (DF["alfa"]==ALFA), "nombre"])        

        #------------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Matriz de Adyacencia
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.concatenate((Opifinales, np.array(Datos[5][:-1:], dtype="float")), axis = None)
        
        #------------------------------------------------------------------------------------------
        # Con las opiniones finales de todas las simulaciones lo que hago es calcular la entropia
        # de la distribución de las opiniones. Esto es una segunda métrica para ver si las opiniones
        # finales de los agentes están dispersas en dos puntos grandes, o no. Además, esto sería
        # sensible a cómo es esa distribución.
        
        ZZ[arrayAlfa.shape[0]-1-fila,columna] = Entropia(Opifinales)
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Entropia Opiniones EP.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Entropia Opiniones",figsize=(20,15))
    plt.xlabel("umbral")
    plt.ylabel(r"$\alpha$")
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.title("Entropía de opiniones en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Entropia Opiniones")


#-----------------------------------------------------------------------------------------------

def Entropia(Array):
    
    # Primero tengo que binnear mi distribución. Como sé que mi distribución va de 0 a 1,
    # voy a separar eso en 20 bines. Hist tiene la cantidad de datos que caen en ese bin.
    Hist,Bines = np.histogram(Array, bins = 20, range = (0,1))
    
    # Calculo la proba de que los valores de mi distribución caigan en cada bin
    Probas = Hist[Hist != 0] / Array.shape[0] # Saco los ceros para no calcularles logs
    
    # Calculo la entropía y la returneo
    return np.matmul(Probas, np.log2(Probas))*(-1)


#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la entropía como métrica.

def Grafico_histograma(DF,path,carpeta):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    arrayN = np.unique(DF["n"])
    arrayAlfa = np.unique(DF["alfa"])
    arrayUmbral = np.unique(DF["umbral"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,alfa,umbral) for n in arrayN
                   for i,alfa in enumerate(arrayAlfa)
                   for umbral in arrayUmbral]
    
    # Me preparo mis figuras antes de arrancar.
    plt.rcParams.update({'font.size': 24})
    for i in range(arrayAlfa.shape[0]):
        plt.figure(i,figsize=(20,15))
        plt.ylabel("Probabilidad")
        plt.xlabel("Interés")
        plt.grid()


    #--------------------------------------------------------------------------------
    
    
    for AGENTES,ialfa,ALFA,UMBRAL in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["umbral"]==UMBRAL) & 
                                    (DF["alfa"]==ALFA), "nombre"])        

        #------------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Matriz de Adyacencia
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.concatenate((Opifinales, np.array(Datos[5][:-1:], dtype="float")), axis = None)
            
        #------------------------------------------------------------------------------------------
        
        # Tomo los datos de opiniones finales, los distribuyo en un histograma y con eso armo una
        # curva. La idea es tener una curva para cada umbral.
        
        Hist,Bines = np.histogram(Opifinales, bins = 60, range = (0,1))
        Y = Hist / Opifinales.shape[0]
        X = (Bines[1::]+Bines[:-1])/2
        
        
        # Abro la figura correspondiente para graficarla
        plt.figure(ialfa)
        plt.plot(X,Y,"--", linewidth = 4,label = "Umbral = {}".format(UMBRAL))
        
    #-----------------------------------------------------------------------------------------------
    
    # Ya revisé todos los datos, ahora cierro los gráficos y los guardo
    
    for indice,ALFA in enumerate(arrayAlfa):
        
        # Defino el path de guardado del archivo
        direccion_guardado = Path("../../../Imagenes/{}/Histograma_alfa={}.png".format(carpeta,ALFA))
        
        # Hago los retoques finales, guardo las figuras y cierro todo.
        plt.figure(indice)
        plt.legend(ncols = 2)
        plt.title(" Alfa = {}".format(ALFA))
        plt.xlim(0,1)
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close(indice)