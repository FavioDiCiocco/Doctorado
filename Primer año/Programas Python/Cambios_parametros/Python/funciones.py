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

# Esta función revisa el máximo promedio de interés de los agentes y me lo grafica en función
# de los valores de olvido. 

def MaxProm_vs_olvido(DF,path,carpeta,T=2):
    # Primero defino los parámetros que voy a iterar al armar mis gráficos.
    
    arrayN = np.unique(DF["n"])
    arrayLambda = np.unique(DF["lambda"])
    
    Tupla_total = [(n,indice,olvido) for n in arrayN
                   for indice,olvido in enumerate(arrayLambda)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    # Armo mi array en el cuál pondré los valores de promedios de los máximos de interés
    Promedios = np.zeros(arrayLambda.shape[0])
    
    
    for AGENTES,ilambda,LAMBDA in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["lambda"]==LAMBDA), "nombre"])

        #-----------------------------------------------------------------------------------------
        
        # Armo mi array que voy a llenar con los máximos de interés
        # Para eso tengo que saber cuántos testigos tengo, así que
        # levanto un archivo cualquiera y lo saco de ahí.
        
        nombre = archivos[0]
        Datos = ldata(path / nombre)
        Cant_testigos = len(Datos[1])-1
        
        Maximos = np.zeros(archivos.shape[0]*Cant_testigos)
        
        for indice,nombre in enumerate(archivos):
            
            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
            # Estos archivos tienen las opiniones de dos agentes.
            
            Datos = ldata(path / nombre)
            
            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
            
            for i,fila in enumerate(Datos[1:-1:]):
                Testigos[i] = fila[:-1]
            
            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Ahora extraigo los máximos de cada agente
            
            for agente in range(Cant_testigos):
                Maximos[indice*Cant_testigos+agente] = Testigos[:,agente].max()

        #----------------------------------------------------------------------------------------------------------------------------------
        
        # Una vez que tengo todos los máximos anotados, les tomo el promedio y me lo guardo
        Promedios[ilambda] = np.mean(Maximos)
        
    #----------------------------------------------------------------------------------------------------------------------------------
    
    # Una vez que tengo armado el array de Promedios, grafico mis datos        
    direccion_guardado = Path("../../../Imagenes/{}/Promedios_vs_olvido.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 32})
    plt.figure("Promedios",figsize=(20,15))
    plt.semilogx(arrayLambda,Promedios, "--", linewidth = 6, color = "green")
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Promedio máximos interés")
    plt.grid(alpha = 0.5)
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close("Promedios")


# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_opi_vs_tiempo(DF,path,carpeta,T=2,nombre_parametro_1="parametro1",nombre_parametro_2="parametro2"):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    Ns = np.unique(DF["n"])
    Array_parametro_1 = np.unique(DF["parametro_1"])[[0,math.floor(len(np.unique(DF["parametro_1"]))/2),len(np.unique(DF["parametro_1"]))-1]]
    Array_parametro_2 = np.unique(DF["parametro_2"])[[0,math.floor(len(np.unique(DF["parametro_2"]))/2),len(np.unique(DF["parametro_2"]))-1]]
    
    Tupla_total = [(n,parametro_1,parametro_2) for n in Ns
                   for parametro_1 in Array_parametro_1
                   for parametro_2 in Array_parametro_2]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    for AGENTES,PARAMETRO_1,PARAMETRO_2 in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])

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
            direccion_guardado = Path("../../../Imagenes/{}/OpivsT_N={:.0f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,nombre_parametro_1,PARAMETRO_1,nombre_parametro_2,PARAMETRO_2,repeticion))
            
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

def Mapa_Colores_Varianza_opiniones(DF,path,carpeta,titulo_parametro_1="parametro 1" ,titulo_parametro_2="parametro 2"):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    arrayN = np.unique(DF["n"])
    Array_parametro_1 = np.unique(DF["parametro_1"])
    Array_parametro_2 = np.unique(DF["paramtero_2"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,parametro_1,j,parametro_2) for n in arrayN
                   for i,parametro_1 in enumerate(Array_parametro_1)
                   for j,parametro_2 in enumerate(Array_parametro_2)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Array_parametro_2,np.flip(Array_parametro_1))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    
    # Itero en los valores de mis parámetros alfa y umbral.
    for AGENTES,fila,PARAMETRO_1,columna,PARAMETRO_2 in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])        

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
        ZZ[Array_parametro_1.shape[0]-1-fila,columna] = np.var(Opifinales)
        
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Varianza Opiniones EP.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Varianza Opiniones",figsize=(20,15))
    plt.xlabel(r"${}$".format(titulo_parametro_2))
    plt.ylabel(r"${}$".format(titulo_parametro_1))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Varianza de opiniones en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Varianza Opiniones")


#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la entropía como métrica.

def Mapa_Colores_Entropia_opiniones(DF,path,carpeta,titulo_parametro_1="parametro 1",titulo_parametro_2="parametro 2"):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    Ns = np.unique(DF["n"])
    Array_parametro_1 = np.unique(DF["parametro_1"])
    Array_parametro_2 = np.unique(DF["paramtero_2"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,parametro_1,j,parametro_2) for n in Ns
                   for i,parametro_1 in enumerate(Array_parametro_1)
                   for j,parametro_2 in enumerate(Array_parametro_2)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Array_parametro_2,np.flip(Array_parametro_1))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    for AGENTES,fila,PARAMETRO_1,columna,PARAMETRO_2 in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas
        # mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])        

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
        
        ZZ[Array_parametro_1.shape[0]-1-fila,columna] = Entropia(Opifinales)
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Entropia Opiniones EP.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Entropia Opiniones",figsize=(20,15))
    plt.xlabel(r"${}$".format(titulo_parametro_2))
    plt.ylabel(r"${}$".format(titulo_parametro_1))
    
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

def Grafico_histograma(DF,path,carpeta,nombre_parametro_1="parametro_1",titulo_parametro_2="parametro_2"):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    arrayN = np.unique(DF["n"])
    Array_parametro_1 = np.unique(DF["parametro_1"])
    Array_parametro_2 = np.unique(DF["paramtero_2"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,parametro_1,parametro_2) for n in arrayN
                   for i,parametro_1 in enumerate(Array_parametro_1)
                   for parametro_2 in Array_parametro_2]
    
    # Me preparo mis figuras antes de arrancar.
    plt.rcParams.update({'font.size': 24})
    for i in range(Array_parametro_1.shape[0]):
        plt.figure(i,figsize=(20,15))
        plt.ylabel("Probabilidad")
        plt.xlabel("Interés")
        plt.grid()


    #--------------------------------------------------------------------------------
    
    
    for AGENTES,indice_parametro_1,PARAMETRO_1,PARAMETRO_2 in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])        

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
        plt.figure(indice_parametro_1)
        plt.plot(X,Y,"--", linewidth = 4,label = r"${}$ = {}".format(titulo_parametro_2,PARAMETRO_2))
        
    #-----------------------------------------------------------------------------------------------
    
    # Ya revisé todos los datos, ahora cierro los gráficos y los guardo
    
    for indice,PARAMETRO_1 in enumerate(Array_parametro_1):
        
        # Defino el path de guardado del archivo
        direccion_guardado = Path("../../../Imagenes/{}/Histograma_{}={}.png".format(carpeta,nombre_parametro_1,PARAMETRO_1))
        
        # Hago los retoques finales, guardo las figuras y cierro todo.
        plt.figure(indice)
        # plt.legend(ncols = 2)
        plt.title("{} = {}".format(nombre_parametro_1,PARAMETRO_1))
        plt.xlim(0,1)
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close(indice)

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando el valor medio de la opinión.

def Mapa_Colores_Promedio_opiniones(DF,path,carpeta,titulo_parametro_1="parametro 1" ,titulo_parametro_2="parametro 2"):
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Defino los arrays de parámetros diferentes
    
    Ns = np.unique(DF["n"])
    Array_parametro_1 = np.unique(DF["parametro_1"])
    Array_parametro_2 = np.unique(DF["parametro_2"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(n,i,parametro_1,j,parametro_2) for n in Ns
                   for i,parametro_1 in enumerate(Array_parametro_1)
                   for j,parametro_2 in enumerate(Array_parametro_2)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Array_parametro_2,np.flip(Array_parametro_1))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    for AGENTES,fila,PARAMETRO_1,columna,PARAMETRO_2 in Tupla_total:
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas
        # mis simulaciones
        Opifinales = np.array([])
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])        

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
        # Con las opiniones finales de todas las simulaciones lo que hago es calcular el promedio de
        # las opiniones. No hago distinción de tópicos porque considero que los agentes tenderán
        # a los mismos valores en todos sus tópicos.
        
        ZZ[Array_parametro_1.shape[0]-1-fila,columna] = np.mean(Opifinales)
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Promedio Opiniones EP.png".format(carpeta))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Promedio Opiniones",figsize=(20,15))
    plt.xlabel(r"${}$".format(titulo_parametro_2))
    plt.ylabel(r"${}$".format(titulo_parametro_1))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "cividis")
    plt.colorbar()
    plt.title("Promedio de opiniones en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Promedio Opiniones")
    
    
#--------------------------------------------------------------------------------

# Esta función me construye el gráfico de Saturación en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_sat_vs_tiempo(DF,path,carpeta,T=2):
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
    TIPO = "Saturacion"
    
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
            direccion_guardado = Path("../../../Imagenes/{}/SatvsT_N={:.0f}_alfa={:.1f}_umbral={:.1f}_sim={}.png".format(carpeta,AGENTES,ALFA,UMBRAL,repeticion))
            
            if repeticion in [0,1]:
            
                plt.rcParams.update({'font.size': 32})
                plt.figure("Saturacion",figsize=(20,15))
                X = np.arange(Testigos.shape[0])*0.01
                for sujeto in range(int(Testigos.shape[1]/T)):
                    for topico in range(T):
                        plt.plot(X,Testigos[:,sujeto*T+topico], linewidth = 6)
                plt.xlabel("Tiempo")
                plt.ylabel("Saturación")
                plt.grid(alpha = 0.5)
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close("Saturacion")

