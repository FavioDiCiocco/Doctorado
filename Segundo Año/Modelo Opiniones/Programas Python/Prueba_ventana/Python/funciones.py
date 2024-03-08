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

# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_opi_vs_tiempo(DF,path,carpeta,T,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_iteraciones = np.unique(DF["iteracion"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y,int(iteracion)) for param_x in Arr_param_x
                   for param_y in Arr_param_y
                   for iteracion in Arr_iteraciones]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y,REP in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y) &
                                        (DF["iteracion"]==REP), "nombre"])
    
            #-----------------------------------------------------------------------------------------
            
            if len(archivos) > 1:
                
                Testigos_Total = np.empty((0,AGENTES*T))
                
                for orden in range(1,len(archivos)+1):
                    for nombre in archivos:
                        
                        cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                        
                        if cont == orden:
                        
                            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
                            # Estos archivos tienen las opiniones de dos agentes.
                            
                            Datos = ldata(path / nombre)
                            
                            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                            
                            for i,fila in enumerate(Datos[1:-1:]):
                                Testigos[i] = fila[:-1]
                            
                            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                            
                            Testigos_Total = np.concatenate((Testigos_Total,Testigos),axis=0)
                            
                            break
                    
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                
                
                # Armo mi gráfico, lo guardo y lo cierro
                for topico in range(T):
                    direccion_guardado = Path("../../../Imagenes/{}/OpivsT_Iter={}_N={:.0f}_{}={:.1f}_{}={:.1f}_{}={:.1f}_Topico={}.png".format(carpeta,REP,AGENTES,
                                              ID_param_x,PARAM_X,ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,topico))
                    plt.rcParams.update({'font.size': 44})
                    plt.figure("Topico",figsize=(32,24))
                    X = np.arange(Testigos_Total.shape[0])*0.01+20
                    Agentes_graf = int(AGENTES/2)
                    for sujeto in range(Agentes_graf):
                        plt.plot(X,Testigos_Total[:,sujeto*T+topico],color = "tab:brown", linewidth = 2, alpha = 0.5)
                    plt.xlabel(r"Tiempo$(10^3)$")
                    plt.ylabel(r"$x^i$")
                    plt.title("Evolución temporal Tópico {}, {} Agentes graficados ".format(topico, Agentes_graf))
                    plt.grid(alpha = 0.5)
                    plt.savefig(direccion_guardado ,bbox_inches = "tight")
                    plt.close("Topico")


#-----------------------------------------------------------------------------------------------

# Esta función me construye el gráfico de densidad de trayectorias.

def Graf_densidad_trayectorias(DF,path,carpeta,T,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_iteraciones = np.unique(DF["iteracion"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y,int(iteracion)) for param_x in Arr_param_x
                   for param_y in Arr_param_y
                   for iteracion in Arr_iteraciones]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y,REP in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y) &
                                        (DF["iteracion"]==REP), "nombre"])
    
            #-----------------------------------------------------------------------------------------
            
            if len(archivos) > 1:
                
                Testigos_Total = np.empty((0,AGENTES*T))
                
                for orden in range(1,len(archivos)+1):
                    for nombre in archivos:
                        
                        cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                        
                        if cont == orden:
                        
                            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
                            # Estos archivos tienen las opiniones de dos agentes.
                            
                            Datos = ldata(path / nombre)
                            
                            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                            
                            for i,fila in enumerate(Datos[1:-1:]):
                                Testigos[i] = fila[:-1]
                                Testigos[i] = Testigos[i]/EXTRAS
                            
                            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                            
                            Testigos_Total = np.concatenate((Testigos_Total,Testigos),axis=0)
                            
                            break
                
                #--------------------------------------------------------------------------------
                
                # Armo mi gráfico, lo guardo y lo cierro
                bins = np.linspace(-1,1,21)
                
                for topico in range(T):
                    direccion_guardado = Path("../../../Imagenes/{}/Densidad_trayec_Iter={}_N={:.0f}_{}={:.1f}_{}={:.1f}_{}={:.1f}_Topico={}.png".format(carpeta,REP,AGENTES,
                                              ID_param_x,PARAM_X,ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,topico))
                    plt.rcParams.update({'font.size': 44})
                    plt.figure("Topico",figsize=(32,24))
                    
                    # Construyo las grillas que voy a necesitar para el pcolormesh.
                    X = np.arange(Testigos_Total.shape[0])*0.01+20
                    XX,YY = np.meshgrid(X,np.flip(np.linspace(-0.95,0.95,20)))
                    ZZ = np.zeros(XX.shape)
                    
                    # Guardo los valores de las fracciones de trayectorias
                    for tiempo in range(Testigos_Total.shape[0]):
                        dens, descarte = np.histogram(Testigos_Total[tiempo,topico::T],bins=bins)
                        ZZ[:,tiempo] = np.flip(dens) / AGENTES
                    
                    # Ploteo mis datos
                    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "BuGn")
                    plt.colorbar()
                    
                    plt.xlabel(r"Tiempo$(10^3)$")
                    plt.ylabel(r"$x^i$")
                    plt.title("Densidad de trayectorias tópico {}".format(topico))
                    plt.savefig(direccion_guardado ,bbox_inches = "tight")
                    plt.close("Topico")


#-----------------------------------------------------------------------------------------------             

# Esta función es la que arma los gráficos de los histogramas de opiniones
# finales en el espacio de tópicos

def Graf_Histograma_opiniones_2D(DF,path,carpeta,bins,cmap,
                                 ID_param_x,ID_param_y,ID_param_extra_1):
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
                Opifinales = np.array(Datos[5][:-1:], dtype="float")
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                ventana = int(DF.loc[DF["nombre"]==nombre,"ventana"])
                direccion_guardado = Path("../../../Imagenes/{}/Hist_opi_2D_vent={}.png".format(carpeta,ventana))
                # Armo mi gráfico, lo guardo y lo cierro
                
                plt.rcParams.update({'font.size': 44})
                plt.figure(figsize=(20,15))
                _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                                         range=[[-PARAM_X,PARAM_X],[-PARAM_X,PARAM_X]],density=True,
                                         cmap=cmap)
                plt.xlabel(r"$x_i^1$")
                plt.ylabel(r"$x_i^2$")
                plt.title('Histograma 2D ventana {}'.format(ventana))
                cbar = plt.colorbar(im, label='Frecuencias')
                cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close()

#-----------------------------------------------------------------------------------------------
                
# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Histograma_Varianza_vs_Promedio(DF,path,carpeta,T,bins,cmap,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_iteraciones = np.unique(DF["iteracion"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y,int(iteracion)) for param_x in Arr_param_x
                   for param_y in Arr_param_y
                   for iteracion in Arr_iteraciones]
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y,REP in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y) &
                                        (DF["iteracion"]==REP), "nombre"])
    
            #-----------------------------------------------------------------------------------------
            
            Testigos_Total = np.empty((0,AGENTES*T))
            
            for orden in range(1,len(archivos)+1):
                for nombre in archivos:
                    
                    cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                    
                    if cont == orden:
                    
                        # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
                        # Estos archivos tienen las opiniones de dos agentes.
                        
                        Datos = ldata(path / nombre)
                        
                        Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                        
                        for i,fila in enumerate(Datos[1:-1:]):
                            Testigos[i] = fila[:-1]
                            Testigos[i] = Testigos[i]/EXTRAS
                        
                        # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                        
                        Testigos_Total = np.concatenate((Testigos_Total,Testigos),axis=0)
                        
                        break
                
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Una vez que tengo mi array de Testigos_Total, lo que hago es calcular la Varianza y Promedio de todos los agentes
            # en cada uno de los tópicos.
            
            Varianza = np.var(Testigos_Total,axis=0)
            Promedio = np.mean(Testigos_Total,axis=0)
            
            #----------------------------------------------------------------------------------------------------------------------------------
                
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            direccion_guardado = Path("../../../Imagenes/{}/Hist_Var_Opi_sim={}_N={:.0f}_{}={:.1f}_{}={:.1f}_{}={:.1f}.png".format(carpeta,REP,AGENTES,ID_param_x,PARAM_X,
                                          ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS))
            
            # Armo mi gráfico, lo guardo y lo cierro
            
            plt.rcParams.update({'font.size': 44})
            plt.figure(figsize=(28,21))
            _, _, _, im = plt.hist2d(Varianza, Promedio, bins=bins,density=True,cmap=cmap)
            plt.xlabel("Varianza")
            plt.ylabel("Promedio")
            plt.title('Histograma Varianza vs Promedio Opiniones, {}={:.1f}_{}={:.1f}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y,))
            cbar = plt.colorbar(im, label='Frecuencias')
#            cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()


#-----------------------------------------------------------------------------------------------
                
# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Fraccion_vs_Varianza(DF,path,carpeta,T,ID_param_x,ID_param_y,
                         ID_param_extra_1):
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_iteraciones = np.unique(DF["iteracion"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y,int(iteracion)) for param_x in Arr_param_x
                   for param_y in Arr_param_y
                   for iteracion in Arr_iteraciones]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y,REP in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y) &
                                        (DF["iteracion"]==REP), "nombre"])
    
            #-----------------------------------------------------------------------------------------
            if len(archivos) > 1:
                
                Testigos_Total = np.empty((0,AGENTES*T))
                
                for orden in range(1,len(archivos)+1):
                    for nombre in archivos:
                        
                        cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                        
                        if cont == orden:
                        
                            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
                            # Estos archivos tienen las opiniones de dos agentes.
                            
                            Datos = ldata(path / nombre)
                            
                            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                            
                            for i,fila in enumerate(Datos[1:-1:]):
                                Testigos[i] = fila[:-1]
                                Testigos[i] = Testigos[i]/EXTRAS
                            
                            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                            
                            Testigos_Total = np.concatenate((Testigos_Total,Testigos),axis=0)
                            
                            break
                    
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Una vez que tengo mi array de Testigos_Total, lo que hago es calcular la Varianza y Promedio de todos los agentes
                # en cada uno de los tópicos.
                
                Varianza = np.var(Testigos_Total,axis=0)
                
                #----------------------------------------------------------------------------------------------------------------------------------
                    
                # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
                direccion_guardado = Path("../../../Imagenes/{}/Fraccion_Varianza_sim={}_N={:.0f}_{}={:.1f}_{}={:.1f}_{}={:.1f}.png".format(carpeta,REP,AGENTES,ID_param_x,PARAM_X,
                                              ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS))
                
                colores = ["tab:blue", "tab:orange"]
                
                plt.rcParams.update({'font.size': 44})
                plt.figure(figsize=(28,21))
                
                for topico in range(T):
                    
                    X = np.logspace(-11,-1,num=50)
                    Y = np.zeros(X.shape[0])
                    
                    for i,x in enumerate(X):
                        Y[i] = np.count_nonzero(Varianza[topico::T] >= x) / AGENTES
                    
                    # Armo mi gráfico, lo guardo y lo cierro
                    plt.semilogx(X,Y,linestyle="--",color=colores[topico], label ="Topico {}".format(topico),linewidth=8)
                    plt.axvline(x=10**(-6), color = "blue", linewidth = 4)
                    plt.text(0.55, 0.4+0.1*(1-topico),
                             r'Fracción tópico {} = {}'.format(topico ,np.count_nonzero(Varianza[topico::T] >= 10**(-6)) / AGENTES), transform=plt.gcf().transFigure)
                
                plt.xlabel("Varianza")
                plt.ylabel("Fracción")
                plt.grid()
                plt.legend()
                plt.title('Fraccion de agentes en función de la varianza, {}={:.1f}_{}={:.1f}'.format(ID_param_x,PARAM_X,ID_param_y,PARAM_Y,))
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close()


#-----------------------------------------------------------------------------------------------

# Esta función me construye el gráfico de la variación promedio en función del tiempo

def Varprom_vs_T(DF,path,carpeta,ID_param_x,ID_param_y,
                 ID_param_extra_1):
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    
    Arr_EXTRAS = np.unique(DF["Extra"])
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    Arr_iteraciones = np.unique(DF["iteracion"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y,int(iteracion)) for param_x in Arr_param_x
                   for param_y in Arr_param_y
                   for iteracion in Arr_iteraciones]
    
    #----------------------------------------------------------------------------------------------------
    # Me armo el vector de Testigos_Total, el cuál voy a necesitar para calcular
    # la variación promedio post simulación
    
    
    for EXTRAS in Arr_EXTRAS:
        for PARAM_X,PARAM_Y,REP in Tupla_total:
            
            # Defino el tipo de archivo del cuál tomaré los datos
            TIPO = "Testigos"
            T=2
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Extra"]==EXTRAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y) &
                                        (DF["iteracion"]==REP), "nombre"])
    
            #-----------------------------------------------------------------------------------------
            if len(archivos) > 1:
                
                Testigos_Total = np.empty((0,AGENTES*T))
                
                for orden in range(1,len(archivos)+1):
                    for nombre in archivos:
                        
                        cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                        
                        if cont == orden:
                        
                            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
                            # Estos archivos tienen las opiniones de dos agentes.
                            
                            Datos = ldata(path / nombre)
                            
                            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
                            
                            for i,fila in enumerate(Datos[1:-1:]):
                                Testigos[i] = fila[:-1]
                                Testigos[i] = Testigos[i]/EXTRAS
                            
                            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
                            
                            Testigos_Total = np.concatenate((Testigos_Total,Testigos),axis=0)
                            
                            break
        
        #---------------------------------------------------------------------------------------------------
        
                    
                # Defino el tipo de archivo del cuál tomaré los datos
                TIPO = "Opiniones"
        
                # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
                archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                            (DF["n"]==AGENTES) & 
                                            (DF["Extra"]==EXTRAS) & 
                                            (DF["parametro_x"]==PARAM_X) &
                                            (DF["parametro_y"]==PARAM_Y) &
                                            (DF["iteracion"]==REP), "nombre"])
        
                #-----------------------------------------------------------------------------------------
                
                Varprom = np.empty(0)
                
                for orden in range(len(archivos)+1):
                    for nombre in archivos:
                        
                        cont = int(DF.loc[DF["nombre"]==nombre,"continuacion"])
                        
                        if cont == orden:
                        
                            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                            # Opinión Inicial del sistema
                            # Variación Promedio
                            # Opinión Final
                            # Semilla
                            
                            Datos = ldata(path / nombre)
                            
                            Varprom = np.concatenate((Varprom,np.array(Datos[3][:-1], dtype=float)))
                            
                            break
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Ahora construyo mi Varprom post simulación a partir de los datos de Testigos_Total
                # Voy a primero armar un cálculo de Variación Promedio restando cada fila con la anterior.
                # Esto se debería parecer a lo hecho durante la simulación, porque estoy restando estados
                # separados 100 pasos entre ellos, aunque en el código lo que hace es construir un promedio
                # de las 100 iteraciones y resta esos dos promedios. Pero un poco debería parecerse.
                
                Varprom_testigos = np.zeros(Testigos_Total.shape[0]-1)
                
                for tiempo in range(Varprom_testigos.shape[0]):
                    Norma = np.linalg.norm(Testigos_Total[tiempo+1,:]-Testigos_Total[tiempo,:])
                    Varprom_testigos[tiempo] = Norma / np.sqrt(AGENTES*T)
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Ya que estoy, pongo a prueba una idea que tengo sobre esto, intento graficar promediando
                # cada 1000 pasos. Que en este caso se resuelve promediando 10 filas y restándole las
                # anteriores 10 promediadas. Veamos que pasa en ese caso
                
                Varprom_mil_pasos = np.zeros(math.floor(Testigos_Total.shape[0]/10)-1)
                T_mil = np.zeros(Varprom_mil_pasos.shape[0])
                
                for tiempo in range(Varprom_mil_pasos.shape[0]):
                    Norma = np.linalg.norm(np.mean(Testigos_Total[(tiempo+1)*10:(tiempo+2)*10,:],axis=0)-np.mean(Testigos_Total[tiempo*10:(tiempo+1)*10,:], axis=0))
                    Varprom_mil_pasos[tiempo] = Norma / np.sqrt(AGENTES*T)
                    T_mil[tiempo] = tiempo*0.1+20
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Armo mi gráfico, lo guardo y lo cierro
                direccion_guardado = Path("../../../Imagenes/{}/Varprom_Iter={}_N={:.0f}_{}={:.1f}_{}={:.1f}_{}={:.1f}.png".format(carpeta,REP,AGENTES,
                                          ID_param_x,PARAM_X,ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS))
                
                plt.rcParams.update({'font.size': 44})
                plt.figure("Varprom",figsize=(32,24))
                T = np.arange(Varprom.shape[0])*0.01
                plt.semilogy(T[0:5000:10],Varprom[0:5000:10],color = "tab:purple", label="Simulación" ,linewidth = 6, alpha = 0.9)
                plt.semilogy(T[0:3000:10]+20,Varprom_testigos[0:3000:10],color = "tab:blue", label="Testigos" ,linewidth = 6, alpha = 0.9)
                plt.semilogy(T_mil[0:300],Varprom_mil_pasos[0:300],color = "tab:orange", label="Ventana Ancha" ,linewidth = 6, alpha = 0.9)
                plt.axhline(0.001,linestyle = "--", color = "red")
                plt.xlabel(r"Tiempo$(10^3)$")
                plt.ylabel("Variación Promedio")
                plt.title("Curva de Variación promedio vs T")
                plt.grid()
                plt.legend()
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close("Varprom")


#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Calculo_Entropia(DF,path,N):
    
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
            
            entropias = np.zeros(archivos.shape[0])
            
            for indice,nombre in enumerate(archivos):
        
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
        
#                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,N,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                entropias[indice] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)+0.00001
        
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y] = entropias[entropias != 0]/np.log2(N*N)
    
    return Salida

#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando el valor medio de la opinión.

def Mapa_Colores_Promedio_opiniones(DF,path,carpeta,T,
                                    SIM_param_x,SIM_param_y,
                                    SIM_param_extra_1,ID_param_extra_1,
                                    Condicion_curvas_kappa=False):
    
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
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    
    # Separo la construcción del ZZ para poder meterla en un for de los tópicos
    
    for topico in range(T):
        
        #--------------------------------------------------------------------------------
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            # Me defino el array en el cual acumulo los datos de las opiniones finales de todas
            # mis simulaciones
            Opifinales = np.array([])
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Kappas"]==KAPPAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])      
    
            #------------------------------------------------------------------------------------------
            
            for nombre in archivos:
                
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
                
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                
                # Leo los datos de las Opiniones Finales
                Opifinales = np.concatenate((Opifinales, np.array(Datos[5][:-1:], dtype="float")), axis = None)
            
            #------------------------------------------------------------------------------------------
            # Voy a primero tomar el promedio de opiniones de una simulación, luego a eso tomarle
            # el valor absoluto y por último promediarlo
            
            # Armo un array donde me guardo el promedio de cada una de mis simulaciones
            Promedios = np.zeros(archivos.shape[0])
            
            for simulacion in range(archivos.shape[0]):
                Promedios[simulacion] = np.mean(Opifinales[AGENTES*T*simulacion+topico:AGENTES*T*(simulacion+1)+topico:2])
            
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(np.abs(Promedios))
        
        #--------------------------------------------------------------------------------
        
        # Una vez que tengo el ZZ completo, armo mi mapa de colores
        direccion_guardado = Path("../../../Imagenes/{}/Promedio Opiniones Topico {}.png".format(carpeta,topico))
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("Promedio Opiniones",figsize=(20,15))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.title(r"Promedio de opiniones Topico {}".format(topico))
        
        # Guardo la figura y la cierro
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Promedio Opiniones")
    

#-----------------------------------------------------------------------------------------------

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
    KAPPAS = int(np.unique(DF["Kappas"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    ZZ = np.zeros(XX.shape)
    
    #--------------------------------------------------------------------------------
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Kappas"]==KAPPAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])         
        
        # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
        Tiempos = np.zeros(len(archivos))

        #------------------------------------------------------------------------------------------
        
        for indice,nombre in enumerate(archivos):
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            
            # Leo los datos de las Opiniones Finales
            Tiempos[indice] = len(Datos[3])
        
        #------------------------------------------------------------------------------------------
        # Con los tiempos de las simulaciones calculo el promedio de los tiempos de convergencia
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.log(np.mean(Tiempos))
        
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Tiempo_Convergencia_{}={}.png".format(carpeta,ID_param_extra_1,KAPPAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Tiempo_Convergencia",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Tiempo de Convergencia en Espacio de Parametros")

    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Tiempo_Convergencia")


#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Mapa_Colores_Traza_Covarianza(DF,path,carpeta,
                       SIM_param_x,SIM_param_y,
                       ID_param_extra_1):
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

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
        
        for indice,nombre in enumerate(archivos):
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Semilla
            
            # Levanto los datos del archivo
            Datos = ldata(path / nombre)
            if len(Datos) < 7:
                continue
            
            # Leo los datos de las Opiniones Finales
            Opifinales = np.zeros((T,AGENTES))
            
            # Normalizo mis datos usando el valor de Kappa
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/EXTRAS
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            M_cov = np.cov(Opifinales)
            Covarianzas[indice] = np.trace(M_cov)/T
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Covarianzas[Covarianzas != 0])
            
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Traza_Covarianza_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Traza_Covarianza",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Traza Matriz Covarianza en Espacio de Parametros")
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Traza_Covarianza")
    


#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Calculo_Traza_Covarianza(DF,path):
    
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
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    Salida = dict()
    for KAPPAS in Arr_KAPPAS:
        Salida[KAPPAS] = dict()
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
            archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                        (DF["n"]==AGENTES) & 
                                        (DF["Kappas"]==KAPPAS) & 
                                        (DF["parametro_x"]==PARAM_X) &
                                        (DF["parametro_y"]==PARAM_Y), "nombre"])
            #-----------------------------------------------------------------------------------------
            
            covarianzas = np.zeros(archivos.shape[0])
            
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
        
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
        
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
        
                M_cov = np.cov(Opifinales)
                covarianzas[repeticion] = np.trace(M_cov)/(2*(KAPPAS-1)*(KAPPAS-1))
        
            if PARAM_X not in Salida[KAPPAS].keys():
                Salida[KAPPAS][PARAM_X] = dict()
            Salida[KAPPAS][PARAM_X][PARAM_Y] = covarianzas
    
    return Salida

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
            
            for indice,nombre in enumerate(archivos):
                
        
                # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                # Opinión Inicial del sistema
                # Variación Promedio
                # Opinión Final
                # Semilla
        
                # Levanto los datos del archivo
                Datos = ldata(path / nombre)
                if len(Datos)< 7:
                    continue
                
                # Leo los datos de las Opiniones Finales
                Opifinales = np.zeros((T,AGENTES))
        
                for topico in range(T):
                    Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")
                    Opifinales[topico,:] = Opifinales[topico,:]/ EXTRAS
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
        
#                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
        
                M_cov = np.cov(Opifinales)
                Varianza_X[indice] = M_cov[0,0]
                Varianza_Y[indice] = M_cov[1,1]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales/EXTRAS
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,N,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                Entropia[indice] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
                
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            if PARAM_Y not in Salida[EXTRAS][PARAM_X].keys():
                Salida[EXTRAS][PARAM_X][PARAM_Y] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Entropia"] = Entropia[Entropia != 0]/np.log2(N*N)
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"] = Varianza_X[Varianza_X != 0]
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"] = Varianza_Y[Varianza_Y != 0]
            
    return Salida

#-----------------------------------------------------------------------------------------------

def Identificacion_Estados(Entropia, Sigma_X, Sigma_Y):
    
    Resultados = np.zeros(len(Entropia))
    
    for i,ent,sx,sy in zip(np.arange(len(Entropia)),Entropia,Sigma_X,Sigma_Y):
        
        # Reviso la entropía y separo en casos con y sin anchura
        
        if ent <= 0.3:
            
            # Estos son casos sin anchura
            
            if sx < 0.1 and sy < 0.1:
                # Caso de un sólo extremo
                Resultados[i] = 0
            
            # Casos de dos extremos
            elif sx >= 0.1 and sy < 0.1:
                # Dos extremos horizontal
                Resultados[i] = 1
            elif sx < 0.1 and sy >= 0.1:
                # Dos extremos vertical
                Resultados[i] = 2
                
            else:
                if ent < 0.18:
                    # Dos extremos ideológico
                    Resultados[i] = 3
                elif ent < 0.22:
                    # Tres extremos
                    Resultados[i] = 4
                else:
                    # Cuatro extremos
                    Resultados[i] = 5
        
        else:
            
            # Estos son los casos con anchura
            
            if sx < 0.1 and sy < 0.1:
                # Caso de un sólo extremo
                Resultados[i] = 6
            
            # Casos de dos extremos
            elif sx >= 0.1 and sy < 0.1:
                # Dos extremos horizontal
                Resultados[i] = 7
            elif sx < 0.1 and sy >= 0.1:
                # Dos extremos vertical
                Resultados[i] = 8
            
            else:
                # Dos extremos ideológico, tres extremos y cuatro extremos
                Resultados[i] = 9
                
    return Resultados


#-----------------------------------------------------------------------------------------------
# Esta función arma todos los mapas de colores de frecuencias de los estados finales.    

def Mapas_Colores_FEF(DF,path,carpeta,
                       SIM_param_x,SIM_param_y,
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
    # Voy a armar 10 mapas de colores
    ZZ = np.zeros((10,XX.shape[0],XX.shape[1]))
    
    #--------------------------------------------------------------------------------
    
    # Diccionario con la entropía, Sigma_x y Sigma_y de todas las simulaciones
    # para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20)
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
                
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"])
        
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        for grafico in range(10):
            if Frecuencias.shape[0] != 0 :
                ZZ[grafico,(Arr_param_y.shape[0]-1)-fila,columna] = np.count_nonzero(Frecuencias == grafico)/Frecuencias.shape[0]
            else:
                ZZ[grafico,(Arr_param_y.shape[0]-1)-fila,columna] = 0.0001
            
    #--------------------------------------------------------------------------------
    
    for grafico in range(10):
        # Una vez que tengo el ZZ completo, armo mi mapa de colores
        direccion_guardado = Path("../../../Imagenes/{}/FEF{}_{}={}.png".format(carpeta,grafico,ID_param_extra_1,EXTRAS))
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("FEF",figsize=(20,15))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ[grafico],shading="nearest", cmap = "plasma")
        plt.colorbar()
        plt.title("Frecuencia del estado {}".format(grafico))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("FEF")

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



# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la entropía como métrica.

def Mapa_Colores_Entropia_opiniones(DF,path,carpeta,
                                    SIM_param_x,SIM_param_y,
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
    
    Entropias = Calculo_Entropia(DF, path, 20)
    
    #--------------------------------------------------------------------------------
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        
        #------------------------------------------------------------------------------------------
        # Armo mi matriz con los valores de entropía y con los valores de la varianza
        
        ZZ[0,(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Entropias[EXTRAS][PARAM_X][PARAM_Y])
        ZZ[1,(Arr_param_y.shape[0]-1)-fila,columna] = np.var(Entropias[EXTRAS][PARAM_X][PARAM_Y])
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Entropia EP_{}={}.png".format(carpeta,ID_param_extra_1,EXTRAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Entropia Opiniones",figsize=(20,15))
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
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Varianza Entropia",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ[1],shading="nearest", cmap = "magma")
    plt.colorbar()
    plt.title("Varianza de Entropía en Espacio de Parametros")
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Varianza Entropia")

