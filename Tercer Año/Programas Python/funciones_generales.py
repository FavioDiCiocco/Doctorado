# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 11:33:00 2022

@author: Favio
"""

# Este archivo es para definir funciones

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.spatial.distance import jensenshannon
import numpy as np
import time
import math
import pandas as pd
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
        f = open(archive)
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

# FUNCIONES ANALÍTICAS

##################################################################################
##################################################################################

#--------------------------------------------------------------------------------

# Defino las funciones que uso para calcular los puntos críticos y los Kappa

def Derivada_kappa(x,alfa,epsilon):
    return np.exp(alfa*x-epsilon)+1-alfa*x

def Kappa(x,alfa,epsilon):
    return x*( 1 + np.exp(-alfa*x +epsilon) )

def Ecuacion_dinamica(x,K,A,Cdelta,Eps):
    return -x+K*(1/(1+np.exp(-A*(1+Cdelta)*x+Eps)))

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

#-----------------------------------------------------------------------------------------------

# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_opi_vs_tiempo(DF,path,carpeta,T=2,
                       nombre_parametro_1="parametro1",nombre_parametro_2="parametro2"):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    Ns = np.unique(DF["n"])
    
    """
    # Defino los valores de Parametro_1 que planeo graficar
    Valores_importantes_1 = [0,math.floor(len(np.unique(DF["parametro_1"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_1"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_1"]))/4),
                            len(np.unique(DF["parametro_1"]))-1]
    
    # Defino los valores de Parametro_2 que planeo graficar
    Valores_importantes_2 = [0,math.floor(len(np.unique(DF["parametro_2"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_2"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_2"]))/4),
                            len(np.unique(DF["parametro_2"]))-1]
    
    # Armo los arrays de mis parámetros y después armo la Tupla_Total
    Array_parametro_1 = np.unique(DF["parametro_1"])[Valores_importantes_1]
    Array_parametro_2 = np.unique(DF["parametro_2"])[Valores_importantes_2]
    
    #-----------------------------------------------------------------------------
    """
    
    # Armo los arrays de mis parámetros y después armo la Tupla_Total
    
    Array_parametro_1 = np.unique(DF["parametro_1"])
    Array_parametro_2 = np.unique(DF["parametro_2"])

    
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
            
            # Armo mi gráfico, lo guardo y lo cierro
            
            plt.rcParams.update({'font.size': 32})
            plt.figure("Topico",figsize=(20,15))
            X = np.arange(Testigos.shape[0])*0.01
            for sujeto in range(int(Testigos.shape[1]/T)):
                for topico in range(T):
                    plt.plot(X,Testigos[:,sujeto*T+topico], color = "tab:brown" ,linewidth = 2, alpha = 0.3)
            plt.xlabel("Tiempo")
            plt.ylabel(r"$x^i$")
            plt.grid(alpha = 0.5)
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close("Topico")

        
#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la varianza de las opiniones como métrica.

def Mapa_Colores_Varianza_opiniones(DF,path,carpeta,T,
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
    
    for topico in range(T):
    
        # Itero en los valores de mis parámetros alfa y umbral.
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            # Me defino el array en el cual acumulo los datos de las opiniones finales de todas mis simulaciones
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
            # Voy a calcular la varianza de las opiniones de cada simulación y luego promediar
            # esos valores a lo largo de todos los ensambles.
            
            Varianzas = np.zeros(archivos.shape[0])
            
            for simulacion in range(archivos.shape[0]):
                Varianzas[simulacion] = np.var(Opifinales[AGENTES*T*simulacion+topico:AGENTES*T*(simulacion+1)+topico:2])
            
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Varianzas)
            
        #--------------------------------------------------------------------------------
        
        # Una vez que tengo el ZZ completo, armo mi mapa de colores
        direccion_guardado = Path("../../../Imagenes/{}/Varianza Opiniones Topico {}.png".format(carpeta,topico))
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("Varianza Opiniones",figsize=(20,15))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        
        plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
        plt.colorbar()
        plt.title("Varianza de opiniones en Espacio de Parametros")
    
        # Guardo la figura y la cierro
    
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Varianza Opiniones")
    
    """
    # Hago el plotteo de las curvas de Kapppa
    
    if Condicion_curvas_kappa:
        
        Epsilons = np.linspace(2,max(Array_parametro_2),50)
        Alfa = 4
        Kappa_min = np.zeros(Epsilons.shape[0])
        Kappa_max = np.zeros(Epsilons.shape[0])
        
        for indice,epsilon in enumerate(Epsilons):
            
            # Calculo dónde se encuentra el mínimo de mi función Derivada_Kappa
            x_min = epsilon/Alfa
            
            # Calculo los puntos críticos donde voy a encontrar los Kappa máximos y mínimos
            raiz_min = fsolve(Derivada_kappa,x_min-3,args=(Alfa*(1+COSDELTA),epsilon))
            raiz_max = fsolve(Derivada_kappa,x_min+3,args=(Alfa*(1+COSDELTA),epsilon))
            
            # Asigno los valores de los Kappa a mis matrices
            Kappa_min[indice] = Kappa(raiz_max, Alfa*(1+COSDELTA), epsilon)
            Kappa_max[indice] = Kappa(raiz_min, Alfa*(1+COSDELTA), epsilon)
            
        # Ahora que tengo las curvas, las grafico
        
        plt.plot(Epsilons,Kappa_min,"--g",linewidth=8)
        plt.plot(Epsilons[Kappa_max < max(Array_parametro_1)],
                 Kappa_max[Kappa_max < max(Array_parametro_1)],
                 "--r",linewidth=8)
    """


#-----------------------------------------------------------------------------------------------

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

def Grafico_histograma(DF,path,carpeta,
                       nombre_parametro_1="parametro_1",titulo_parametro_2="parametro_2"):
    
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
    
    # Hago el plotteo de las curvas de Kapppa
    """
    if Condicion_curvas_kappa:
        
        Alfas = np.linspace(Array_parametro_2[0],Array_parametro_2[-1],50)
        epsilon = 4
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
        
        plt.plot(Alfas,Kappa_min,"--g",linewidth=8)
        plt.plot(Alfas[Kappa_max < max(Array_parametro_1)],
                 Kappa_max[Kappa_max < max(Array_parametro_1)],
                 "--r",linewidth=8)
    """

    
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

#--------------------------------------------------------------------------------

# Esta función me construye el gráfico de punto fijo en función del parámetro 2.
# Para eso toma los valores de interés final de cada simulación y los promedia, obteniendo
# un punto que indica el punto fijo al cuál tiende el sistema.

def Graf_Punto_fijo_vs_parametro(DF,path,carpeta,T=2,
                                 nombre_parametro_2="parametro2",titulo_parametro_1="parametro 1",
                                 titulo_parametro_2="parametro 2", 
                                 Condicion_punto_inestable_Kappa_Epsilon = False,
                                 Condicion_punto_inestable_Epsilon_Kappa = False):
   
    # Armo mi generador de números aleatorios
#    rng = np.random.default_rng(seed = 50)
    
    AGENTES = int(np.unique(DF["n"]))
    COSDELTA = float(np.unique(DF["cosdelta"]))
    
    # Defino los valores de Parametro_1 que planeo graficar
    Valores_importantes = [0,math.floor(len(np.unique(DF["parametro_1"]))/3),
                           math.floor(2*len(np.unique(DF["parametro_1"]))/3),
                           len(np.unique(DF["parametro_1"]))-1]
    
    Array_parametro_1 = np.unique(DF["parametro_1"])[Valores_importantes]
    Array_parametro_2 = np.unique(DF["parametro_2"])
    
    Tupla_total = [(parametro_1,numero_2,parametro_2) for parametro_1 in Array_parametro_1
                   for numero_2,parametro_2 in enumerate(Array_parametro_2)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Armo arrays vacío para los valores de X e Y
    X = np.array([])
    Y = np.array([])
    
    # Armo la lista de colores y propiedades para graficar mis datos
    default_cycler = (cycler(color=["r","g","b","c"]))
    
    # Abro el gráfico y fijo algunos parámetros
    plt.rcParams.update({'font.size': 32})
    plt.rc("axes",prop_cycle = default_cycler)
    plt.figure("Puntofijo",figsize=(20,15))
    plt.xlabel(r"${}$".format(titulo_parametro_2))
    plt.ylabel("Interés final promedio")
    plt.grid(alpha = 0.5)
    
    
    for PARAMETRO_1,Numero_2,PARAMETRO_2 in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2), "nombre"])

        #-----------------------------------------------------------------------------------------
        
        # Armo unos arrays provisorios para acumular los datos de todas las simulaciones asociadas a un valor del parametro 2
        X_i = np.ones(archivos.shape[0]) * PARAMETRO_2
        Y_i = np.zeros(archivos.shape[0])
        
        for indice_archivo,nombre in enumerate(archivos):
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Matriz de Adyacencia
            # Semilla
            
            Datos = ldata(path / nombre)
            
            Y_i[indice_archivo] = np.mean(np.array(Datos[5][:-1],dtype="float")) # Tomo los intereses finales y les tomo un promedio
            
            #----------------------------------------------------------------------------------------------------------------------------------
        
        # Agrego los datos calculados a los vectores que voy a usar para graficar
        
        X = np.concatenate((X,X_i),axis=None)
        Y = np.concatenate((Y,Y_i),axis=None)
        
        
        # Armo un if que me grafique si recorrí todos los valores en el array de Parametro 2
        if Numero_2 == Array_parametro_2.shape[0]-1:
#            X = X + rng.normal(scale = 0.2, size = X.shape)
            plt.scatter(X,Y, label = r"${} = {}$".format(titulo_parametro_1,PARAMETRO_1), s = 200)
            
            # Reseteo mis vectores
            X = np.array([])
            Y = np.array([])
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    
    # Armo la parte de los puntos inestables
    
    if Condicion_punto_inestable_Kappa_Epsilon:
        
        # Armo los arrays para plotear los puntos inestables
        X_inestable = np.zeros(Array_parametro_2.shape[0])
        Y_inestable = np.zeros(Array_parametro_2.shape[0])
        
        for PARAMETRO_1,Numero_2,PARAMETRO_2 in Tupla_total:
            
            # Calculo el valor del punto fijo inestable
            X_inestable[Numero_2] = PARAMETRO_2
            
            raices = Raices_Ecuacion_Dinamica(PARAMETRO_1, 4, COSDELTA, PARAMETRO_2)
            if(raices != 0).all():
                Y_inestable[Numero_2] = raices[1]
            else:
                Y_inestable[Numero_2] = 0
                
            
            if Numero_2 == Array_parametro_2.shape[0]-1:
                
                plt.plot(X_inestable[Y_inestable != 0],Y_inestable[Y_inestable != 0],"--",linewidth = 6)
                X_inestable = np.zeros(Array_parametro_2.shape[0])
                Y_inestable = np.zeros(Array_parametro_2.shape[0])
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    if Condicion_punto_inestable_Epsilon_Kappa:
        
        # Armo los arrays para plotear los puntos inestables
        X_inestable = np.zeros(Array_parametro_2.shape[0])
        Y_inestable = np.zeros(Array_parametro_2.shape[0])
        
        for PARAMETRO_1,Numero_2,PARAMETRO_2 in Tupla_total:
            
            # Calculo el valor del punto fijo inestable
            X_inestable[Numero_2] = PARAMETRO_2
            
            raices = Raices_Ecuacion_Dinamica(PARAMETRO_2, 4, COSDELTA, PARAMETRO_1)
            if(raices != 0).all():
                Y_inestable[Numero_2] = raices[1]
            else:
                Y_inestable[Numero_2] = 0
                
            
            if Numero_2 == Array_parametro_2.shape[0]-1:
                
                plt.plot(X_inestable[Y_inestable != 0],Y_inestable[Y_inestable != 0],"--",linewidth = 6)
                X_inestable = np.zeros(Array_parametro_2.shape[0])
                Y_inestable = np.zeros(Array_parametro_2.shape[0])
    
    #----------------------------------------------------------------------------------------------------------------------------------
    
    direccion_guardado = Path("../../../Imagenes/{}/Puntofijovs{}_N={:.0f}_Cdelta={:.1f}.png".format(carpeta,nombre_parametro_2,AGENTES,COSDELTA))
    plt.legend()
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close("Puntofijo")
   
    
#--------------------------------------------------------------------------------

# Esta función me construye el gráfico de punto fijo en un plot 3D
# Para eso toma los valores de interés final de cada simulación y los promedia, obteniendo
# un punto que indica el punto fijo al cuál tiende el sistema.


def Graf_Punto_fijo_3D(DF,path,carpeta,T=2,
                       titulo_parametro_1="parametro 1",
                       titulo_parametro_2="parametro 2",
                       titulo_parametro_3="parametro_3"):
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los valores de Parametro_1 que planeo graficar
    Valores_importantes = [0,math.floor(len(np.unique(DF["parametro_1"]))/3),
                            math.floor(2*len(np.unique(DF["parametro_1"]))/3),
                            len(np.unique(DF["parametro_1"]))-1]
    
    Array_parametro_1 = np.unique(DF["parametro_1"])[Valores_importantes]
    Array_parametro_2 = np.unique(DF["parametro_2"])
    Array_parametro_3 = np.unique(DF["parametro_3"])
    
    Tupla_total = [(parametro_1,numero_2,parametro_2,numero_3,parametro_3) for parametro_1 in Array_parametro_1
                   for numero_2,parametro_2 in enumerate(Array_parametro_2)
                   for numero_3,parametro_3 in enumerate(Array_parametro_3)]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    
    # Armo arrays vacío para los valores de X, Y y Z
    X = np.array([])
    Y = np.array([])
    Z = np.array([])
    
    # Armo la lista de colores y propiedades para graficar mis datos
    default_cycler = (cycler(color=["r","g","b","c"])*cycler(marker = "o"))
    
    # Abro el gráfico y fijo algunos parámetros
    plt.rcParams.update({'font.size': 32})
    plt.rc("axes",prop_cycle = default_cycler)
    fig = plt.figure("Puntofijo",figsize=(60,45))
    ax = fig.add_subplot(projection = "3d")
    ax.set_xlabel(r"${}$".format(titulo_parametro_2),labelpad = 30)
    ax.set_ylabel(r"${}$".format(titulo_parametro_3),labelpad = 30)
    ax.set_zlabel("Interés final promedio", labelpad = 30)
    # ax.grid(alpha = 0.5)
    
    
    for PARAMETRO_1,Numero_2,PARAMETRO_2,Numero_3,PARAMETRO_3 in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["parametro_1"]==PARAMETRO_1) & 
                                    (DF["parametro_2"]==PARAMETRO_2) &
                                    (DF["parametro_3"]==PARAMETRO_3), "nombre"])

        #-----------------------------------------------------------------------------------------
        
        # Armo unos arrays provisorios para acumular los datos de todas las simulaciones asociadas a un valor del parametro 2
        X_i = np.ones(archivos.shape[0]) * PARAMETRO_2
        Y_i = np.ones(archivos.shape[0]) * PARAMETRO_3
        Z_i = np.zeros(archivos.shape[0])
        
        for indice_archivo,nombre in enumerate(archivos):
            
            # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
            # Opinión Inicial del sistema
            # Variación Promedio
            # Opinión Final
            # Matriz de Adyacencia
            # Semilla
            
            Datos = ldata(path / nombre)
            
            Z_i[indice_archivo] = np.mean(np.array(Datos[5][:-1],dtype="float")) # Tomo los intereses finales y les tomo un promedio
            
            #----------------------------------------------------------------------------------------------------------------------------------
        
        # Agrego los datos calculados a los vectores que voy a usar para graficar
        
        X = np.concatenate((X,X_i),axis=None)
        Y = np.concatenate((Y,Y_i),axis=None)
        Z = np.concatenate((Z,Z_i),axis=None)
        
        # Armo un if que me grafique si recorrí todos los valores en el array de Parametro 2 y 3
        if Numero_2 == Array_parametro_2.shape[0]-1 and Numero_3 == Array_parametro_3.shape[0]-1:
#            X = X + rng.normal(scale = 0.2, size = X.shape)
            ax.scatter(X,Y,Z,label = r"${} = {}$".format(titulo_parametro_1,PARAMETRO_1), s = 400)
            X = np.array([])
            Y = np.array([])
            Z = np.array([])
    
    
    plt.legend()
    direccion_guardado = Path("../../../Imagenes/{}/Puntofijo3D_angulo.png".format(carpeta))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    ax.view_init(0,0,0)
    direccion_guardado = Path("../../../Imagenes/{}/Puntofijo3D_frente.png".format(carpeta))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    ax.view_init(0,90,0)
    direccion_guardado = Path("../../../Imagenes/{}/Puntofijo3D_perfil.png".format(carpeta))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close("Puntofijo")

#-------------------------------------------------------------------------------------------

# Esto me halla las soluciones de la ecuación dinámica y me devuelve un array con
# esas soluciones. Si tengo tres soluciones, me devuelve las tres en orden.
# Si tengo una sola, me devuelve un array de tres elementos, pero con dos ceros,
# si todo funciona bien.

def Raices_Ecuacion_Dinamica(Kappa,Alfa,Cdelta,Epsilon):
    
    x0 = 0 # Condición incial que uso para hallar las soluciones

    raices = np.zeros(3) # Array en el que guardo las raíces.
    indice = 0 # Es la posición del vector en la cuál guardo la raíz encontrada

    while x0 < Kappa:
        
        # Calculo un valor que anula mi ecuación dinámica
        resultado = fsolve(Ecuacion_dinamica,x0,args=(Kappa,Alfa,Cdelta,Epsilon))[0]
        
        # Reviso si el valor hallado es una raíz o un resultado de que el solver se haya estancado
        Condicion_raiz = np.isclose(Ecuacion_dinamica(resultado,Kappa,Alfa,Cdelta,Epsilon),0,atol=1e-06)
        
        if not(np.isclose(raices,np.ones(3)*resultado).any()) and Condicion_raiz:
            
            # Fijo mi nueva raíz en el vector de raices
            raices[indice] = resultado
            indice += 1
        
        x0 += 0.1
        
    return raices


#-----------------------------------------------------------------------------------------------

# Esta función es la que arma los gráficos de los mapas de colores en el espacio de
# parámetros de alfa y umbral usando la varianza de las opiniones como métrica.

def Mapa_Colores_Tiempo_convergencia(DF,path,carpeta,
                                    SIM_param_x,SIM_param_y,
                                    ID_param_extra_1,
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
    
    """
    # Hago el plotteo de las curvas de Kapppa
    
    if Condicion_curvas_kappa:
        
        Alfas = np.linspace(Array_parametro_2[0],Array_parametro_2[-1],50)
        epsilon = 4
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
        
        plt.plot(Alfas,Kappa_min,"--g",linewidth=8)
        plt.plot(Alfas[Kappa_max < max(Array_parametro_1)],
                 Kappa_max[Kappa_max < max(Array_parametro_1)],
                 "--r",linewidth=8)
    """
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Tiempo_Convergencia")


#-----------------------------------------------------------------------------------------------

# Esta función me construye el gráfico de opinión en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_Derivada_vs_tiempo(DF,path,carpeta,T=2,
                       nombre_parametro_1="parametro1",nombre_parametro_2="parametro2"):
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir valores de cada array.
    
    Ns = np.unique(DF["n"])
    
    # Defino los valores de Parametro_1 que planeo graficar
    Valores_importantes_1 = [0,math.floor(len(np.unique(DF["parametro_1"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_1"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_1"]))/4),
                            len(np.unique(DF["parametro_1"]))-1]
    
    # Defino los valores de Parametro_2 que planeo graficar
    Valores_importantes_2 = [0,math.floor(len(np.unique(DF["parametro_2"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_2"]))/4),
                            math.floor(3*len(np.unique(DF["parametro_2"]))/4),
                            len(np.unique(DF["parametro_2"]))-1]
    
    # Armo los arrays de mis parámetros y después armo la Tupla_Total
    Array_parametro_1 = np.unique(DF["parametro_1"])[Valores_importantes_1]
    Array_parametro_2 = np.unique(DF["parametro_2"])[Valores_importantes_2]
    
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
            direccion_guardado = Path("../../../Imagenes/{}/DerivadavsT_N={:.0f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,nombre_parametro_1,PARAMETRO_1,nombre_parametro_2,PARAMETRO_2,repeticion))
            
            # Armo mi gráfico, lo guardo y lo cierro
            
            dt = 0.01 # Paso temporal
            
            plt.rcParams.update({'font.size': 32})
            plt.figure("Topico",figsize=(20,15))
            X = np.arange(Testigos.shape[0])*dt
            for sujeto in range(int(Testigos.shape[1]/T)):
                for topico in range(T):
                    Derivada = (Testigos[1:,sujeto*T+topico] - Testigos[0:-1,sujeto*T+topico])/dt
                    plt.plot(X[0:-1],Derivada, color = "firebrick" ,linewidth = 1.5, alpha = 0.4)
            plt.xlabel("Tiempo")
            plt.ylabel("Derivada Interes")
            plt.grid(alpha = 0.5)
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close("Topico")


#-----------------------------------------------------------------------------------------------

# Esta función me construye el gráfico trayectorias en el espacio de fases

def Graf_trayectorias_opiniones(DF,path,carpeta,
                       ID_param_x,ID_param_y,
                       ID_param_extra_1):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    KAPPAS = int(np.unique(DF["Kappas"]))
    Arr_param_x = np.unique(DF["parametro_x"])
    Arr_param_y = np.unique(DF["parametro_y"])
    
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(param_x,param_y) for param_x in Arr_param_x
                   for param_y in Arr_param_y]
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Testigos"
    
    # Sólo tiene sentido graficar en dos dimensiones, en una es el 
    # Gráfico de Opi vs T y en tres no se vería mejor.
    T=2
    
    
    for PARAM_X,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[(DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["Kappas"]==KAPPAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])
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
            direccion_guardado = Path("../../../Imagenes/{}/Trayectorias_opiniones_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                                                                                                                             ID_param_y,PARAM_Y,ID_param_extra_1,KAPPAS,repeticion))
            
            # Armo mi gráfico, lo guardo y lo cierro
            
            plt.rcParams.update({'font.size': 32})
            plt.figure("Trayectorias",figsize=(20,15))
            # X = np.arange(Testigos.shape[0])*0.01
            for sujeto in range(int(Testigos.shape[1]/T)):
                plt.plot(Testigos[:,sujeto*T],Testigos[:,sujeto*T+1], color = "tab:gray" ,linewidth = 3, alpha = 0.3)
            plt.scatter(Testigos[-1,0::2],Testigos[-1,1::2], s=30,label="Opinión Final")
            plt.xlabel(r"$x_i^1$")
            plt.ylabel(r"$x_i^2$")
            # plt.grid(alpha = 0.5)
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close("Trayectorias")


#-----------------------------------------------------------------------------------------------

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
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                if repeticion < 3 :
                    direccion_guardado = Path("../../../Imagenes/{}/Histograma_opiniones_2D_N={:.0f}_{}={:.2f}_{}={:.2f}_{}={:.2f}_sim={}.png".format(carpeta,AGENTES,ID_param_x,PARAM_X,
                                                                                                                                                     ID_param_y,PARAM_Y,ID_param_extra_1,EXTRAS,repeticion))
                    
                    # Armo mi gráfico, lo guardo y lo cierro
                    
                    plt.rcParams.update({'font.size': 32})
                    plt.figure(figsize=(20,15))
                    _, _, _, im = plt.hist2d(Opifinales[0::T], Opifinales[1::T], bins=bins,
                                             range=[[-EXTRAS,EXTRAS],[-EXTRAS,EXTRAS]],density=True,
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

def Mapa_Colores_Antidiagonales_Covarianza(DF,path,carpeta,
                       SIM_param_x,SIM_param_y,
                       ID_param_extra_1):
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    KAPPAS = int(np.unique(DF["Kappas"]))
    # COSD = int(np.unique(DF["cosdelta"]))
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
                                    (DF["kappas"]==KAPPAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        Antidiagonales = np.zeros(archivos.shape[0])
        
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
            
            # Normalizo mis datos usando el valor de Kappa
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/KAPPAS
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            M_cov = np.cov(Opifinales)
            Antidiagonales[indice] = (M_cov[0,1]+M_cov[1,0])/2
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Antidiagonales)
            
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Antidiagonales_Covarianza_{}={}.png".format(carpeta,ID_param_extra_1,KAPPAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Antidiagonales_Covarianza",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Antidiagonales Matriz Coviaranza en Espacio de Parametros")
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Antidiagonales_Covarianza")
    

#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Mapa_Colores_Determinante_Covarianza(DF,path,carpeta,
                       SIM_param_x,SIM_param_y,
                       ID_param_extra_1):
    
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los arrays de parámetros diferentes
    KAPPAS = int(np.unique(DF["Kappas"]))
    # COSD = int(np.unique(DF["cosdelta"]))
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
                                    (DF["kappas"]==KAPPAS) & 
                                    (DF["parametro_x"]==PARAM_X) &
                                    (DF["parametro_y"]==PARAM_Y), "nombre"])
        #-----------------------------------------------------------------------------------------
        
        Determinantes = np.zeros(archivos.shape[0])
        
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
            
            # Normalizo mis datos usando el valor de Kappa
            for topico in range(T):
                Opifinales[topico,:] = np.array(Datos[5][topico:-1:T], dtype="float")/KAPPAS
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            M_cov = np.cov(Opifinales)
            Determinantes[indice] = np.linalg.det(M_cov)
            
        #------------------------------------------------------------------------------------------
        # Con el vector covarianzas calculo el promedio de los trazas de las covarianzas
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = np.mean(Determinantes)
            
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Determinantes_Covarianza_{}={}.png".format(carpeta,ID_param_extra_1,KAPPAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Determinantes_Covarianza",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = "plasma")
    plt.colorbar()
    plt.title("Determinantes Matriz Covarianza en Espacio de Parametros")
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Determinantes_Covarianza")


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

def Calculo_Antidiagonales_Covarianza(DF,path):
    
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
            
            antidiagonales = np.zeros(archivos.shape[0])
            
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
                antidiagonales[repeticion] = (M_cov[0,1]+M_cov[1,0])/2
        
            if PARAM_X not in Salida[KAPPAS].keys():
                Salida[KAPPAS][PARAM_X] = dict()
            Salida[KAPPAS][PARAM_X][PARAM_Y] = antidiagonales
    
    return Salida


#-----------------------------------------------------------------------------------------------

# Esta función calcula la traza de la matriz de Covarianza de las distribuciones
# de opiniones respecto a los T tópicos

def Calculo_Determinante_Covarianza(DF,path):
    
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
            
            determinantes = np.zeros(archivos.shape[0])
            
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
                determinantes[repeticion] = np.linalg.det(M_cov)
        
            if PARAM_X not in Salida[KAPPAS].keys():
                Salida[KAPPAS][PARAM_X] = dict()
            Salida[KAPPAS][PARAM_X][PARAM_Y] = determinantes
    
    return Salida

#----------------------------------------------------------------------------------------------

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
                    Opifinales[topico,:] = Opifinales[topico,:]/ PARAM_X
                
                # Esta función normaliza las Opiniones Finales usando la 
                # variable EXTRA, porque asume que EXTRA es el Kappa. De no serlo,
                # corregir a que EXTRAS sea PARAM_X o algo así
                
                # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
                Identidad[indice] = repeticion
        
                M_cov = np.cov(Opifinales)
                Varianza_X[indice] = M_cov[0,0]
                Varianza_Y[indice] = M_cov[1,1]
                Covarianza[indice] = M_cov[0,1]
                Promedios[indice] = np.linalg.norm(np.array(Datos[5][:-1:], dtype="float"),ord=1) / np.array(Datos[5][:-1:], dtype="float").shape[0]
                
                # Tengo que rearmar Opifinales para que sea un sólo vector con todo
                
                Opifinales = np.array(Datos[5][:-1], dtype="float")
                Opifinales = Opifinales/PARAM_X
                
                # Armo mi array de Distribucion, que tiene la proba de que una opinión
                # pertenezca a una región del espacio de tópicos
                Probas = Clasificacion(Opifinales,N,T)
                
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
            Salida[EXTRAS][PARAM_X][PARAM_Y]["Entropia"] = Entropia[Ubicacion] / np.log2(N*N)
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
        
        if ent <= 0.35:
            
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
                if np.abs(cov) > 0.85:
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
            if sx >= 0.5 and sy < 0.5:
                # Dos extremos horizontal
                Resultados[i] = 6
            elif sx < 0.5 and sy >= 0.5:
                # Dos extremos vertical
                Resultados[i] = 6
            
            else:
                # Polarización
                # Polarización ideológica
                if np.abs(cov) >= 0.5:
                    Resultados[i] = 7
                
                # Transición con anchura
                elif np.abs(cov) >= 0.2 and np.abs(cov) < 0.5:
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
    Dic_Total = Diccionario_metricas(DF,path,20)
    
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
        Distribucion[fila*Ny+columna] += 1
    
    # Una vez armada mi distribucion, la normalizo.
    Distribucion = Distribucion/np.sum(Distribucion)
    
    # Returneo la distribucion
    return Distribucion

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
                Probas = Clasificacion(Opifinales,N,T)
                
                # Con esa distribución puedo directamente calcular la entropía.
                entropias[repeticion] = np.matmul(Probas[Probas != 0], np.log2(Probas[Probas != 0]))*(-1)
        
            if PARAM_X not in Salida[EXTRAS].keys():
                Salida[EXTRAS][PARAM_X] = dict()
            Salida[EXTRAS][PARAM_X][PARAM_Y] = entropias/np.log2(N*N)
    
    return Salida

#---------------------------------------------------------------------------------------------------------------------------------------------------------------

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
    Fraccion_polarizados = np.zeros((10,Arr_param_y.shape[0]))
    
    #--------------------------------------------------------------------------------
    
    # Diccionario con la entropía, Sigma_x y Sigma_y de todas las simulaciones
    # para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF,path,20)
    
    for indice,PARAM_Y in Tupla_total:
                
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                             Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"])
        
        for estado in range(10):
            Fraccion_polarizados[estado,indice] = np.count_nonzero(Frecuencias == estado)/Frecuencias.shape[0]

    for estado in range(10):
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
    Arr_param_x = np.array([0.2,0.5,0.8])
    Arr_param_y = np.array([0.3,0.7,0.9])
    
    # Defino la cantidad de filas y columnas que voy a graficar
    Filas = 10
    Columnas = T
    
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
            plt.figure(figsize=(40,21))
            plots = [[plt.subplot(Filas, Columnas, i*Columnas + j + 1) for j in range(Columnas)] for i in range(Filas)]
            
            for nombre in archivos:

                repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])                
                if repeticion < Filas:
                
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
                    
                    for topico in range(T):
                        hist, bines = np.histogram(Opifinales[0::T], bins=np.linspace(-1, 1, 21), density=True)
                        X = (bines[1:]+bines[0:-1])/2
                        plots[repeticion][topico].plot(X,hist,linewidth=8,color='tab:blue')
                        plots[repeticion][topico].set_xlim(-1, 1)  # Set x-axis limits
            
            # Le pongo nombres a los ejes más externos
            for i, row in enumerate(plots):
                for j, subplot in enumerate(row):
                    if j == 0:  # First column, set y label
                        subplot.set_ylabel('Densidad')
                    if i == Filas - 1:  # Last row, set x label
                        subplot.set_xlabel("Opiniones")# r"$x_i$")
                        
            # Set titles for each column
            column_titles = ['Histogramas tópico 0', 'Histogramas Tópico 1']
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
            fig, axs = plt.subplots(1, 3, figsize=(60, 21))
            
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
            hist, bines = np.histogram(OpiTotales[0::T], bins=np.linspace(-10, 10, 21), density=True)
            X = (bines[1:]+bines[0:-1])/2
            axs[1].plot(X,hist,linewidth=8,color='tab:blue')
            axs[1].set_xlim(-10, 10)  # Set x-axis limits
            axs[1].set_xlabel("Opiniones")
            axs[1].set_ylabel("Fracción")
            axs[1].set_title('Tópico 0')
            
            # Armo el gráfico del histograma del tópico 1
            hist, bines = np.histogram(OpiTotales[1::T], bins=np.linspace(-10, 10, 21), density=True)
            X = (bines[1:]+bines[0:-1])/2
            axs[2].plot(X,hist,linewidth=8,color='tab:blue')
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
                       'V201362x':'Allowing Felons to vote', 'V201372x':'Pres didnt worry Congress',
                       'V201375x':'Restricting Journalist access', 'V201382x':'Corruption increased or decreased since Trump',
                       'V201386x':'Impeachment', 'V201405x':'Require employers to offer paid leave to parents',
                       'V201408x':'Service same sex couples', 'V201411x':'Transgender Policy', 'V201420x':'Birthright Citizenship',
                       'V201423x':'Should children brought illegally be sent back','V201426x':'Wall with Mexico',
                       'V201429':'Best way to deal with Urban Unrest','V201605x':'Political Violence compared to 4 years ago',
                       'V202236x':'Allowing refugees to come to US','V202239x':'Effect of Illegal inmigration on crime rate',
                       'V202242x':'Providing path to citizenship','V202245x':'Returning unauthorized immigrants to native country',
                       'V202248x':'Separating children from detained immigrants','V202255x':'Less or more Government',
                       'V202256':'Good for society to have more government regulation',
                       'V202259x':'Government trying to reduce income inequality','V202276x':'People in rural areas get more/less from Govt.',
                       'V202279x':'People in rural areas have too much/too little influence','V202282x':'People in rural areas get too much/too little respect',
                       'V202286x':'Easier/Harder for working mother to bond with child','V202290x':'Better/Worse if man works and woman takes care of home',
                       'V202320x':'Economic Mobility compared to 20 years ago','V202328x':'Obamacare','V202331x':'Vaccines Schools',
                       'V202336x':'Regulation on Greenhouse Emissions','V202341x':'Background checks',
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

# Esta función construye la matriz de DJS y la returnea
    
def Matriz_DJS(DF_datos,DF_Anes,Dic_ANES,path):
    
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
    ZZ = np.zeros((XX.shape[0],XX.shape[1],100))
    
    #--------------------------------------------------------------------------------
    
    # Extraigo mis distribuciones sacando la cruz, que sería sacar a los agentes que
    # respondieron neutro en alguna encuesta.
    
    # Separo las opiniones de 0
    df_aux = DF_Anes.loc[(DF_Anes[Dic_ANES["code_1"]]>0) & (DF_Anes[Dic_ANES["code_2"]]>0)]
    # Reviso la cantidad de respuestas de cada pregunta
    resp_1 = np.unique(df_aux[Dic_ANES["code_1"]]).shape[0]
    resp_2 = np.unique(df_aux[Dic_ANES["code_2"]]).shape[0]
    
    # Los clasifico como código x y código y
    if resp_1 >= resp_2:
        code_x = Dic_ANES["code_1"]
        code_y = Dic_ANES["code_2"]
    else:
        code_x = Dic_ANES["code_2"]
        code_y = Dic_ANES["code_1"]
    
    
    # Dos preguntas con siete respuestas
    if resp_1 == 7 and resp_2 == 7:
        
        # Saco la cruz
        df_filtered = df_aux[(df_aux[code_x] != 4) & (df_aux[code_y] != 4)] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[Dic_ANES["weights"]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:][:,np.arange(7) != 3]
        Distr_Enc = Distr_Enc.flatten()
    
    # Una pregunta con siete respuestas y otra con seis
    elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
        
        # Saco la cruz
        df_filtered = df_aux[df_aux[code_x] != 4] 
        
        hist2d, xedges, yedges, im = plt.hist2d(x=df_filtered[code_x], y=df_filtered[code_y], weights=df_filtered[Dic_ANES["weights"]], vmin=0, cmap = "inferno", density = True,
                  bins=[np.arange(df_filtered[code_x].min()-0.5, df_filtered[code_x].max()+1.5, 1), np.arange(df_filtered[code_y].min()-0.5, df_filtered[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d[np.arange(7) != 3,:]
        Distr_Enc = Distr_Enc.flatten()
    
    # Dos preguntas con seis respuestas
    elif resp_1 == 6 and resp_2 == 6:
        
        # No hay necesidad de sacar la cruz
        hist2d, xedges, yedges, im = plt.hist2d(x=df_aux[code_x], y=df_aux[code_y], weights=df_aux[Dic_ANES["weights"]], vmin=0,cmap = "inferno", density = True,
                  bins=[np.arange(df_aux[code_x].min()-0.5, df_aux[code_x].max()+1.5, 1), np.arange(df_aux[code_y].min()-0.5, df_aux[code_y].max()+1.5, 1)])
        plt.close()
        
        Distr_Enc = hist2d.flatten()
    
    #--------------------------------------------------------------------------------
    
    for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        #-----------------------------------------------------------------------------------------
        
        Dist_previa = np.zeros(4)
        
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
            Distr_Sim = Clasificacion(Opifinales,hist2d.shape[0],hist2d.shape[1],T)
            
            # La Distr_Orig es un array plano. Lo que estaria bueno es armarlo de 6x6,
            # total la Distr_Enc tiene tamaño 6x6 siempre
            
            Distr_Sim = np.reshape(Distr_Sim, hist2d.shape)
            
            # Dos preguntas con siete respuestas
            if resp_1 == 7 and resp_2 == 7:
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:][:,np.arange(7) != 3]
            
            # Una pregunta con siete respuestas y otra con seis
            elif (resp_1 == 6 and resp_2 == 7) or (resp_1 == 7 and resp_2 == 6):
                Distr_Sim = Distr_Sim[np.arange(7) != 3,:]
            
            # Dos preguntas con seis respuestas (En este caso no necesita hacer nada, ya es de 6x6 la distribución)
            
            
            # Como removí parte de mi distribución, posiblemente ya no esté normalizada
            # la distribución. Así que debería ahora sumar agentes de a 1 hasta asegurarme
            # de que otra vez esté normalizada
            Distr_Sim = Distr_Sim.flatten()
            if np.sum(Distr_Sim) != 1:
                agentes_agregar = int((1-np.sum(Distr_Sim))/frac_agente_ind)
                for i in range(agentes_agregar):
                    Distr_Sim[np.argmin(Distr_Sim)] += frac_agente_ind
            # Luego de volver a normalizar mi distribución, si quedaron lugares
            # sin agentes, los relleno
            restar = np.count_nonzero(Distr_Sim == 0)
            Distr_Sim[Distr_Sim == 0] = np.ones(restar)*frac_agente_ind
            Distr_Sim[np.argmax(Distr_Sim)] -= frac_agente_ind*restar
            
            #-----------------------------------------------------------------------------------------
            
            # Calculo la distancia Jensen-Shannon de la distribución de la encuesta con la distribución
            # de la simulación. Hago esto cuatro veces porque tengo que rotar cuatro veces la distribución
            # simulada, cosa de asegurarme de que no importe la dirección del estado final.
            
            for rotacion in range(4):
                
                Dist_previa[rotacion] = jensenshannon(Distr_Enc,Distr_Sim)
                
                # Una vez que hice el cálculo de la distancia y todo, roto la matriz
                Distr_Sim = np.reshape(Distr_Sim, (6,6))
                Distr_Sim = Rotar_matriz(Distr_Sim)
                Distr_Sim = Distr_Sim.flatten()
                
            #-----------------------------------------------------------------------------------------
            
            # Una vez que calcule las 4 distancias habiendo rotado 4 veces la distribución,
            # lo que me queda es guardar eso en las matrices ZZ correspondientes.
            
            repeticion = int(DF_datos.loc[DF_datos["nombre"]==nombre,"iteracion"])
            
            ZZ[(Arr_param_y.shape[0]-1)-fila,columna,repeticion] = np.min(Dist_previa)
            
            #-----------------------------------------------------------------------------------------
            
    # Resuelta la matriz, returneo mi info
    
    return ZZ, code_x, code_y

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

#-----------------------------------------------------------------------------------------------

# Armo dos mapas de colores de DJS. El mapa de colores
# no tiene los agentes que hayan opinado neutro en ninguna de las preguntas.
# En ambos casos estoy considerando que ambas preguntas tienen 7 respuestas. Voy a tener que ir
# viendo cómo resolver si tienen 6 respuestas.

def Mapas_Colores_DJS(Dist_JS, code_x, code_y, DF_datos, Dic_ANES, dict_labels, carpeta,
                      ID_param_x,SIM_param_x,ID_param_y,SIM_param_y):
    
    # Defino los arrays de parámetros diferentes
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    #--------------------------------------------------------------------------------
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    # Organizo las matrices Dist_JS según su similaridad
    Dist_JS = np.sort(Dist_JS)
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin cruz
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}.png".format(carpeta,code_y,code_x))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,np.mean(Dist_JS, axis=2),shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.title("Distancia Jensen-Shannon sin cruz\n {} vs {}".format(dict_labels[code_y],dict_labels[code_x]))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon")
    
    # Y ahora me armo el gráfico de promedios de distancia JS según cantidad de simulaciones
    # consideradas, con las simulaciones ordenadas de las de menos distancia a las de más distancia
    
    for i in range(10):
        
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Distancia Jensen-Shannon",figsize=(28,21))
        plt.xlabel(r"${}$".format(SIM_param_x))
        plt.ylabel(r"${}$".format(SIM_param_y))
        
        # Hago el ploteo del mapa de colores con el colormesh
        Dist_JS_prom = np.mean(Dist_JS[:,:,0:10+i*10],axis=2)
        
        # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
        tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
        
        plt.pcolormesh(XX,YY,Dist_JS_prom,shading="nearest", cmap = "cividis")
        plt.colorbar()
        plt.scatter(XX[tupla],YY[tupla], marker="X", color = "red", s = 180)
        
        # if XX[tupla]+0.05 < np.max(XX):
        #     xtext = XX[tupla]
        # else:
        #     xtext = XX[tupla]-0.05


        # if YY[tupla]-0.05 < np.min(YY):
        #     ytext = YY[tupla]+0.05
        # else:
        #     ytext = YY[tupla]-0.05
        
        # plt.text(xtext,ytext, r"${}$ = {:.2f},${}$ = {:.2f} ".format(SIM_param_y, YY[tupla], SIM_param_x, XX[tupla]), fontsize= 36)
        plt.title("Distancia Jensen-Shannon sin cruz {} simulaciones\n {} vs {}".format(10+i*10,dict_labels[code_y],dict_labels[code_x]))
        
        # Guardo la figura y la cierro
        
        plt.savefig(direccion_guardado , bbox_inches = "tight")
        plt.close("Ranking Distancia Jensen-Shannon")
        
        
        
    

#-----------------------------------------------------------------------------------------------

# Esta función arma los histogramas de opiniones máxima y mínima similaridad entre las 10 simulaciones
# más similares con la distribución de la encuesta

def Hist2D_similares_FEF(Dist_JS, code_x, code_y, DF_datos, Dic_ANES, dict_labels, carpeta, path, bins,
                         ID_param_x,SIM_param_x,ID_param_y,SIM_param_y):
    
    # Hago los gráficos de histograma 2D de las simulaciones que más se parecen y que menos se parecen
    # a mis distribuciones de las encuestas
    Dist_JS_sorted = np.sort(Dist_JS)
    
    # Diccionario con la entropía, Sigma_x, Sigma_y, Promedios y Covarianzas
    # de todas las simulaciones para cada punto del espacio de parámetros.
    Dic_Total = Diccionario_metricas(DF_datos,path, 20, 20)
    
    #-------------------------------------------------------------------------------------------------
    
    # Antes de ordenar mis matrices, voy a obtener la info del gráfico que más se parece
    # y del décimo que más se parece se parece a lo que estoy queriendo comparar.
    
    iMin = np.unravel_index(np.argmin(Dist_JS),Dist_JS.shape)
    
    # Hallo el décimo que más se parece a la distribución. Arranco con el que no tiene centro
    
    flattened_array = Dist_JS.flatten()
    sorted_indices = np.argsort(flattened_array)
    tenth_element_flat_index = sorted_indices[9]
    iMax = np.unravel_index(tenth_element_flat_index, Dist_JS.shape)
    
    #--------------------------------------------------------------------------------

    # Defino la cantidad de agentes de la red
    AGENTES = int(np.unique(DF_datos["n"]))
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Defino el tipo de archivo del cuál tomaré los datos
    TIPO = "Opiniones"
    T = 2
    
    # Armo una lista de tuplas que tengan organizados los parámetros a utilizar
    Tupla_total = [(i,param_x,j,param_y) for i,param_x in enumerate(Arr_param_x)
                   for j,param_y in enumerate(Arr_param_y)]
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    #--------------------------------------------------------------------------------
    
    # Armo listas de strings y números para mis archivos
    Lista_similaridad = ["min_distancia","max_distancia"]
    Valor_distancia = [np.min(Dist_JS_sorted),np.max(Dist_JS_sorted[:,:,0:10])]
    
    Nombres = ["Consenso neutral", "Consenso radicalizado", "Polarización 1D y Consenso",
           "Polarización Ideológica", "Transición", "Polarización Descorrelacionada",
           "Polarización 1D y Consenso con anchura",
           "Polarización Ideológica con anchura", "Transición con anchura",
           "Polarización Descorrelacionada con anchura"]
    
    for tupla,simil,distan in zip([iMin, iMax],Lista_similaridad,Valor_distancia):
    
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
                
                direccion_guardado = Path("../../../Imagenes/{}/Hist_2D_{}_{}vs{}.png".format(carpeta /"Sin Cruz",simil,code_y,code_x))
                
                indice = np.where(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Identidad"] == repeticion)[0][0]
                estado = int(Frecuencias[indice])
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Tengo que armar los valores de X e Y que voy a graficar
                
                X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
                
                
                #----------------------------------------------------------------------------------------------------------------------------------
                
                # Armo mi gráfico, lo guardo y lo cierro
                
                plt.rcParams.update({'font.size': 44})
                plt.figure(figsize=(28,21))
                _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
                plt.xlabel(r"$x_i^1$")
                plt.ylabel(r"$x_i^2$")
                # Set x-ticks and y-ticks from -10 to 10 using plt.xticks() and plt.yticks()
                # plt.xticks(np.arange(-10, 11, 1))
                # plt.yticks(np.arange(-10, 11, 1))
                plt.title('Distancia JS = {:.2f}, {}={:.2f}, {}={:.2f} \n {} \n {} vs {}'.format(distan,ID_param_x,PARAM_X,ID_param_y,PARAM_Y,Nombres[estado],dict_labels[code_y],dict_labels[code_x]))
                plt.colorbar(im, label='Fracción')
#                cbar.set_clim(Vmin, Vmax)
                plt.savefig(direccion_guardado ,bbox_inches = "tight")
                plt.close()
             
    
    #--------------------------------------------------------------------------------
    
    # Lo que quiero hacer acá es armar gráficos de promedios de opiniones rankeados.
    
    for i in range(10):
        
        # Hago el ploteo del mapa de colores con el colormesh
        Dist_JS_prom = np.mean(Dist_JS_sorted[:,:,0:10+i*10],axis=2)
        # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
        tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
        
        PARAM_X = XX[tupla[0],tupla[1]]
        PARAM_Y = YY[tupla[0],tupla[1]]
        
    #-----------------------------------------------------------------------------------------
    
        OpiTotales = np.empty(0)
        cant_simulaciones = 10+i*10
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF_datos.loc[(DF_datos["tipo"]==TIPO) & 
                                    (DF_datos["n"]==AGENTES) & 
                                    (DF_datos["Extra"]==EXTRAS) & 
                                    (DF_datos["parametro_x"]==PARAM_X) &
                                    (DF_datos["parametro_y"]==PARAM_Y), "nombre"])
        
        i_Proms = np.argsort(Dist_JS[tupla])[0:cant_simulaciones]
        
        for nombre in archivos[i_Proms]:
        
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
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            OpiTotales = np.concatenate((OpiTotales,Opifinales),axis=0)
            
        X_0 = OpiTotales[0::T]
        Y_0 = OpiTotales[1::T]
        
        # Tengo que armar los valores de X e Y que voy a graficar
        
        X = X_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
        Y = Y_0[((X_0>bins[4]) | (X_0<bins[3])) & ((Y_0>bins[4]) | (Y_0<bins[3]))]
        
        direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Hists_prom_{}vs{}_r{}.png".format(carpeta,code_y,code_x,i))
        
        plt.rcParams.update({'font.size': 44})
        plt.figure("Ranking Opiniones Promedio",figsize=(28,21))
        _, _, _, im = plt.hist2d(X, Y, bins=bins,density=True,cmap="inferno")
        plt.xlabel(r"$x_i^1$")
        plt.ylabel(r"$x_i^2$")
        # Set x-ticks and y-ticks from -10 to 10 using plt.xticks() and plt.yticks()
        # plt.xticks(np.arange(-10, 11, 1))
        # plt.yticks(np.arange(-10, 11, 1))
        plt.title(r'Promedio de Histogramas, {} simulaciones, ${}$={}, ${}$={} \n {} vs {}'.format(cant_simulaciones,SIM_param_x,PARAM_X,SIM_param_y,PARAM_Y,dict_labels[code_y],dict_labels[code_x]))
        plt.colorbar(im, label='Fracción')
#                cbar.set_clim(Vmin, Vmax)
        plt.savefig(direccion_guardado ,bbox_inches = "tight")
        plt.close("Ranking Opiniones Promedio")
        
    
    #-------------------------------------------------------------------------------------------------
    
    
    # Lo que quiero hacer acá es armar gráficos de frecuencia de estados de los dos
    # estados más probables en el espacio de parámetros.
    
    # Arranco con un barrido en el ranking
    for i in range(10):
        
        cant_simulaciones = 10+i*10
        # Me construyo la matriz en la que voy a guardar los estados clasificados
        Estados_clasificados = np.zeros((XX.shape[0],XX.shape[1],cant_simulaciones))
        
        # Luego hago un barrido en los parámetros
        
        for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
            
            Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                         Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
            
            #-----------------------------------------------------------------------------------------
            
            # Reviso los vectores de las distancias Jensen Shannon ya calculados y obtengo las posiciones de
            # las simulaciones cuyas distancias sean menores.
            iElemR = np.argsort(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,:])[cant_simulaciones-1]
#            if PARAM_X == 0.1:
#                print("Cos(delta): ",PARAM_X)
#                print("Beta: ",PARAM_Y)
#                print("indice: ",iElemR)
#                print("Cant True: ",np.count_nonzero(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]))
#                print("Tamaño vector Booleano: ",len(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]))
#                print("Cant Frecuencias: ",len(Frecuencias[Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]]))
#                print(Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna])
            Array_bool = Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna, :] <= Dist_JS[(Arr_param_y.shape[0]-1)-fila,columna,iElemR]
            Estados_clasificados[(Arr_param_y.shape[0]-1)-fila,columna, :] = Frecuencias[Array_bool][0:cant_simulaciones]
            
        # Ahora que tengo los estados clasificados de las simulaciones dentro del conjunto de simulaciones
        # más similares, lo siguiente es determinar cuáles son los estados predominantes, así hago gráficos
        # de los dos estados más predominantes.
        
        Estados, Cuentas = np.unique(Estados_clasificados, return_counts=True)
        estados_dominantes = Estados[np.argsort(Cuentas)[-2:]]
        
        # Construyo mi array ZZ para graficar el mapa de colores de FEF.
        ZZ_estados = np.zeros((2,XX.shape[0],XX.shape[1]))
        
        for grafico,estado in enumerate(estados_dominantes):
            for columna,PARAM_X,fila,PARAM_Y in Tupla_total:
                
                ZZ_estados[grafico,(Arr_param_y.shape[0]-1)-fila,columna] = np.count_nonzero(Estados_clasificados[(Arr_param_y.shape[0]-1)-fila,columna] == estado)/Estados_clasificados.shape[2]
                
        for grafico,titulo,direc in zip(np.arange(2),["Segundo estado más probable","Primer estado más probable"],["SegundoDom","PrimeroDom"]):
            # Una vez que tengo el ZZ completo, armo mi mapa de colores
            direccion_guardado = Path("../../../Imagenes/{}/FEF {}_{}vs{}_r{}.png".format(carpeta/"Sin Cruz",direc,code_y,code_x,i))
            
            plt.rcParams.update({'font.size': 44})
            plt.figure("FEF",figsize=(28,21))
            plt.xlabel(r"${}$".format(SIM_param_x))
            plt.ylabel(r"${}$".format(SIM_param_y))
            
            # Hago el ploteo del mapa de colores con el colormesh

            plt.pcolormesh(XX,YY,ZZ_estados[grafico],shading="nearest", cmap = "plasma",vmin = 0, vmax = 1)
            plt.colorbar()
            plt.title("Fracción de estados de {} \n {}, {} simulaciones \n {} vs {}".format(Nombres[int(estados_dominantes[grafico])],titulo,cant_simulaciones,dict_labels[code_y],dict_labels[code_x]))
            
            # Guardo la figura y la cierro
            
            plt.savefig(direccion_guardado , bbox_inches = "tight")
            plt.close("FEF")
    

#-----------------------------------------------------------------------------------------------
            
# Armo una función que en el punto de mínima distancia media construya un histograma de las distancias de JS

def Histograma_distancias(Dist_JS, code_x, code_y, DF_datos, dict_labels, carpeta,
                          ID_param_x, ID_param_y):

    
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    #-----------------------------------------------------------------------------------------
    
    # Lo que quiero hacer acá es armar gráficos de promedios de opiniones rankeados.
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    bines = np.linspace(0,1,21)
    Y, _ = np.histogram(Dist_JS[tupla], bins = bines)
    
    # Set the figure size
    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(28, 21))  # Adjust width and height as needed
    plt.bar(bines[:-1], Y/np.sum(Y), width = (bines[1]-bines[0])*0.9, align = "edge")
    plt.xlabel("Distancia JS")
    plt.ylabel("Probabilidad")
    plt.title("{} vs {}".format(dict_labels[code_y],dict_labels[code_x]))
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Hist distancias_{} vs {}_{}={}_{}={}.png".format(carpeta,code_y,code_x,ID_param_y,YY[tupla],ID_param_x,XX[tupla]))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()
    
#-----------------------------------------------------------------------------------------------

# Lo que quiero es ver cuál es la composición de los estados que son parte del cluster
# de distancias pequeñas que observo en el histograma de Distancias. 

def Comp_estados(Dist_JS, code_x, code_y, DF_datos, Dic_Total, dict_labels, carpeta, path, dist_lim,lminimos,
                 ID_param_x, SIM_param_x, ID_param_y, SIM_param_y):
    
    # Defino los arrays de parámetros diferentes
    EXTRAS = int(np.unique(DF_datos["Extra"]))
    Arr_param_x = np.unique(DF_datos["parametro_x"])
    Arr_param_y = np.unique(DF_datos["parametro_y"])
    
    # Construyo las grillas que voy a necesitar para el pcolormesh.
    XX,YY = np.meshgrid(Arr_param_x,np.flip(Arr_param_y))
    
    
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    # tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    for imin,tupla in enumerate(lminimos):
        cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
        
        PARAM_X = XX[tupla]
        PARAM_Y = YY[tupla]
        
        # for PARAM_X,PARAM_Y in zip(XX[tupla[0]-1:tupla[0]+2,tupla[1]-1:tupla[1]+2].flatten(),YY[tupla[0]-1:tupla[0]+2,tupla[1]-1:tupla[1]+2].flatten()):
        
        Frecuencias = Identificacion_Estados(Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Entropia"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmax"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Sigmay"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Covarianza"],
                                                     Dic_Total[EXTRAS][PARAM_X][PARAM_Y]["Promedios"])
        
        Nombres = ["Cons Neut", "Cons Rad", "Pol 1D y Cons","Pol Id", "Trans", "Pol Desc","Pol 1D y Cons anch","Pol Id anch", "Trans anch","Pol Desc anch"]
        
        bin_F = np.arange(-0.5,10.5)
        bin_D = np.linspace(0,1,21)
        X = np.arange(10)
        
        for i,dmin,dmax in zip(np.arange(bin_D.shape[0]-1),bin_D[0:-1],bin_D[1:]):
            Arr_bool = (Dist_JS[tupla] >= dmin) & (Dist_JS[tupla] <= dmax) 
            if np.count_nonzero(Arr_bool) == 0:
                continue
            plt.rcParams.update({'font.size': 44})
            plt.figure(figsize=(28, 21))  # Adjust width and height as needed
            plt.hist(Frecuencias[Arr_bool], bins = bin_F, density = True)
            plt.ylabel("Fracción")
            plt.title('{} vs {}\n'.format(dict_labels[code_y], dict_labels[code_x]) + r'Cantidad simulaciones {}, ${}$={},${}$={}, Distancias entre {:.2f} y {:.2f}'.format(np.count_nonzero(Arr_bool), SIM_param_y, PARAM_Y, SIM_param_x, PARAM_X, dmin, dmax))
            plt.xticks(ticks = X, labels = Nombres, rotation = 45)
            direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/Comp est_{}vs{}_min={}_b{}.png".format(carpeta,code_y,code_x,imin+1,i))
            plt.savefig(direccion_guardado ,bbox_inches = "tight")
            plt.close()
    
    #-----------------------------------------------------------------------------------------
        
    # Una vez que tengo el ZZ completo, armo mi mapa de colores para el caso sin cruz
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/DistanciaJS_recortado_{}vs{}.png".format(carpeta,code_y,code_x))
    
    plt.rcParams.update({'font.size': 44})
    plt.figure("Distancia Jensen-Shannon",figsize=(28,21))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
    
    # Hago el ploteo del mapa de colores con el colormesh
    Dist_JS_prom = np.mean(np.sort(Dist_JS)[:,:,0:cant_sim],axis=2)
    
    # Hago el ploteo del mapa de colores con el colormesh
    
    plt.pcolormesh(XX,YY,Dist_JS_prom,shading="nearest", cmap = "viridis")
    plt.colorbar()
    plt.scatter(XX[tupla],YY[tupla], marker="X", color = "red", s = 1500)
    plt.title("Distancia Jensen-Shannon \n {} vs {}\nCantidad simulaciones {}".format(dict_labels[code_y],dict_labels[code_x], cant_sim))
    
    # Guardo la figura y la cierro
    
    plt.savefig(direccion_guardado , bbox_inches = "tight")
    plt.close("Distancia Jensen-Shannon")
    
#-----------------------------------------------------------------------------------------------

# A partir de mis gráficos de histogramas de distancias, quiero armar un gráfico
# que mire los histogramas y construya la cantidad de gráficos que tienen X configuraciones
# con distancias menor a la distancia límite, que por ahora es 0.45.

def FracHist_CantEstados(Dist_JS, code_x, code_y, DF_datos, dict_labels, carpeta, path, dist_lim):
    
    # Primero reviso mis conjuntos de distancias para cada punto del espacio. En esos puntos,
    # cuento cuántas de las simulaciones tienen distancia menor a la distancia de corte
    Cantidad = np.zeros(Dist_JS.shape[0]*Dist_JS.shape[1])
    for indice,distancias in enumerate(np.reshape(Dist_JS,(Dist_JS.shape[0]*Dist_JS.shape[1],Dist_JS.shape[2]))):
        Cantidad[indice] = np.count_nonzero(distancias <= dist_lim)
    
    # Con eso puedo entonces contar cuántos histogramas tienen X simulaciones con distancias
    # menor a la distancia de corte.
    Y = np.zeros(Dist_JS.shape[2]+1)
    unicos,cant = np.unique(Cantidad, return_counts = True)
    Y[unicos.astype(int)] = cant/np.sum(cant)
    
    
    # Promedio las distancias del espacio de parámetros
    Dist_JS_prom = np.mean(Dist_JS, axis=2)
    # Calculo el mínimo de la distancia Jensen-Shannon y marco los valores de Beta y Cosd en el que se encuentra
    tupla = np.unravel_index(np.argmin(Dist_JS_prom),Dist_JS_prom.shape)
    cant_sim = np.count_nonzero(Dist_JS[tupla] <= dist_lim)
    
    # Set the figure size
    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(28, 21))  # Adjust width and height as needed
    plt.plot(np.arange(Dist_JS.shape[2]+1), Y, "--g", linewidth = 6)
    plt.xlabel("Número de configuraciones con distancia menor a 0.45")
    plt.ylabel("Fracción de Histogramas")
    plt.axvline(x=cant_sim, linestyle = "--", color = "red", linewidth = 4)
    plt.title("{} vs {}\n Fracción de histogramas en función de la cantidad de configuraciones con distancia menor a {}".format(dict_labels[code_y],dict_labels[code_x],dist_lim))
    direccion_guardado = Path("../../../Imagenes/{}/Sin Cruz/FracHistvsEstados_{} vs {}.png".format(carpeta,code_y,code_x))
    plt.savefig(direccion_guardado ,bbox_inches = "tight")
    plt.close()

