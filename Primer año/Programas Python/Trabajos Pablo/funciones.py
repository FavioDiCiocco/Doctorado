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


def Graf_opi_vs_tiempo(DF,path,carpeta):
    # Partiendo de la idea de que el pandas no me tira error si el parámetro no está en la lista, sino que simplemente
    # me devolvería un pandas vacío, puedo entonces simplemente iterar en todos los parámetros y listo. Para eso
    # me armo una lista de tuplas, y desempaco esas tuplas en todos mis parámetros.
    
    # Como graficar en todas las combinaciones de parámetros implica muchos gráficos, voy a 
    # simplemente elegir tres valores de cada array, el primero, el del medio y el último.
    
    arrayN = np.unique(DF["n"])
    arrayAlfa = np.unique(DF["alfa"])[::math.floor((len(np.unique(DF["alfa"]))-1)/2)]
    arrayCdelta = np.unique(DF["cdelta"])[::math.floor((len(np.unique(DF["cdelta"]))-1)/2)]
    arrayMu = np.unique(DF["mu"])[::math.floor((len(np.unique(DF["mu"]))-1)/2)]
    
    Tupla_total = [(n,alfa,cdelta,mu) for n in arrayN
                   for alfa in arrayAlfa
                   for cdelta in arrayCdelta
                   for mu in arrayMu]
    
    #  No me parece mal definir esto a mano.
    TIPO = "Testigos"
    
    for AGENTES,ALFA,CDELTA,MU in Tupla_total:
        
        # Acá estoy recorriendo todos los parámetros combinados con todos. Lo que queda es ponerme a armar la lista de archivos a recorrer
        archivos = np.array(DF.loc[((DF["tipo"]==TIPO) & 
                                    (DF["n"]==AGENTES) & 
                                    (DF["mu"]==MU) & 
                                    (DF["cdelta"]==CDELTA) & 
                                    (DF["alfa"]==ALFA), "nombre")])

        #-----------------------------------------------------------------------------------------
        
        for nombre in archivos:
            
            # De los archivos de Testigos levanto las opiniones de todos los agentes a lo largo de todo el proceso.
            # Estos archivos tienen las opiniones de seis agentes.
            
            Datos = ldata("{}/{}".format(path,nombre))
            
            Testigos = np.zeros((len(Datos)-2,len(Datos[1])-1)) # Inicializo mi array donde pondré las opiniones de los testigos.
            
            for i,fila in enumerate(Datos[1:-1:]):
                Testigos[i] = fila[:-1]
            
            # De esta manera tengo mi array que me guarda los datos de los agentes a lo largo de la evolución del sistema.
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            repeticion = int(nombre.strip(".file").split("_")[5].split("=")[1])
            
            if repeticion in [0,1]:
            
                plt.rcParams.update({'font.size': 32})
                plt.figure("Topico",figsize=(20,15))
                X = np.arange(Testigos.shape[0])*0.01
                for sujeto in range(int(AGENTES)):
                    plt.semilogy(X,Testigos[:,sujeto*2], linewidth = 6)
                    plt.semilogy(X,Testigos[:,sujeto*2+1], linewidth = 6)
                plt.xlabel("Tiempo")
                plt.ylabel("Tópico")
                plt.grid(alpha = 0.5)
                plt.savefig("../../Imagenes/{}/OpivsT_N={:.0f}_alfa={:.3f}_Cdelta={:.2f}_mu={:.2f}_sim={}.png".format(carpeta,AGENTES,ALFA,CDELTA,MU,repeticion),bbox_inches = "tight")
                plt.close("Topico")
            
            #-----------------------------------------------------------------------------------------------