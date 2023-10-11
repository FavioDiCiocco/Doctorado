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
        # Con las opiniones finales de todas las simulaciones lo que hago es calcular la entropia
        # de la distribución de las opiniones. Esto es una segunda métrica para ver si las opiniones
        # finales de los agentes están dispersas en dos puntos grandes, o no. Además, esto sería
        # sensible a cómo es esa distribución.
        
        ZZ[(Arr_param_y.shape[0]-1)-fila,columna] = Entropia(Opifinales)
    
    #--------------------------------------------------------------------------------
    
    # Una vez que tengo el ZZ completo, armo mi mapa de colores
    direccion_guardado = Path("../../../Imagenes/{}/Entropia Opiniones EP_{}={}.png".format(carpeta,ID_param_extra_1,KAPPAS))
    
    plt.rcParams.update({'font.size': 24})
    plt.figure("Entropia Opiniones",figsize=(20,15))
    plt.xlabel(r"${}$".format(SIM_param_x))
    plt.ylabel(r"${}$".format(SIM_param_y))
    
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
    KAPPAS = int(np.unique(DF["Kappas"]))
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
            Opifinales = np.array(Datos[5][:-1:], dtype="float")
            
            # De esta manera tengo mi array que me guarda las opiniones finales de los agente.
            
            #----------------------------------------------------------------------------------------------------------------------------------
            
            # Esto me registra la simulación que va a graficar. Podría cambiar los nombres y colocar la palabra sim en vez de iter.
            repeticion = int(DF.loc[DF["nombre"]==nombre,"iteracion"])
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


"""
#--------------------------------------------------------------------------------

# Esta función me construye el gráfico de Saturación en función del tiempo
# para cada tópico para los agentes testigos.

def Graf_Punto_fijo_3D(DF,path,carpeta,T=2,nombre_parametro_2="parametro2",titulo_parametro_1="parametro 1" ,titulo_parametro_2="parametro 2"):
    
    AGENTES = int(np.unique(DF["n"]))
    
    # Defino los valores de Parametro_1 que planeo graficar
    Valores_importantes = [0] #,math.floor(len(np.unique(DF["parametro_1"]))/3),
#                           math.floor(2*len(np.unique(DF["parametro_1"]))/3),
#                           len(np.unique(DF["parametro_1"]))-1]
    
    Array_parametro_1 = np.unique(DF["parametro_1"])[Valores_importantes]
    Array_parametro_2 = np.unique(DF["parametro_2"])
    
    Tupla_total = [(parametro_1,numero_2,parametro_2) for parametro_1 in Array_parametro_1
                   for numero_2,parametro_2 in enumerate(Array_parametro_2)]
    
"""