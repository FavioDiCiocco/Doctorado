# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:53:51 2021

@author: Favio
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import time
import math
import os

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

#--------------------------------------------------------------------------------

# Voy a definir TODAS las funciones que voy a usar, total definirlas no roba
# tiempo o proceso al programa.


# Esto printea una cantidad de valores cant de un objeto iterable que paso
# en la parte de lista.
def scan(lista,cant=10):
    i=0
    for x in lista:
        print(x)
        i+=1
        if i>cant:
            break
            
        
# Esto va al final de un código, simplemente printea cuánto tiempo pasó desde la última
# vez que escribí el inicio del cronómetro t0=time.time()
def Tiempo():
    t1=time.time()
    print("Esto tardó {} segundos".format(t1-t0))


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

    
# Esta función calcula el valor del Alfa crítico que grafico en los mapas de colores
# A diferencia de lo que tenía antes, voy a tener que introducir al GM como una variable
def AlfaC(x,GM):
    T = 2 # Número de tópicos
    K = 3 # Influencia social
    if(x>0):
        alfa = (T-1)/((GM*K)*(T-1+x))
    else:
        alfa = (T-1)/((GM*K)*(T-1-x))
    return alfa


# Voy a definir una función para usar en esta celda, que me permite asignar
# a cada ángulo un color. La idea es que esta función reciba como datos
# el vector a clasificar y la cantidad de pedacitos en la cual divido los
# 360 grados de la circunferencia. Luego me devuelve un número, que va a ser
# el índice en el cual se halla el color de ese vector. El índice lo uso
# para buscar el color en un vector que previamente voy a definir con
# ciertos valores de colores en cada índice. 
# IMPORTANTE: Esto vale sólo para vectores 2D

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


# Voy a definir una función que tome un estado del sistema y a partir de las opiniones
# de los agentes pueda determianr si el sistema se encuentra en un estado de Consenso,
# Polarización Descorrelacionada o Estado ideológico. La idea es que la función reciba
# el estado final del sistema y a partir de este devuelva un string que defina en cual
# de mis tres estados me encuentro. Básicamente la idea de esto es que si mis opiniones
# están muy cerca del cero, entonces estoy en consenso porque ese cae al cero fuerte.
# Entonces pedir que el máximo sea menor a 0,1 es en sí una cota bastante generosa.
# Por el otro lado, el estado ideológico me deja valores sobre una sola diagonal,
# entonces si el producto de la opinión del tópico 1 con la del tópico 2 para
# todos los agentes me da con un único signo, estoy en el caso de estado ideológico.
# Finalmente, si algunos de estos productos me dan positivos y otros negativos,
# entonces estoy en Polarización Descorrelacionada.

def EstadoFinal(Array,Histo,Bins):
    
    Topicos = 2

    # Primero identifico los tres posibles picos
    
    Nbins = len(Bins) # Esto es la cantidad de bins totales en los que dividí mi distribución
    
    Pcero = Histo[int((Nbins-1)/2)] # Esto es cuánto mide el pico en cero
    
    DistNegativa = Histo[0:int((Nbins-1)/2)] # Este es la distribución de las opiniones negativas
    Pmenos = max(DistNegativa) # Esto es la altura del pico de Distribución negativa
    
    DistPositiva = Histo[int((Nbins-1)/2)+1::] # Este es la distribución de las opiniones positivas
    Pmas = max(DistPositiva) # Esto es la altura del pico de Distribución positiva
    
    
    ###########################################################################
    
    # Ahora que tengo los picos, puedo empezar a definir el tema de los estados
    #finales. Arranco por el Consenso
    
    if Pcero == max(Pcero,Pmas,Pmenos):   #Pcero*0.85 > Pmax:  (Esto es la versión con umbral)
        return "Consenso"
    
    ###########################################################################
    
    # Ahora veamos el caso de región de transición. Este estado
    # Lo caracterizo porque el sistema tiene un pico central y picos por fuera
    # que no son definitorios del estado
    
#    Pmediano = min(Pmas,Pmenos)
    
#    if Pmaximo*0.7 < Pcero < Pmaximo*1.3 and Pmaximo*0.7 < Pmas < Pmaximo*1.3 and Pmaximo*0.7 < Pmenos < Pmaximo*1.3 :
#        return "RegionTrans"
    
    ###########################################################################
    
#    indicemenos = np.where(DistNegativa == Pmenos)[0][0]
#    indicemas = np.where(DistPositiva == Pmas)[0][0]
    
    if Pcero == min(Pcero,Pmas,Pmenos):  #and Pcero < Pmediano*0.85: (Esto es de la versión umbral)
        
        # Filtro los agentes que se hayan desviado apenas, cosa
        # de que el criterio final se decida con los agentes que se polarizaron
        # correctamente y no los que terminaron en cualquier lugar cerca del
        # cero.
        ArrayT1 = Array[0::2]
        ArrayT2 = Array[1::2]
        Maximo = max(np.absolute(Array))
        
        OpinionesFiltradas = np.zeros(len(Array))
        
        for agente,x1,x2 in zip(np.arange(len(ArrayT1)),ArrayT1,ArrayT2):
            if abs(x1) > Maximo*0.3 and abs(x2) > Maximo*0.3:
                OpinionesFiltradas[0+agente*Topicos:2+agente*Topicos] = [x1,x2]
        
        # Ahora veamos los otros dos casos. Primero voy a armar
        # un array que tenga las opiniones del tópico 1, y otro
        # con las opiniones del tópico 2.
        
        ArrayCuad = ClasificacionCuadrantes(OpinionesFiltradas)
        
        Cant1 = np.count_nonzero(ArrayCuad == 1)
        Cant2 = np.count_nonzero(ArrayCuad == 2)
        Cant3 = np.count_nonzero(ArrayCuad == 3)
        Cant4 = np.count_nonzero(ArrayCuad == 4)
        
        if Cant2 > 0 and Cant4 > 0 and Cant1 == 0 and Cant3 == 0:
            return "Ideologico"
        elif Cant2 == 0 and Cant4 == 0 and Cant1 > 0 and Cant3 > 0:
            return "Ideologico"
        else:
            return "Polarizacion"
    
    return "RegionTrans"


# Voy a definir una función que tome un array con opiniones del sistema y me 
# diga en qué cuadrante se encuentran cada una de estas coordenadas. Luego a
# la salida me devuelve un array con el cual me voy a armar un histograma

def ClasificacionCuadrantes(Array):
    
    # Primero tomo el array y con sign reviso si sus variables son positivas o negativas.
    # Luego, creo el array Resultado que es lo que voy a returnear.
    # Lo siguiente es crear el array SwitchDic, que va a funcionar como un Switch para los
    # casos que voy a considerar.
    
    Resultado = np.zeros(int(len(Array)/2))
    SwitchDic = dict()
    
    # Defino todos los casos posibles
    
    SwitchDic[(1,1)] = 1
    SwitchDic[(-1,1)] = 2
    SwitchDic[(-1,-1)] = 3
    SwitchDic[(1,-1)] = 4

    
    # Repaso los elementos en Signos para identificar los cuadrantes de mis objetos.
    
    for x1,x2,indice in zip(Array[0::2],Array[1::2],np.arange(len(Array[0::2]))):
        Absolutos = np.abs(np.array([x1,x2]))
        if max(Absolutos)<0.5 or x1==0 or x2==0:
            Resultado[indice] = 0
        else:
            Signos = np.sign(np.array([x1,x2]))
            Resultado[indice] = SwitchDic[(Signos[0],Signos[1])]
  
    return Resultado


# Acá lo que voy a hacer es preparar los colores que voy a usar para definir los puntos finales
# de las trayectorias de las opiniones

Divisiones = 144
color=cm.rainbow(np.linspace(0,1,Divisiones))


# Lo que hice acá es definir una ¿lista? que tiene en cada casillero los datos que definen un color.
# Tiene diferenciados 144 colores, es decir que tengo un color para cada región de 2.5 grados. Estas regiones
# las voy a distribuir centrándolas en cada ángulo que cada color representa. Por lo tanto,
# Los vectores que tengan ángulo entre -1.25º y 1.25º tienen el primer color. Los que tengan entre
# 1.25º y 3.75º tienen el segundo color. Y así. Por tanto yo tengo que hallar una fórmula que para
# cada ángulo le asigne el casillero que le corresponde en el vector de color. Luego, cuando grafique
# el punto, para el color le agrego un input que sea: c = color[n]



# Estas son todas las funciones que voy a necesitar. Lo último no es una función,
# es la lista de colores que necesitaba para usar la función Indice_Color.
# Ahora voy a definir la parte que se encarga de levantar los nombre y el path
# de los archivos.

t0 = time.time()

#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

T=2 # Defino acá el número de tópicos porque es algo que no cambia por ahora,
# pero no tenía dónde más definirlo

SuperDiccionario = dict()

for Carpeta in ["PruebaMod"]:

   # CÓDIGO PARA LEVANTAR ARCHIVOS DE UNA CARPETA CON TODOS LOS ARCHIVOS MEZCLADOS
    
    CarpCheck=[[root,files] for root,dirs,files in os.walk("../{}".format(Carpeta))]
    
    # El elemento en la posición x[0] es el nombre de la carpeta
    
    for x in CarpCheck:
        # dada = x[0].split("\\")
        Archivos_Datos = [nombre for nombre in x[1]]
        Archivos_Datos.insert(0,x[0])
    
    #-------------------------------------------------------------------------------------------------------
    
    # Es importante partir del hecho de que mis archivos llevan por nombre: "Opiniones_alfa=$_N=$_Cosd=$_Iter=$.file"
    # También tengo otros archivos llamados "Testigos_alfa=$_N=$_Cosd=$_Iter=$.file"
    
    Conjunto_Direcciones = [Archivos_Datos[0]]
    
    SuperDiccionario[Carpeta] = dict()
    
    for nombre in Archivos_Datos[1::]:
        tipo = nombre.split("_")[0]
        alfa = float(nombre.split("_")[1].split("=")[1])
        N = int(nombre.split("_")[2].split("=")[1])
        Cdelta = float(nombre.split("_")[3].split("=")[1])
        if tipo not in SuperDiccionario[Carpeta].keys():
            SuperDiccionario[Carpeta][tipo] = dict()
            SuperDiccionario[Carpeta][tipo][N] = dict()
            SuperDiccionario[Carpeta][tipo][N][Cdelta] = dict()
            SuperDiccionario[Carpeta][tipo][N][Cdelta][alfa] = [nombre]
        if N not in SuperDiccionario[Carpeta][tipo].keys():
            SuperDiccionario[Carpeta][tipo][N] = dict()
            SuperDiccionario[Carpeta][tipo][N][Cdelta] = dict()
            SuperDiccionario[Carpeta][tipo][N][Cdelta][alfa] = [nombre]
        elif Cdelta not in SuperDiccionario[Carpeta][tipo][N].keys():
            SuperDiccionario[Carpeta][tipo][N][Cdelta] = dict()
            SuperDiccionario[Carpeta][tipo][N][Cdelta][alfa] = [nombre]
        elif alfa not in SuperDiccionario[Carpeta][tipo][N][Cdelta].keys():
            SuperDiccionario[Carpeta][tipo][N][Cdelta][alfa] = [nombre]
        else:
            SuperDiccionario[Carpeta][tipo][N][Cdelta][alfa].append(nombre)

    
    # Le hice una modificación a esta parte del código, ahora esto trabaja
    # armando el SuperDiccionario también, no sólo los Conjuntos de Alfa, Cdelta
    # y demás. Lo bueno de esto es que ahora el armado del SuperDiccionario
    # es mucho más rápido.
    
    # En este caso SuperDiccionario está juntando tanto los nombres
    # de Datos Testisgos con los de Datos Opiniones.
    #--------------------------------------------------------------------------------------------
    
    for AGENTES in [200]:
        
        #---------------------------------------------------------------------------------------------------------------
        
        TIPO = "Testigos"
        
        # Acá me voy a inicializar los arrays para graficar los mapas de colores de la red.
        
        
        # Defino el conjunto de valores de Cdelta a recorrer. Tengo que ordenarlos porque
        # el SuperDiccionario los ordena como los recibe, y el os.walk no necesariamente recorre
        # todo en orden.
        
        Conjunto_Cdelta = list(SuperDiccionario[Carpeta][TIPO][AGENTES].keys())
        Conjunto_Cdelta.sort()
        
        # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
        # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
        # El CDELTA que estoy usando 0. Igual
        # no importa, la idea es que definitivamente todos los Cdeltas tienen el mismo
        # conjunto de alfas. De no ser así, no podría armar correctamente estos gráficos.
        
        
        Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][0].keys())
        Conjunto_Alfa.sort()
        
        """
        
        Conjunto_Alfa.reverse() # Lo invierto para que me quede el uno arriba y no abajo
            
        x = np.array(Conjunto_Cdelta)
        y = np.array(Conjunto_Alfa)
        
        Conjunto_Alfa.reverse() # Lo vuelvo a invertir para que de nuevo esté como siempre.
        
        XX,YY = np.meshgrid(x,y)
        
        ZZ = np.zeros(XX.shape)
        
        # Con esto ya tengo armados los grids de XX,YY y de paso me armo el grid
        # del ZZ para ir rellenándolo a medida que corro todo el programa, o usando
        # los datos que ya guardé de antes. Para eso es el módulo siguiente
        
        ZZV = np.zeros(XX.shape) # Esta es la matriz de valores para la varianza
        ZZP = np.zeros(XX.shape) # Esta es la matriz de valores para los valores de opiniones máximas
        ZZE = np.zeros(XX.shape) # Esta es la matriz con valores de entropía para las distribuciones por cuadrantes
        # Con esto ya tengo armados los grids de XX,YY y de paso me armo el grid
        # del ZZ para ir rellenándolo a medida que corro todo el programa, o usando
        # los datos que ya guardé de antes. Para eso es el módulo siguiente
            
        print("Inicialicé las matrices de los mapas de colores")
        
        """
        #----------------------------------------------------------------------------------------------------------
        
        # Armo tres arrays, uno para guardar los grados medio de todas mis redes a lo largo de todas las simulaciones,
        # uno para guardar la cantidad de iteraciones que tomaron todas las redes a lo largo de todas las simulaciones,
        # y uno para guardar los valores de actividades medios de todas las redes a lo largo de todas las simulaciones.
        GradosTotales = np.zeros((len(Conjunto_Cdelta),len(Conjunto_Alfa)*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])))
        TiemposTotales = np.zeros((len(Conjunto_Cdelta),len(Conjunto_Alfa)*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])))
        ActividadesTotales = np.zeros((len(Conjunto_Cdelta),len(Conjunto_Alfa)*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])))
        
        
        for CDELTA,icdelta in zip(Conjunto_Cdelta,np.arange(len(Conjunto_Cdelta))):
            
            # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
            # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
            
            Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA].keys())
            Conjunto_Alfa.sort()
            
            """
            # Voy a armar el OpiMaxima para cada Cdelta
            # Reinicio la OpiMaxima en 0
            OpiMaxima = 0
            
            
            for ALFA,ialfa in zip(Conjunto_Alfa[0:30],np.arange(len(Conjunto_Alfa))):
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                    
                    
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla

                    #--------------------------------------------------------------------------------------------
                
                    # Levanto los datos del archivo original y separo los datos en tres listas.
                    # Una para la matriz de Adyacencia, una para la matriz de superposición y una para los vectores de opiniones
                
                    # Tengo que hacer una diferenciación entre levantar datos de las carpetas subdividida en 4 y de la carpeta con todo mezclado.

                    # PARA LEVANTAR DATOS DE LA CARPETA TODA MEZCLADA
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de las Opiniones Finales
                    Opi = np.array([float(x) for x in Datos[5][1::]])

                    
                    #-----------------------------------------------------------------------------------------------
                    
                    # Acá voy armando el cálculo de la OpiMáxima
                    
                    OpiMaxima = max(max(np.absolute(Opi)),OpiMaxima)
                    
                    #------------------------------------------------------------------------------------------------
            
            print("Calculé OpiMaxima")
            Tiempo()
            
            """
            for ALFA,ialfa in zip(Conjunto_Alfa,np.arange(len(Conjunto_Alfa))):
                
                # Armo mis arrays de Opiniones iniciales y finales
                
                OpinionesFinales = np.array([])
                OpinionesIniciales = np.array([])
                
                Graficar = [10,20] 
                
                # Inicio mis arrays para guardar los datos de Tiempos de Simulación y de Promedios
                
                TideSi = np.zeros(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))
                Promedios = np.zeros(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                    
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla

                    #--------------------------------------------------------------------------------------------
                
                    # Levanto los datos del archivo original y separo los datos en tres listas.
                    # Una para la matriz de Adyacencia, una para la matriz de superposición y una para los vectores de opiniones
                
                    # Tengo que hacer una diferenciación entre levantar datos de las carpetas subdividida en 4 y de la carpeta con todo mezclado.

                    # PARA LEVANTAR DATOS DE LA CARPETA TODA MEZCLADA
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de Opinión iniciales
#                    Opi0 = np.array([float(x) for x in Datos[1][1::]])
                    
                    # Levanto los datos de Variación Promedio
                    Var = np.array([float(x) for x in Datos[3][:-1]])
                    
                    # Levanto los datos de las Opiniones Finales
                    Opi = np.array([float(x) for x in Datos[5][:-1]])
                    
                    #-----------------------------------------------------------------------------------------------
                    
                    # Acá voy armando el array de las Opiniones Finales
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                    #-----------------------------------------------------------------------------------------------

                    # Voy a ir armando mi array con los valores de Tiempo de Simulación.
                    # Lo voy a llamar TideSi.
                    # Voy a guardar también los tiempos Totales acá.
                    
                    TideSi[numero] = Var.shape[0]*0.01 # 0,01 es el paso temporal dt
                    
                    TiemposTotales[icdelta,ialfa*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])+numero] = Var.shape[0]
                    
                    # Esto debería funcionar derechito.
                    
                    #------------------------------------------------------------------------------------------------
            
                    # Quiero hacer un estudio de cómo varía el máximo al cual llega el sistema en función del
                    # cos(delta) y alfa. La idea es mirar las opiniones finales, promediar los máximos en
                    # las simulaciones realizadas y desde ahí asignar ese valor a cada punto del colormap.
                    
                    Promedios[numero] = np.mean((np.abs(Opi)))
                    
                    # Esto me guarda en cada posición del array el máximo de las opiniones finales alcanzadas
                    # en cada simulación
                    
                    #----------------------------------------------------------------------------------------------
                    
                    """
                    
                    # Me guardo los datos de los grados en un solo array y con eso después calculo el grado
                    # medio del sistema "TOTAL"
                    
                    GradosTotales[icdelta,ialfa*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])+numero] = np.mean(Grados)

                    #----------------------------------------------------------------------------------------------
                    # Me guardo los datos de las actividades en un solo array y con eso después calculo la actividad
                    # media del sistema "TOTAL"
                    
                    ActividadesTotales[icdelta,ialfa*len(SuperDiccionario[Carpeta][TIPO][AGENTES][0][0])+numero] = np.mean(Actividad)

                    #----------------------------------------------------------------------------------------------
                    """
                
                    # De los archivos de Testigos levanto las opiniones de un grupo de agentes
                    # a lo largo de todo el proceso. Los archivos de Testigos tienen en su fila 1
                    # los valores de actividad de los agentes
                    
                    if TIPO == "Testigos":
                                                
                        Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                        
                        Agentes = [ numero for numero in range(int(len(Datos[1])/2))] # Lo hago un range porque al final no importa cuál testigo es cual, es pura casualidad.
                        
                        Testigos = np.zeros((len(Datos)-1,len(Datos[1])-1))
                        
                        
                        for fila,i in zip(Datos[1::], np.arange(len(Datos[1::]))):
                            Testigos[i] = fila[:-1]
                            
                        # De esta manera tengo mi array que me guarda los datos de los
                        # agentes a lo largo de la evolución del sistema.
                        
                    #----------------------------------------------------------------------------------------------------------------------------------
                    
                        repeticion = int(nombre.strip(".file").split("_")[4].split("=")[1])
                        
                        if repeticion in Graficar and ALFA in [0,2] and CDELTA in [0,1]:
                        
                            plt.rcParams.update({'font.size': 28})
                            plt.figure("Topico",figsize=(20,15))
                            X = np.arange(Testigos.shape[0])*0.01
                            for sujeto in range(int(len(Testigos[0])/2)):
                                plt.plot(X,Testigos[:,sujeto*2], linewidth = 4)
                                plt.plot(X,Testigos[:,sujeto*2+1], linewidth = 4)
                            plt.xlabel("Tiempo")
                            plt.ylabel("Tópico")
                            plt.grid(alpha = 0.5)
                            plt.annotate(r"$\alpha$={},cos($\delta$)={},N={}".format(ALFA,CDELTA,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                            plt.savefig("../../Imagenes/{}/TopicosvsT_alfa={:.3f}_Cdelta={:.2f}_N={}_iter={}.png".format(Carpeta,ALFA,CDELTA,AGENTES,repeticion),bbox_inches = "tight")
                            plt.close("Topico")
                        
                    
                     
                    print("Tengo las opiniones finales alfa={}, Cdelta={}".format(ALFA,CDELTA))
                    Tiempo()
                
                #----------------------------------------------------------------------------------------------
                
                """
                
                # Acá lo que voy a hacer es rellenar el grid de ZZ con los valores de los resultados de
                # opiniones finales
                
                # Voy a agregar un paso extra en esto porque el sistema está teniendo problemas en clasificar algunos casos límite
                EC = [("Consenso",0),("Polarizacion",1),("Ideologico",2),("RegionTrans",3)] # EC es Estados y colores. Tiene tuplas con los colores asociados
                
#                OpiMaxima = max(np.absolute(OpinionesFinales))
#                if OpiMaxima < 10:
#                    Bordes = np.round(np.linspace(-OpiMaxima,OpiMaxima,22),3)
#                else:
#                    Bordes = np.round(np.linspace(-OpiMaxima,OpiMaxima,int(OpiMaxima)*2),3)
                OpiMaxima = max(np.absolute(OpinionesFinales))
                # if OpiMaxima < 150:
                #     Bordes = np.round(np.linspace(-OpiMaxima,OpiMaxima,22),5)
                # else:
                #     BordeI = np.mean(OpinionesFinales[OpinionesFinales < 0])-30
                #     BordeD = np.mean(OpinionesFinales[OpinionesFinales > 0])+30
                #     BordeMax = max(abs(BordeI),BordeD)
                #     Bordes = np.round(np.linspace(-BordeMax,BordeMax,int(OpiMaxima/50)*2),3)
                # else:
                    # Bordes = np.round(np.linspace(-60,60,40),3)
                Bordes = np.round(np.linspace(-60,60,40),3)
                Bins = (Bordes[1::]+Bordes[0:len(Bordes)-1])/2
                
                Histo,nada = np.histogram(OpinionesFinales,bins=Bordes)
                
                
                ResultadoEF = EstadoFinal(OpinionesFinales,Histo,Bins)
                
                # Esto es ya lo mismo que antes, dependiendo de ResultadoEF es el número que va en la matriz ZZ
                
                for estado in EC:
                    if ResultadoEF == estado[0]:
                        ZZ[len(Conjunto_Alfa)-1-ialfa, icdelta] = estado[1]
                
                
                #------------------------------------------------------------------------------------------------
                
                # Acá lo que voy a hacer es rellenar el grid de ZZV con los valores de los resultados de
                # opiniones finales. Vamos a modificar esto porque la Varianza resulta muy sensible ante
                # outliers. Vamos a tomar la mediana de los tiempos de simulación y con eso ver si obtenemos
                # un gráfico más claro de cómo surgen las regiones de transición entre consenso y polarización
                # a partir de la variación en los Tiempos de Simulación.
                
                Cuantiles = np.quantile(TideSi,[0.05,0.95])
               
                ZZV[len(Conjunto_Alfa)-1-ialfa, icdelta] = np.log10(Cuantiles[1]-Cuantiles[0]+1)
                
                # Esto ya debería correctamente armarme el ZZV ubicando para cada par de valores
                # alfa y cos(delta) la diferencia entre el valor que define el cuantil de 0,05 y el de 0,95.
                # Lo siguiente es tomar eso al final y graficarlo.
                    
                #-------------------------------------------------------------------------------------------------
                
                # Acá lo que voy a hacer es rellenar el grid de ZZP con los valores de los maximos promedios
                # de las opiniones finales
            
                ZZP[len(Conjunto_Alfa)-1-ialfa, icdelta] = np.mean(Promedios)
                
                # Esto ya debería correctamente armarme el ZZM ubicando para cada par de valores
                # alfa y cos(delta) su respectivo valor máximo. Lo siguiente es tomar eso al final y graficarlo.
                
                #-------------------------------------------------------------------------------------------------

                # Ok, esto me confundió mucho en un principio. La idea es que uso ClasificaciónCuadrantes para
                # armar el array que tenga los números de los cuadrantes en los que se encuentran las opiniones.
                # Defino los bins a mano, total sé que distribuyo mis opiniones en los cuadrantes del 1 al 4, y
                # el 0 es para aquellos en los que están muy cerca del cero.
                
#                    plt.figure("Distribucion por cuadrantes")

                Cuadrantes = ClasificacionCuadrantes(OpinionesFinales)
                Bins = [-0.5,0.5,1.5,2.5,3.5,4.5]
                
                # Calculo la distribución. Del np.histogram obtengo dos arrays, el primero es el histograma
                # y el segundo es los bins. Para evitar problemas, mando eso a la nada misma y listo.
                
                [Yhisto,nada] = np.histogram(Cuadrantes, bins = Bins)
                
                # Esto lo saqué del manual de uso. Simplemente grafica en el sistema de ejes del subplot
                # el gráfico de barras y se encarga de tomar siempre como piso el anterior gráfico de barras
                # de manera de ir stackeando las barras. Eso me va a permitir mostrar variso gráficos 
                # juntos y con eso recorrer varios alfas.

                
                #---------------------------------------------------------------------------------------------------
                
#                Voy a aprovechar la cuenta hecha para distribucion por Cuadrantes para ir calculando la Entropía.
#                A fin de cuentas, la entropía va a estar vista en función de la distribución de las opiniones en
#                los cuadrantes. De paso, Seba me dijo que ignore el tema de las opiniones que se van a cero.
#                Son muy pocas en general, así que no importan mucho, ¿Pero debería simplemente ignorarlas?
#                Definitivamente son despreciables, dale para adelante.
                
                Probabilidades = Yhisto[Yhisto != 0]/np.sum(Yhisto)
                ZZE[len(Conjunto_Alfa)-1-ialfa, icdelta] = np.matmul(Probabilidades, np.log2(Probabilidades))*(-1)
                
                #-----------------------------------------------------------------------------------------------------
        
        # Con el Ycdelta ya armado, grafico el Grado medio del sistema en función de Cdelta y veo si
        # ahora esto da alguna relación o nada que ver.
        
        # Me armo mi array de grados medios para cada Cdelta
        Ycdelta = np.zeros(len(Conjunto_Cdelta))
        for i in range(len(Conjunto_Cdelta)):
            Ycdelta[i] = np.mean(GradosTotales[i])
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("gradomediovscdelta",figsize=(20,12))
        plt.plot(Conjunto_Cdelta,Ycdelta,"-o",linewidth=4,markersize=8)
        plt.ylabel("<k>")
        plt.xlabel(r"cos($\delta$)")
        plt.grid()
        plt.savefig("../Imagenes/RedAct/{}/GradMedvsCdelta.png".format(Carpeta), bbox_inches = "tight")
        plt.close("gradomediovscdelta")
        
        #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        
        # Como una primer medida burda, voy a definir el Gradomedio de la red como
        # el grado medio promediado de todas las redes conexas que se llegaron a crear.
        # Estaría bueno armar un histograma de esos datos. Eso lo mandaré mañana quizás
        
        GM = np.mean(GradosTotales)
        
        # También voy a definir el grado medio calculado a partir de la actividad
        # media de los agentes. Ese es el grado medio teórico de las redes de actividad
        # obtenido utilizando la teoría del modelo de redes de Actividad.
        # El cálculo es <k> = 2*m*<a>*<reconstrucciones>. Las reconstrucciones es la cantidad
        # de veces que una red se rearma. m es la cantidad de enlaces que genera un nodo
        # al activarse y <a> es la actividad media. Esta actividad, al igual que las
        # reconstrucciones, están promediadas a lo largo de todas las simulaciones.
        
        gradomedio = 2*10*np.mean(ActividadesTotales)*(np.mean(TiemposTotales)/100)
        
        print(GM)
        print(gradomedio)
        print(np.mean(TiemposTotales))

        # Acá termino mi gráfico de Fases, una vez que recorrí todos los Alfa y Cdelta.

        # Armo mi colormap. Para eso armo una lista con los colores que voy a usar,
        # los cuales son una tuplas de tres elementos que combinan colores en proporción
        # donde el primer número es el color rojo, el segundo el verde y el tercero el azul.
        # Con eso luego uso la función de LinearSegmentedeColormap para que me traduzca
        # mi lista de colores a un colormap, y como sólo voy a graficar tres valores
        # en N pongo un 3 para separar el binneado de mi mapa de colores en sólo 3.
        # Luego le mando ese colormap al pcolormesh y listo.
        
        
        colors = [(0, 0.7, 0), (0, 0.3, 1), (1, 0.3, 0),(0.8, 0, 0.8)] #,(0,0,0)]  #  R -> G -> B -> Viol -> N
        cmap_name = 'Mi_lista'
        ColMap = LinearSegmentedColormap.from_list(cmap_name, colors, N=4)
        
        # Defino los parámetros usuales de mi gráfico
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("Espacio parametros",figsize=(20,12))
        plt.xlabel(r"cos($\delta$)")
        plt.ylabel(r"$\alpha$")
        # plt.title("Estados finales en Espacio de Parametros")
        
        # Grafico la línea del alfa con GM variable
        
        # Xv = np.array(Conjunto_Cdelta)
        # Yv = np.array([])
        # Yv = [AlfaC(x,gm) for x,gm in zip(Conjunto_Cdelta,Ycdelta)]
        # plt.plot(Xv,Yv,color = "cyan",linewidth = 5, label = r"$\alpha_v$")
        
        
        # Grafico la línea del alfa con gradomedio teórico según redes de actividad
        
        Xt = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Yt = np.array([AlfaC(x,gradomedio) for x in Xt])
        plt.plot(Xt,Yt,color = "cyan",linewidth = 5, label = r"$\alpha_t$")
        
        
        # Grafico la línea del Alfa Crítico teórico
        
        Xa = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Ya = np.array([AlfaC(x,GM) for x in Xa])
        plt.plot(Xa,Ya,"--",color = "black",linewidth = 5, label = r"$\alpha_c$")
        
        
        # Hago el ploteo del mapa de colores con el colormesh y usando el mapa de colroes creado por mi.
        
        plt.legend()
        plt.pcolormesh(XX,YY,ZZ,shading="nearest", cmap = ColMap)
        
        # plt.ylim(min(Conjunto_Alfa)-0.005,max(Conjunto_Alfa)+0.005)
        plt.savefig("../Imagenes/RedAct/{}/Estados finales EP.png".format(Carpeta), bbox_inches = "tight")
        plt.close("Espacio parametros")


        #---------------------------------------------------------------------------------------------------
        
        # Me falta guardar los valores de ZZ en un archivo, por si necesito hacerle un retoque al archivo y no 
        # quiero volver a correr todo de una
        
        # np.savetxt("Grafico Fases.csv", ZZ, delimiter = ",")

        # -------------------------------------------------------------------------------------------------

        
        # Acá termino mi gráfico de Fases, una vez que recorrí todos los Alfa y Cdelta.
    
        # Defino los parámetros usuales de mi gráfico
    
        plt.rcParams.update({'font.size': 24})
        plt.figure("Cuantil TideSi",figsize=(20,12))
        plt.xlabel(r"cos($\delta$)")
        plt.ylabel(r"$\alpha$")
        
        # Grafico la línea del alfa con GM variable
        
        # Xv = Conjunto_Cdelta
        # Yv = [AlfaC(x,gm) for x,gm in zip(Conjunto_Cdelta,Ycdelta)]
        # plt.plot(Xv,Yv,color = "green",linewidth = 5, label = r"$\alpha_v$")
        
        
        # Grafico la línea del alfa con gradomedio teórico según redes de actividad
        
        Xt = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Yt = np.array([AlfaC(x,gradomedio) for x in Xt])
        plt.plot(Xt,Yt,color = "green",linewidth = 5, label = r"$\alpha_t$")
    
        # Grafico la línea del Alfa Crítico teórico
    
        Xa = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Ya = np.array([AlfaC(x,GM) for x in Xa])
        plt.plot(Xa,Ya,"--",color = "red",linewidth = 5, label = r"$\alpha_c$")
    
        # Hago el ploteo del mapa de colores con el colormesh y usando el mapa de colroes creado por mi.
    
        plt.legend()
        plt.pcolormesh(XX,YY,ZZV,shading="nearest", cmap = "plasma")
        plt.colorbar()
        plt.savefig("../Imagenes/RedAct/{}/Cuantil EP.png".format(Carpeta), bbox_inches = "tight")
        plt.close("Cuantil TideSi")
    #     plt.show()
    
        #---------------------------------------------------------------------------------------------------
        
        # Acá termino mi gráfico de Fases, una vez que recorrí todos los Alfa y Cdelta.
    
        # Defino los parámetros usuales de mi gráfico
    
        plt.rcParams.update({'font.size': 24})
        plt.figure("Promedios EP",figsize=(20,12))
        plt.xlabel(r"cos($\delta$)")
        plt.ylabel(r"$\alpha$")
        
        # Grafico la línea del alfa con GM variable
        
        # Xv = Conjunto_Cdelta
        # Yv = [AlfaC(x,gm) for x,gm in zip(Conjunto_Cdelta,Ycdelta)]
        # plt.plot(Xv,Yv,color = "green",linewidth = 5, label = r"$\alpha_v$")
        
        
        # Grafico la línea del alfa con gradomedio teórico según redes de actividad
        
        Xt = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Yt = np.array([AlfaC(x,gradomedio) for x in Xt])
        plt.plot(Xt,Yt,color = "green",linewidth = 5, label = r"$\alpha_t$")
        
        # Grafico la línea del Alfa Crítico teórico
        
        Xa = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Ya = np.array([AlfaC(x,GM) for x in Xa])
        plt.plot(Xa,Ya,"--",color = "red",linewidth = 5, label = r"$\alpha_c$")
    #
    #    # Hago el ploteo del mapa de colores con el colormesh y usando el mapa de colores creado por mi.
    #
        plt.legend()
        plt.pcolormesh(XX,YY,ZZP,shading="nearest", cmap = "viridis")
        plt.colorbar()
        plt.savefig("../Imagenes/RedAct/{}/Promedios EP.png".format(Carpeta), bbox_inches = "tight")
        plt.close("Promedios EP")
    #     plt.show()
    
        #---------------------------------------------------------------------------------------------------
        
        # Acá termino mi gráfico de Fases, una vez que recorrí todos los Alfa y Cdelta.
    
        # Defino los parámetros usuales de mi gráfico
    
        plt.rcParams.update({'font.size': 24})
        plt.figure("Entropia EP",figsize=(20,12))
        plt.xlabel(r"cos($\delta$)")
        plt.ylabel(r"$\alpha$")
        
        # Grafico la línea del alfa con GM variable
        
        # Xv = Conjunto_Cdelta
        # Yv = [AlfaC(x,gm) for x,gm in zip(Conjunto_Cdelta,Ycdelta)]
        # plt.plot(Xv,Yv,color = "green",linewidth = 5, label = r"$\alpha_v$")
        
        
        # Grafico la línea del alfa con gradomedio teórico según redes de actividad
        
        Xt = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Yt = np.array([AlfaC(x,gradomedio) for x in Xt])
        plt.plot(Xt,Yt,color = "green",linewidth = 5, label = r"$\alpha_t$")
    
        # Grafico la línea del Alfa Crítico teórico
    
        Xa = np.arange(min(Conjunto_Cdelta),max(Conjunto_Cdelta),0.002)
        Ya = np.array([AlfaC(x,GM) for x in Xa])
        plt.plot(Xa,Ya,"--",color = "red",linewidth = 4, label = r"$\alpha_c$")
    
        # Hago el ploteo del mapa de colores con el colormesh y usando el mapa de colores creado por mi.
    
        plt.legend()
        plt.pcolormesh(XX,YY,ZZE,shading="nearest", cmap = "viridis")
        plt.colorbar()
        plt.savefig("../Imagenes/RedAct/{}/Entropía EP.png".format(Carpeta), bbox_inches = "tight")
        plt.close("Entropia EP")
        
        print("Ya terminé los mapas de colores")
        
        #---------------------------------------------------------------------------------------------------
        
        # Acá voy a armar un gráfico que sea la distribución de los grados medios de las redes, para ver
        # si la distribución de todas las redes da una curva más o menos normal.
        
        plt.rcParams.update({'font.size': 24})
        plt.figure("Distribucion gradomedio",figsize=(20,12))
        
        Y,Bines = np.histogram(GradosTotales.flatten(),bins=100,density=True)
        X = (Bines[0:Bines.shape[0]-1]+Bines[1::])/2
        
        plt.semilogy(X[Y != 0],Y[Y != 0],"-o",color = "red",linewidth = 4,markersize = 8)
        plt.grid()
        plt.xlabel("<k>")
        plt.ylabel(r"$P_{<k>}$")
            
        plt.savefig("../Imagenes/RedAct/{}/Distribucion_gradomedio.png".format(Carpeta),bbox_inches = "tight")
        plt.close("Distribucion gradomedio")

        """
        
        #---------------------------------------------------------------------------------------------------------
                    
        # Acá me voy a tener que hacer el armado de datos para graficar el Conjunto de Variaciones Promedio 
        
        TIPO = "Opiniones"
        
        for CDELTA,icdelta in zip(Conjunto_Cdelta,np.arange(len(Conjunto_Cdelta))):
        
            # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
            # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
            
            Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA].keys())
            Conjunto_Alfa.sort()
            
            # Acá estoy haciendo los gráficos de Varprom individuales. Si quisiera los gráficos de manera conjunta debería revisar la parte
            # que está comentada más abajo.
            
            for ALFA,ialfa in zip(Conjunto_Alfa,np.arange(len(Conjunto_Alfa))):
                
                Colores2 = cm.rainbow(np.linspace(0,1,len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA])))
                plt.rcParams.update({'font.size': 28})
                fig, ax = plt.subplots(figsize = (20, 15))
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Varprom = np.array([float(x) for x in Datos[3][:-1]])
                    
                    # ----------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto es el tiempo a graficar
                    X = np.arange(len(Varprom))*0.01
                    
                    # Ahora grafico las curvas de Variación de Opiniones
                    ax.semilogy(X,Varprom,"--",c = Colores2[numero], linewidth = 4)
                    
                    # ----------------------------------------------------------------------------------------------------------------------------
                
                # Acá coloco ciertas marcas importantes para el gráfico
                
                CritCorte = 0.0005
                ax.axhline(CritCorte)
                ax.annotate(r"$\alpha$={},cos($\delta$)={},N={}".format(ALFA,CDELTA,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                ax.grid(alpha = 0.5)
                        
            #-------------------------------------------------------------------------------------------
            
                # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                # en volverse conexo
                
                plt.savefig("../../Imagenes/{}/Varprom_alfa={:.2f}_Cdelta={:.2f}.png".format(Carpeta,ALFA,CDELTA),bbox_inches = "tight")
                plt.close(fig)
            
            """
            # Inicio la figura con los múltiples gráficos de Variación Promeido
            plt.rcParams.update({'font.size': 24})
            fig = plt.figure("Varprom",figsize=(64,36))
            gs = fig.add_gridspec(6,5,hspace=0,wspace=0)
            axs = gs.subplots(sharex=True, sharey=True)
            
            # Estas son las columnas y filas a recorrer para el gráfico
            Columnas = [0,1,2,3,4]*6
            Filas = [i for i in range(0,6) for j in range(0,5)]
            
            for ALFA,fila,columna in zip(Conjunto_Alfa[0:30],Filas,Columnas):
                
                # Defino los colores para graficar las variaciones Promedio
                Colores2 = cm.rainbow(np.linspace(0,1,len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA])))
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Varprom = np.array([float(x) for x in Datos[3][1::]])
                    
                # ----------------------------------------------------------------------------------------------------------------------------
                    
                    # Esto es el tiempo a graficar
                    X = np.arange(len(Varprom))*0.01
                    
                    # Ahora grafico las curvas de Variación de Opiniones
                    axs[fila,columna].semilogy(X,Varprom,"--",c = Colores2[numero],linewidth = 4)
                    
                # ----------------------------------------------------------------------------------------------------------------------------
                
                # Acá coloco ciertas marcas importantes para el gráfico
                
                CritCorte = 0.0005
                axs[fila,columna].axhline(CritCorte)
                axs[fila,columna].annotate(r"$\alpha=${}".format(ALFA), xy=(0.8,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                axs[fila,columna].grid()
                        
            #-------------------------------------------------------------------------------------------
            
            # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
            # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
            # en volverse conexo
            
            plt.savefig("../Imagenes/RedAct/{}/Varprom_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
            plt.close("Varprom")
                
            """
        #-----------------------------------------------------------------------------------------------------------
                    
        # Acá me voy a tener que hacer el armado de datos para graficar el Conjunto de Hist2D   
        
        for CDELTA,icdelta in zip(Conjunto_Cdelta,np.arange(len(Conjunto_Cdelta))):
        
            # Para cada Cdelta defino el conjunto de Alfas asociados a recorrer.
            # De nuevo, lo ordeno porque el SuperDiccionario no los tiene necesariamente ordenados.
            
            Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA].keys())
            Conjunto_Alfa.sort()
            
            # Acá estoy haciendo los gráficos de Hist2D individuales. Si quisiera los gráficos de manera conjunta debería revisar la parte
            # que está comentada más abajo.
            
            for ALFA,ialfa in zip(Conjunto_Alfa,np.arange(len(Conjunto_Alfa))):
                
                plt.rcParams.update({'font.size': 28})
                fig, ax = plt.subplots(figsize = (20, 15))
                
                # Armo el array de Opiniones Finales
                OpinionesFinales = np.array([])
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Semilla
                    
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][:-1]])
                    
                    # Me voy armando un array con las opiniones finales de los agentes a lo largo
                    # de todas las simulaciones
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                    # -------------------------------------------------------------------------------------------------
                
                # Acá voy a armar gráficos conjuntos de los Histogramas 2D cosa de que el etiquetado se realice de mejor forma.
                # Dios quiera que esto salga fácil y rápido.
                
                # OpiMaxima = max(np.absolute(OpinionesFinales))
                # OpiExtremos = np.array([OpiMaxima,OpiMaxima])
                # Opiniones2D = np.concatenate((OpinionesFinales,OpiExtremos), axis=None)
                

                # Ahora grafico las curvas de distribución de ambas opiniones
                # ax.hist2d(Opiniones2D[0::2],Opiniones2D[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                ax.hist2d(OpinionesFinales[0::2],OpinionesFinales[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                # if ALFA == 2 and CDELTA >= 0:
                    # ax.set_xlim(0,150)
                    # ax.set_ylim(0,150)
                ax.annotate(r"$\alpha$={},cos($\delta$)={},N={}".format(ALFA,CDELTA,AGENTES), xy=(0.7,0.9),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                        
                #-------------------------------------------------------------------------------------------
                
                # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
                # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
                # en volverse conexo
                
                plt.savefig("../../Imagenes/{}/Hist2D_alfa={:.2f}_Cdelta={:.2f}.png".format(Carpeta,ALFA,CDELTA),bbox_inches = "tight")
                plt.close(fig)
        
        
            
            """
            
            # Inicio la figura con los múltiples gráficos de Variación Promeido
            plt.rcParams.update({'font.size': 24})
            fig = plt.figure("ConjuntoHist2D",figsize=(64,36))
            gs = fig.add_gridspec(6,5,hspace=0,wspace=0)
            axs = gs.subplots(sharex=True, sharey=True)
            
            # Estas son las columnas y filas a recorrer para el gráfico
            Columnas = [0,1,2,3,4]*6
            Filas = [i for i in range(0,6) for j in range(0,5)]
            
            # Reinicio la OpiMaxima a 0
            OpiMaxima = 0
            
            for ALFA in Conjunto_Alfa[0:30]:
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                
                    #-----------------------------------------------------------------------------------------------
                        
                    # Acá voy armando el cálculo de la OpiMáxima
                    
                    OpiMaxima = max(max(Opi),OpiMaxima)
                    
            
            for ALFA,fila,columna in zip(Conjunto_Alfa[0:30],Filas,Columnas):
                
                # Armo el array de Opiniones Finales
                OpinionesFinales = np.array([])
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                    
                    # Me voy armando un array con las opiniones finales de los agentes a lo largo
                    # de todas las simulaciones
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                        
                # -------------------------------------------------------------------------------------------------
                
                # Acá voy a armar gráficos conjuntos de los Histogramas 2D cosa de que el etiquetado se realice de mejor forma.
                # Dios quiera que esto salga fácil y rápido.
                
                OpiMaxima = max(np.absolute(OpinionesFinales))
                OpiExtremos = np.array([OpiMaxima,OpiMaxima,-OpiMaxima,OpiMaxima,OpiMaxima,-OpiMaxima,-OpiMaxima,-OpiMaxima])
                Opiniones2D = np.concatenate((OpinionesFinales,OpiExtremos), axis=None)
                

                # Ahora grafico las curvas de distribución de ambas opiniones
                axs[fila,columna].hist2d(Opiniones2D[0::2],Opiniones2D[1::2], bins=(79,79), density=True, cmap=plt.cm.Reds)
                axs[fila,columna].annotate(r"$\alpha=${}".format(ALFA), xy=(0.8,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
                        
            #-------------------------------------------------------------------------------------------
            
            # Acá cierro el gráfico de las Variaciones Promedio. Este gráfico me sirve para ver
            # la varianza del sistema para converger, la cual depende mucho de lo que tarda cada sistema
            # en volverse conexo
            
            plt.savefig("../Imagenes/RedAct/{}/Conjunto_Hist2D_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
            plt.close("ConjuntoHist2D")
        
        #-----------------------------------------------------------------------------------------------------------

        
        
        # Acá me armo los gráficos de Conjunto de Distribución de Opinones

        for CDELTA,icdelta in zip(Conjunto_Cdelta,np.arange(len(Conjunto_Cdelta))):
        
            # Defino los alfas que voy a usar en el gráfico de Distribución de Opiniones
            
            plt.rcParams.update({'font.size': 24})
            
            fig = plt.figure("Distribucion Opiniones",figsize=(64,36))
            
            gs = fig.add_gridspec(4,4, hspace = 0)
            axs = gs.subplots(sharex=True)
            
            # Alfas_Dist = [Conjunto_Alfa[int(indice*math.floor(len(Conjunto_Alfa)/16))] for indice in range(0,16)]
            
            Columnas = [0,1,2,3]*4
            Filas = [i for i in range(0,4) for j in range(0,4)]

            Conjunto_Alfa = list(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA].keys())
            Conjunto_Alfa.sort()
            
            
            # Reinicio la OpiMaxima a 0
            OpiMaxima = 0
            
            for ALFA in Conjunto_Alfa[0:30]:
                
                for nombre,numero in zip(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA],np.arange(len(SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]))):
                
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                        
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                
                    #-----------------------------------------------------------------------------------------------
                        
                    # Acá voy armando el cálculo de la OpiMáxima
                    
                    OpiMaxima = max(max(Opi),OpiMaxima)
                    
                    #-----------------------------------------------------------------------------------------------
            
            
            for ALFA,fila,columna in zip(Conjunto_Alfa[0:16],Filas,Columnas):
                
                OpinionesFinales = np.array([])

                #-------------------------------------------------------------------------------------
                for nombre in SuperDiccionario[Carpeta][TIPO][AGENTES][CDELTA][ALFA]:
                    
                    #----------------------------------------------------------------------------------------------------------------------------------
                    # Acá levanto los datos de los archivos de opiniones. Estos archivos tienen los siguientes datos:
                    # Opinión Inicial del sistema
                    # Variación Promedio
                    # Opinión Final
                    # Grado de Agentes
                    # Actividad de Agentes
                    # Semilla
                    
                    Datos = ldata("{}/{}".format(Conjunto_Direcciones[0],nombre))
                    
                    # Levanto los datos de la Variación Promedio
                    Opi = np.array([float(x) for x in Datos[5][1::]])
                    
                    # Me voy armando un array con las opiniones finales de los agentes a lo largo
                    # de todas las simulaciones
                    
                    OpinionesFinales = np.concatenate((OpinionesFinales,Opi),axis=None)
                    
                    
                        
            # -------------------------------------------------------------------------------------------------
                
                # Acá voy a armar los gráficos de las proyecciones de las opiniones de los agentes. Para eso simplemente
                # voy a tomar los valores de opiniones de los agentes de una simulación, calcularle el histograma
                # con np.histogram y luego graficar eso como líneas.
                
                OpiMaxima = max(np.absolute(OpinionesFinales))
                # if OpiMaxima < 150:
                #     Bordes = np.round(np.linspace(-OpiMaxima,OpiMaxima,22),5)
                # else:
                #     BordeI = np.mean(OpinionesFinales[OpinionesFinales < 0])-30
                #     BordeD = np.mean(OpinionesFinales[OpinionesFinales > 0])+30
                #     BordeMax = max(abs(BordeI),BordeD)
                #     Bordes = np.round(np.linspace(-BordeMax,BordeMax,int(OpiMaxima/50)*2),3)
                # else:
                    # Bordes = np.round(np.linspace(-60,60,40),3)
                Bordes = np.round(np.linspace(-60,60,40),3)
                Bins = (Bordes[1::]+Bordes[0:len(Bordes)-1])/2
                
                Histo,nada = np.histogram(OpinionesFinales,bins=Bordes)
                
                ResultadoEF = EstadoFinal(OpinionesFinales,Histo,Bins)
                # if ALFA == 0:
                #     print("Los valores del histograma de alfa=0 son:")
                #     print(Histo)
                
                # Armo los histogramas correspondientes
                Histo,nada = np.histogram(OpinionesFinales,bins=Bordes,density=True)
                
                # Defino el tema de los colores
                if ResultadoEF == "Consenso":
                    color = "green"
                elif ResultadoEF == "Polarizacion":
                    color = "blue"
                elif ResultadoEF == "Ideologico":
                    color = "red"
                elif ResultadoEF == "RegionTrans":
                    color = "purple"
                
                
                # Ahora grafico las curvas de distribución de ambas opiniones
                axs[fila,columna].plot(Bins,Histo,"-o",color = color,linewidth = 4,markersize = 8, label="Alfa = {}".format(ALFA))
                axs[fila,columna].legend()
                axs[fila,columna].grid()
            
            plt.savefig("../Imagenes/RedAct/{}/Distribucion_opiniones_Cdelta={:.2f}.png".format(Carpeta,CDELTA),bbox_inches = "tight")
#                plt.show()
            plt.close("Distribucion Opiniones")
        
        #--------------------------------------------------------------------------------------------------------
        
        
        # Estos son los parámetros que definen el tamaño del gráfico, tamaño de la letra y nombres de
        # los ejes. Luego de eso guardo la figura y la cierro. Esto es para la figura de
        # TdO.
        
        # Este gráfico me coloca los puntos finales de las opiniones con colores y me 
        # pone un cartelito marcando cuál es el estado final del sistema. Voy a agregar
        # que se dibuje un círculo marcando la región donde se encuentran la mayoría de
        # las opiniones incialmente, puntos negros para las opiniones iniciales y agregar
        # el hecho de que los límites de los gráficos tienen que ser entre -30 y 30
        
        # Armo los valores X e Y que me dibujan el círculo
        Tita = np.linspace(0,2*math.pi,1000)
        Xcirc = 2*math.sqrt(2.5)*np.cos(Tita)
        Ycirc = 2*math.sqrt(2.5)*np.sin(Tita)
        
        
        plt.rcParams.update({'font.size': 18})
        plt.figure("Grafico Opiniones",figsize=(20,15))
        
        # Grafico los puntos iniciales de las opiniones
        for x1,x2 in zip (OpinionesIniciales[0::2],OpinionesIniciales[1::2]):
            plt.plot(x1,x2, "o" ,c = "black", markersize=5, alpha = 0.5)
        
        # Grafico los puntos finales de las opiniones
        for x1,x2 in zip (OpinionesFinales[0::2],OpinionesFinales[1::2]):
            indice = Indice_Color(np.array([x1,x2]),Divisiones)
            plt.plot(x1,x2, "o" ,c = color[indice], markersize=10)
        
        # Grafico el círculo en distancia 2 sigma del cero
        plt.plot(Xcirc,Ycirc,"--",linewidth=5,alpha=0.6)
        
        #            plt.tick_params(left=False,
        #                bottom=False,
        #                labelleft=False,
        #                labelbottom=False)
        plt.xlabel(r"$x^1$")
        plt.ylabel(r"$x^2$")
#                    #            plt.title(r"Trayectoria de las opiniones en el espacio de tópicos para $\alpha$={},cos($\delta$)={} y N={}".format(ALFA,CDELTA,AGENTES))
##                    plt.xlim((xmin,xmax))
##                    plt.ylim((ymin,ymax))
        plt.annotate(r"$\alpha$={},cos($\delta$)={},N={}".format(ALFA,CDELTA,AGENTES), xy=(0.7,0.85),xycoords='axes fraction',fontsize=20,bbox=dict(facecolor='White', alpha=0.7))
        plt.savefig("../Imagenes/RedAct/{}/Grafico_opiniones_alfa={:.3f}_Cdelta={:.2f}_N={}.png".format(Carpeta,ALFA,CDELTA,AGENTES),bbox_inches = "tight")
        plt.close("Grafico Opiniones")
        """

        #------------------------------------------------------------------------------------------------
                
                        
Tiempo()
