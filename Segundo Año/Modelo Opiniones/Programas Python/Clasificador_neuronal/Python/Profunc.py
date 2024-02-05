#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 10:04:40 2022

@author: favio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import math
import time
import os
import funciones as func

# Importo todas las librerías que voy a usar en el programa. Estas son las que
# uso en los tres programas, por lo que no debería necesitar nada más.

t0 = time.time()

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

#-----------------------------------------------------------------------------------------------

# Esta función carga una imagen y la transforma en una matriz
# en escala de grises. Para que cargue la imagen con todos los canales
# de colores, tengo que sacar el convert("L")

def load_image(file_path):
    # Open the image file
    img = Image.open(file_path).convert("L")

    # Convert the image to a NumPy array
    img_array = np.array(img)

    return img_array

#-----------------------------------------------------------------------------------------------

# Esta función carga todas las imágenes en un directorio y a partir de ahí
# levanta las matrices de archivos

def load_images(directory):
    X = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            img_array = load_image(file_path)
            
            # Flatten and normalize grayscale values
            img_array_flat = img_array.flatten() / 255.0
            
            X.append(img_array_flat)

    return np.array(X,dtype="float")

#-----------------------------------------------------------------------------------------------

"""
# Transformo estas cosas en paths. Espero que acostumbrarme a esto valga la pena
Direccion = Path("../{}".format("Imagenes"))
carpeta = Path("Imagenes")

# Recorro las carpetas con datos
CarpCheck=[[root,files] for root,dirs,files in os.walk(Direccion)]

# Me armo una lista con los nombres de todos los archivos con datos.
# Coloco al inicio de la lista el path de la carpeta en la que se encuentran los datos

Archivos_Datos = [nombre for nombre in CarpCheck[0][1]]

# Df_archivos = pd.DataFrame({"nombre": Archivos_Datos})


# Example usage
file_path = Direccion / Archivos_Datos[0]
image_matrix = load_image(file_path)
"""

X = load_images("../Imagenes")


func.Tiempo(t0)
