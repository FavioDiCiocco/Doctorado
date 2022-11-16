#!/bin/bash
# Este programa lo voy a usar para poder correr los programas de C
# y que sean programas separados, de manera de que me tire el tiempo que tardó
# en cada uno. Porque sino tengo que andar mirando los tiempos en los archivos.

make clean
make all

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

echo "El ID del script es $$"

# Voy a Hardcodear algunos Arrays

Arr_Agentes=(2)
# Arr_Alfas=(0.5 1 1.5 2)
Arr_Umbrales=(0.5 1 1.5 2 2.5 3)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..30}
		do
			for Alfa in {1..8}
			do
				for Umbral in ${Arr_Umbrales[@]}
				do
					echo Alfa = $Alfa, Umbral = $Umbral
					./$1.e $N $Alfa $Umbral $iteracion
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
