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
Arr_Lambdas=(0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.06 0.07 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 1)
# Arr_Alfas=(1 2 3 4 5 6 7 8)
# Arr_Umbrales=(0.5 1 1.5 2 2.5 3)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..30}
		do
			for Lambda in ${Arr_Lambdas[@]}
			do
				echo Lambda = $Lambda
				./$1.e $N $Lambda $iteracion
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
