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

Arr_Agentes=(1000)
Arr_Alfas=(5 5.5 6 6.5 7 7.5 8)
Arr_Kappas=(1 1.5 2 2.5 3 3.5 4)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..20}
		do
			for Kappa in ${Arr_Kappas[@]}
			do
				for Alfa in ${Arr_Alfas[@]}
				do
					echo Kappa = $Kappa, Alfa = $Alfa
					./$1.e $N $Kappa $Alfa $iteracion
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
