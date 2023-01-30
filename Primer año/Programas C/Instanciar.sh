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
Arr_Cosenos=(0 0.2 0.4 0.6 0.8 1)
Arr_Kappas=(0.5 0.7 0.9 1.1 1.3 1.5)
Arr_Epsilons=(1.5 1.9 2.3 2.7 3.1 3.5)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {15..19}
		do
			for Coseno in ${Arr_Cosenos[@]}
			do
				for Kappa in ${Arr_Kappas[@]}
				do
					for Epsilon in ${Arr_Epsilons[@]}
					do
						echo Coseno = $Coseno, Kappa = $Kappa, Epsilon = $Epsilon 
						./$1.e $N $Coseno $Kappa $Epsilon $iteracion
					done
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
