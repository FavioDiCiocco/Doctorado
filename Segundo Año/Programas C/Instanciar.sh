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

Arr_Agentes=(10000)
Arr_Beta=(0.5 1 1.5)
Arr_CosD=(0 0.5 1)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..1}
		do
			for Beta in ${Arr_Beta[@]}
			do
				for CosD in ${Arr_CosD[@]}
				do
					echo Beta = $Beta, CosD = $CosD
					./$1.e $N $Beta $CosD $iteracion
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
