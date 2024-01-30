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
Arr_Beta=(0.5 0.6 0.7 0.8 0.9 1)
Arr_CosD=(0)
Arr_Kappas=(10)


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..5}
		do
			for Kappa in ${Arr_Kappas[@]}
			do
				for Beta in ${Arr_Beta[@]}
				do
					for CosD in ${Arr_CosD[@]}
					do
						echo Kappa=$Kappa, Beta = $Beta, CosD = $CosD
						./$1.e $N $Kappa $Beta $CosD $iteracion
					done
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
