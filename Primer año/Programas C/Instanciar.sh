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

Arr_Agentes=(2 3)
Arr_Alfas=(0 0.2 0.4 1)
Arr_Cdeltas=(0 0.5 1)
Arr_Decaimientos=(0.1 0.2 0.3)

if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..10}
		do
			# Alfa=$2
			# while [ $Alfa -le $3 ]
			for Alfa in ${Arr_Alfas[@]}
			do
				for Cdelta in ${Arr_Cdeltas[@]} 
				do
					for Decaimiento in ${Arr_Decaimientos[@]}
					do
						echo Alfa = $Alfa, Cdelta = $Cdelta, Decaimiento = $Decaimiento
						./$1.e $N $Alfa $Cdelta $Decaimiento $iteracion
					done
				done
			# ((Alfa++))
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi


