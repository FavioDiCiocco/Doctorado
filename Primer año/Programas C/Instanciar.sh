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

Arr_Alfas=(0 2)
Arr_Cdeltas=(-1 0 1)
Arr_Decaimientos=(1)

if [ -z $decision ]
then
	for N in 500
	do
		for iteracion in {0..29}
		do
			# Alfa=$2
			# while [ $Alfa -le $3 ]
			for Alfa in ${Arr_Alfas[@]}
			do
				for Cdelta in ${Arr_Cdeltas[@]} 
				do
					for Campo in 0
					do
						for Decaimiento in ${Arr_Decaimientos[@]}
						do
							echo Alfa = $Alfa, Cdelta = $Cdelta, campo = $Campo, Decaimiento = $Decaimiento
							./$1.e $N $Alfa $Cdelta $iteracion $Campo $Decaimiento
						done
					done
				done
			# ((Alfa++))
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi


