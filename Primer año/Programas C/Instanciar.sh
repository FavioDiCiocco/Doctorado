#!/bin/bash
# Este programa lo voy a usar para poder correr los programas de C
# y que sean programas separados, de manera de que me tire el tiempo que tardó
# en cada uno. Porque sino tengo que andar mirando los tiempos en los archivos.

make clean
make all

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

echo "El ID del script es $$"

if [ -z $decision ]
then
	for N in 200
	do
		for iteracion in {0..4}
		do
			# Alfa=$2
			# while [ $Alfa -le $3 ]
			for Alfa in {0..20}
			do
				for Cdelta in {-5..10} 
				do
					for Campo in 0
					do
						for Decaimiento in 1
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


