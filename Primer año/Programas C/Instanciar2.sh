#!/bin/bash
# Este programa lo voy a usar para poder correr los programas de C
# y que sean programas separados, de manera de que me tire el tiempo que tardó
# en cada uno. Porque sino tengo que andar mirando los tiempos en los archivos.

make clean
make all

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

redes="Barabasi ErdosRenyi RandomR"

if [ -z $decision ]
then
	for Red in $redes
	do
		for N in 1000
		do
			for Alfa in {0..20}
			do
				for Cdelta in {-18..-12..2}
				do
					for Gm in {4..16..4}
					do
						echo Alfa = $Alfa, Cdelta = $Cdelta, GM = $Gm, Red = $Red
						for iteracion in {0..60}
						do
							./$1.e $N $Red $Alfa $Cdelta $Gm $iteracion
						done
					done
				done
			done
		done
	done
	
	for Red in $redes
	do
		for N in 1000
		do
			for Alfa in {30..100..10}
			do
				for Cdelta in {-18..-12..2}
				do
					for Gm in {4..16..4}
					do
						echo Alfa = $Alfa, Cdelta = $Cdelta, GM = $Gm
						for iteracion in {0..60}
						do
							./$1.e $N $Red $Alfa $Cdelta $Gm $iteracion
						done
					done
				done
			done
		done
	done
	
	# ./Mover.sh

fi


