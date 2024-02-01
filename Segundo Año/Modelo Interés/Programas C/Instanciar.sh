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
Arr_Alfas=(4)

############################################################

Arr_Epsilon=()

for val in {0..40}
do
	cuenta=`echo $val*0.05+1.5 | bc -l`
	Arr_Epsilon+=( $cuenta )
done

############################################################


Arr_Kappas=()
for val in {0..30}
do
	cuenta=`echo $val*0.05+0.5 | bc -l`
	Arr_Kappas+=( $cuenta )
done



if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		for iteracion in {0..5}
		do
			for Kappa in ${Arr_Kappas[@]}
			do
				for Alfa in ${Arr_Alfas[@]}
				do
					for Epsilon in ${Arr_Epsilon[@]}
					do
						echo Kappa=$Kappa, Epsilon = $Epsilon, Alfa = $Alfa
						./$1.e $N $Kappa $Epsilon $Alfa $iteracion
					done
				done
			done
			echo Complete $iteracion simulaciones totales
		done
	done

fi
