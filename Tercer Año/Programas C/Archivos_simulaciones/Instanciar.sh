#!/bin/bash
# Este programa lo voy a usar para poder correr los programas de C
# y que sean programas separados, de manera de que me tire el tiempo que tardó
# en cada uno. Porque sino tengo que andar mirando los tiempos en los archivos.

# make clean
# make all

# echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
# read decision

echo "El ID del script es $$"

# Voy a armar los arrays de las variables que voy a barrer.

# Armo el array de Agentes. Este en principio es un único valor
Arr_Agentes=(10000)

# Armo el array de valores de Beta. Primero lo hago vacío
# y después lo lleno con la lista de valores a recorrer
Arr_Beta=()

for val in {0..15}
do
	cuenta=`echo $val*0.1 | bc -l`
	Arr_Beta+=( $cuenta )
done


# Armo el array de valores de Cosdelta. Primero lo hago
# vacío y después lo lleno con la lista de valores a recorrer.
Arr_Cosdelta=()

for val in {0..17}
do
	cuenta=`echo $val*0.02+0.16 | bc -l`
	Arr_Cosdelta+=( $cuenta )
done

# Armo el array del Kappa, lo necesito para unas
# cuentas particulares
Arr_Kappas=(10)

#for val in {0..38}
#do
#	cuenta=`echo $val*0.5+1 | bc -l`
#	Arr_Kappas+=( $cuenta )
#done


if [ -z $decision ]
then
	for N in ${Arr_Agentes[@]}
	do
		iteracion=$2
		while [ $iteracion -le $3 ]
		do
			for Kappa in ${Arr_Kappas[@]}
			do
				for Beta in ${Arr_Beta[@]}
				do
					for Cdelta in ${Arr_Cosdelta[@]}
					do
						echo Beta = $Beta, Cdelta = $Cdelta
						./$1.e $Beta $Cdelta $iteracion
					done
				done
			done
		echo Termine la iteracion $iteracion
		((iteracion++))
		done
	done
fi


