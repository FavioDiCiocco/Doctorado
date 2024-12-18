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

# Armo el array de valores de Beta. Primero lo hago vacío
# y después lo lleno con la lista de valores a recorrer
Arr_Beta=()

for val in {0..20}
do
	cuenta=`echo 0.5+$val*0.05 | bc -l`
	Arr_Beta+=( $cuenta )
done


# Armo el array de valores de Cosdelta. Primero lo hago
# vacío y después lo lleno con la lista de valores a recorrer.
#Arr_Cosdelta=()

# for val in {0..24}
# do
#	cuenta=`echo 0.01+$val*0.02 | bc -l`
#	Arr_Cosdelta+=( $cuenta )
# done
Arr_Cosdelta=(1)

#for val in {0..19}
#do
#	cuenta=`echo $val*0.5+0.5 | bc -l`
#	Arr_Kappas+=( $cuenta )
#done


iteracion=56
while [ $iteracion -le 59 ]
do
	for Beta in ${Arr_Beta[@]}
	do
		for Cdelta in ${Arr_Cosdelta[@]}
		do
			echo Beta = $Beta, Cosd = $Cdelta
			./$1.e $Beta $Cdelta $iteracion
		done
	done
echo Termine la iteracion $iteracion
((iteracion++))
done

