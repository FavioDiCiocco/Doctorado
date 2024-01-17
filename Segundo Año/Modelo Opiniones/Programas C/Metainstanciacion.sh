#!/bin/bash
# Este programa lo voy a usar para correr más ràpidamente los archivos
# de Instanciar de manera de en un sólo paso generar en solo paso 
# las corridas de todos los programas que quiero correr


# Por si acaso te lo anoto esto acá, este programa recibe como primer
# input de línea de comando la cantidad de hilos-1 que planeas usar.
# Es decir que si le pones un 5 como primer número, va a usar 6
# hilos. Vos calculá si seis hilos cubren el rango que queres.

make clean
make all

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

base=0
i=0
# gm=10
paso=6

if [ -z $decision ]
then
	while [ $i -lt $1 ]
	do
		let inicio=i*$paso+base
		let final=(i+1)*$paso-1+base
		nohup ./Instanciar.sh Opiniones $inicio $final > "salidaCambio_Parametro_2D$i.out" & 
		echo "Estoy corriendo las iteraciones entre $inicio y $final" 
		((i++))
	done
	let inicio=i*$paso+base
	let final=(i+1)*$paso+base
	nohup ./Instanciar.sh Opiniones $inicio $final > "salidaCambio_Parametro_2D$i.out" &
	echo "Estoy corriendo las iteraciones entre $inicio y $final"

fi


