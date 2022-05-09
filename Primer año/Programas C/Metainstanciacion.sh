#!/bin/bash
# Este programa lo voy a usar para correr más ràpidamente los archivos
# de Instanciar de manera de en un sólo paso generar en solo paso 
# las corridas de todos los programas que quiero correr


make clean
make all

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

i=0

if [ -z $decision ]
then
	while [ $i -lt $1 ]
	do
		let inicio=i*5
		let final=(i+1)*5-1
		nohup ./Instanciar.sh Opiniones $inicio $final > "salidaRA$i.out" & 
		echo "Mande a correr Alfa entre $inicio y $final" 
		((i++))
	done
	let inicio=i*5
	let final=(i+1)*5
	nohup ./Instanciar.sh Opiniones $inicio $final > "salidaRA$i.out" &
	echo "Mande a correr Alfa entre $inicio y $final"

fi


