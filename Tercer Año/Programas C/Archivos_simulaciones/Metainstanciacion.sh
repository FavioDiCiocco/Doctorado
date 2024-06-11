#!/bin/bash
# Este programa lo voy a usar para correr más ràpidamente los archivos
# de Instanciar de manera de en un sólo paso generar en solo paso 
# las corridas de todos los programas que quiero correr


# Por si acaso te lo anoto esto acá, este programa recibe como primer
# input de línea de comando la cantidad de hilos que planeas usar.
# Es decir que si le pones un 5 como primer número, va a usar 6
# hilos. Vos calculá si seis hilos cubren el rango que queres.

# El completados es por si en algún momento tuve que cortar las simulaciones
# y tuve que volver a arrancar más tarde. 

# make clean
# make all

./Compilar.sh opinion

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision

base=20
i=0
# gm=10
paso=2
completados=0

if [ -z $decision ]
then
	while [ $i -lt $1 ]
	do
		let inicio=i*$paso+base+completados
		let final=(i+1)*$paso-1+base
		nohup ./Instanciar.sh opinion $inicio $final > "Final_BetaCosd$i.out" & 
		echo "Estoy corriendo las iteraciones entre $inicio y $final" 
		((i++))
	done
	#let inicio=i*$paso+base+completados
	#let final=(i+1)*$paso+base
	#nohup ./Instanciar.sh Opiniones $inicio $final > "salida_Tiempo_Hilos$i.out" &
	#echo "Estoy corriendo las iteraciones entre $inicio y $final"

fi


