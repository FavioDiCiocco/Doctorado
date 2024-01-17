#!/bin/bash
# Este programa lo voy a usar para probar algunas ideas de cómo
# usar Bash

# Con esto tengo cómo hacer cuentas con números float en Bash
# La idea es usar un comando llamado bc. Entonces, siguiendo
# el formato de abajo, se puede realizar una operación, al parecer
# se tiene que hacer un echo de eso ¿para que le llegue un string?
# al bc y este hace la cuenta. Luego, para asignar el resultado a una
# variable es que se usa los apóstrofes invertidos

Array=()

for val in {0..10}
do
	for divisor in {10..15}
	do
		cuenta=`echo $val/$divisor | bc -l`
		Array+=( $cuenta )
	done
done

# Puedo definir un array o lista a partir de ponerle nombre, signo igual y paréntesis vacíos.
# Luego, puedo simplemente apendear  elementos al array usando += y creo
# que tengo que dejar espacio entre los paréntesis. Luego, para revisar los
# valores dentro del array, tengo que usar corchetes. Ahora, si dentro del
# corchete pongo un número, tomo un único elemento del array. En cambio,
# si uso el @, eso me devuelve todos los valores del array. El tema es que para
# eso necesito entonces usar las llaves para encerrar mi variable para que Bash
# comprenda que la variable es el nombre y los corchetes son la parte del array que quiero.
# Sin las llaves, bash pensaría que los corchetes son parte del string

for numero in ${Array[@]}
do
	echo $numero
done

