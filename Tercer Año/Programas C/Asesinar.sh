#!/bin/bash
# Este programa lo voy a usar para matar todos los programas que haya mandado a correr juntos.

Lista=$( head -1 salida* | cut -f 6 -d ' ' )

for ID in $Lista
do
	kill $ID
done



