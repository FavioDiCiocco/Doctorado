#!/bin/bash
# Voy a usar este programa para correr los demas
# Así me ahorro que me agarre algún quilombo cada vez que los modifico

chmod 755 $1
dos2unix ./$1
./$1 $2 $3 $4 $5 $6 $7