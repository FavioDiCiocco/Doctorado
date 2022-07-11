#!/bin/bash
# Este programa lo voy a usar para poder compilar programas y no tener que memorizar lo del gcc y cosas.

gcc -Wall -o3 -o $1.e $1.c -lm

echo Apreta enter para correr, sino primero apreta alguna letra y despues enter
read decision
if [ -z $decision ]
then
	./$1.e $2 $3 $4 $5 $6
fi
	