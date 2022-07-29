// En este archivo me defino las funciones que inicializan, pero no las declaro

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "general.h"
#include "inicializar.h"


// Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
int GenerarOpi(s_Red var, s_Param par){
	// Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	int i_F,i_C;
	i_F = (int) var.pd_Opi[0];
	i_C = (int) var.pd_Opi[1];
	
	// Inicializo la "matriz" de opiniones de mis agentes.
	for(int i_i=0;i_i<i_F*i_C;i_i++) var.pd_Opi[i_i+2] = par.i_Mopi*(Random()-0.5)*2;
	return 0;
}

// Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
int GenerarAng(s_Red var){
	// Obtengo las dimensiones de la matriz de Superposicion.
	int i_F,i_C;
	i_F = (int) var.pd_Ang[0];
	i_C = (int) var.pd_Ang[1];
	
	// Inicializo la matriz de Superposicion de mi sistema.
	for(int i_i=0; i_i<i_F; i_i++) for(int i_j=0; i_j<i_i; i_j++) var.pd_Ang[i_i*i_C+i_j+2] = 0.5; //*(((-i_i-i_j)%2)*2+1); // Decidí poner 0.5 entre todos los tópicos de mi modelo
	for(int i_i=0; i_i<i_F; i_i++) var.pd_Ang[i_i*i_C+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	for(int i_i=0; i_i<i_F; i_i++) for(int i_j=i_i+1; i_j<i_C; i_j++) var.pd_Ang[i_i*i_C+i_j+2] = var.pd_Ang[i_j*i_C+i_i+2]; // Esta sola línea simetriza la matriz
	return 0;
}


// Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
int GenerarAdy(s_Red var, s_Param par){
	// Obtengo las dimensiones de la matriz de Adyacencia.
	int i_F,i_C;
	i_F = (int) var.pi_Ady[0];
	i_C = (int) var.pi_Ady[1];
	
	// Inicializo la matriz de Adyacencia de mi sistema
	for(int i_i=0; i_i<i_F; i_i++) {
		for(int i_j=0; i_j<i_i; i_j++) {
			// Saco un número random y veo si asigno un 1 o un 0 a esa posición-
			if(Random()<par.f_Pint) var.pi_Ady[i_i*i_C+i_j+2] = 1;
			else var.pi_Ady[i_i*i_C+i_j+2] = 0;
		}
	}
	for(int i_i=0; i_i<i_F; i_i++) for(int i_j=i_i+1; i_j<i_C; i_j++) var.pi_Ady[i_i*i_C+i_j+2] = var.pi_Ady[i_j*i_C+i_i+2]; // Esta sola línea simetriza la matriz
	// for(int i=0; i<par.N; i++) var.Ady[i*par.N+i] = 1; // Esto me pone 1 en toda la diagonal
	return 0;
}

