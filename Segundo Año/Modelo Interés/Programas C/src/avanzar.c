// En este archivo defino todas las funciones de que manipulan datos del sistema, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "inicializar.h"
#include "avanzar.h"


// Esta función resuelve un término de los de la sumatoria. Lo que va a hacer es el producto de la matriz
// de superposición
// Más algunos otros parámetros relevantes.

double Dinamica_sumatoria(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables locales de mi función
	double opiniones_superpuestas = 0; // Es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	double resultado; // Es el valor que returnea la función
	double exponente; // Exponente de la función exponencial
	double denominador; // Denominador de la función logística
	
	// Obtengo el tamaño de columnas de mis dos matrices
	int Co,Cs;
	Co = (int) red->Opi[1]; // Número de columnas en la matriz de opiniones
	Cs = (int) red->Ang[1]; // Número de columnas en la matriz de superposición
	
	// Calculo el producto de la matriz con el vector.
	for(int p=0; p<Cs; p++) opiniones_superpuestas += red->Ang[ red->topico*Cs+p+2 ]*red->Opi[ red->agente_vecino*Co+p+2 ];
	
	// Calculo el exponente de la exponencial
	exponente = param->alfa*opiniones_superpuestas - param->epsilon;
	
	// Calculo el denominador de la logística
	denominador = 1+exp( -exponente );
	
	// Ahora que tengo todo, calculo el resultado y returneo
	resultado = 1.0/denominador;
	return resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Dinamica_interes(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables locales de mi función.
	double resultado; // resultado es lo que voy a returnear.
	double sumatoria = 0; // sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	int grado = red->Ady[red->agente+2][1]; 
	int C = (int) red->Opi[1];
	
	// Aprovecho este for y también calculo el grado del agente
	// La sumatoria es sobre todos los agentes conectados en la red de adyacencia
	for(int i=0; i < grado; i++){
		red->agente_vecino = red->Ady[red->agente+2][i+2];
		sumatoria += red->Exp[red->agente_vecino*C +red->topico +2]; // Sumo los valores de las funciones logísticas
	}
	
	resultado = -red->Opi[ red->agente*C +red->topico+2 ] + param->kappa * (sumatoria/grado);
	
	return resultado;
}

