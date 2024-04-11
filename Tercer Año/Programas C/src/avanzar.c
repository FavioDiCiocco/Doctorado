// En este archivo defino todas las funciones de que manipulan datos del sistema, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "inicializar.h"
#include "avanzar.h"


// Voy a separar la ecuación dinámica en dos funciones. Una que tome el término con la tanh pero que se aplique a 
// un sólo agente externo. Luego esa función la voy a usar en la otra dentro de una sumatoria.
// La gran pregunta es: ¿Me conviene hacer un juego como el que hicimos en Física Computacional, donde armábamos
// una tabla con valores pre calculados y a los valores intermedios los interpolábamos? Por ahora, probemos sin
// eso. Si resulta que tarda mucho el programa, consideraré agregarlo.


// Esta función resuelve un término de los de la sumatoria. Lo que va a hacer es el producto de la matriz
// de superposición y luego calcular el valor de la función logística asociada usando ese producto en su exponente.
// Más algunos otros parámetros relevantes.
double Dinamica_sumatoria(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables locales de mi función
	double opiniones_superpuestas = 0; // Es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	double resultado; // Es el valor que returnea la función
	
	// Obtengo el tamaño de columnas de mis dos matrices
	int Co,Cs;
	Co = (int) red->Opi[1]; // Número de columnas en la matriz de opiniones
	Cs = (int) red->Ang[1]; // Número de columnas en la matriz de superposición
	
	// Calculo el producto de la matriz con el vector.
	for(int p=0; p<Cs; p++) opiniones_superpuestas += red->Ang[ red->topico*Cs+p+2 ]*red->Opi[ red->agente_vecino*Co+p+2 ];
	
	// Ahora que tengo todo, calculo el resultado y returneo
	if(opiniones_superpuestas > 0) resultado = 1;
	if(opiniones_superpuestas < 0) resultado = -1;
	if(opiniones_superpuestas == 0) resultado = 0;
	
	return resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Dinamica_opiniones(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables locales de mi función.
	double resultado; // resultado es lo que voy a returnear.
	double sumatoria = 0; // sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	double denominador = 0; // denominador es el denominador de la normalización de los pesos por homofilia.
	
	// Obtengo los valores de la cantidad de columnas de varias de mis matrices
	int Co, Cs, grado;
	Co = (int) red->Opi[1]; // Número de columnas en la matriz de opiniones
	Cs = (int) red->Sep[1]; // Número de columnas en la matriz de separaciones
	grado = (int) red->Ady[ red->agente+2 ][1]; // Número de columnas en la matriz de adyacencia
	
	// Calculo el denominador de los pesos haciendo la sumatoria de los pesos de la matriz de Separacion.
	// No considero si el agente1 tiene conexión con el agente2, porque de no tener conexión, la matriz de separación tiene un cero.
	for(int i=0; i<grado; i++){
		red->agente_vecino = red->Ady[ red->agente+2 ][i+2];
		denominador += red->Sep[ red->agente*Cs +red->agente_vecino+2 ];
	}
	
	// Calculo la sumatoria de la ecuación diferencial
	// La sumatoria es sobre todos los agentes conectados en la red de adyacencia
	for(int j=0; j<grado; j++){
		red->agente_vecino = red->Ady[ red->agente+2 ][j+2];
		sumatoria += red->Sep[ red->agente*Cs +red->agente_vecino+2 ] * red->Exp[ red->agente_vecino*Co +red->topico +2 ]; // Sumo los valores de las funciones logísticas
	}
	
	resultado = -red->Opi[ red->agente*Co +red->topico+2 ] + param->kappa * (sumatoria / denominador);
	return resultado;
}


// Al implementar la matriz de Separación, esta función perdió utilidad.
/*
// Esta función calcula el denominador que normaliza los pesos en la ecuación dinámica.
// Estos pesos son los que determinan el valor que el agente i le da a la opinión del agente j según la homofilia.
double Normalizacion_homofilia(ps_Red ps_variable, ps_Param ps_parametro){
	// Defino las variables locales de mi función.
	double d_sumatoria = 0; // d_sumatoria es lo que iré sumando de los términos del denominador y después returneo
	double d_distancia = 0; // d_distancia es la distancia en el espacio de opiniones entre el agente i y el agente j
	
	int i_Fo,i_Co,i_Ca;
	i_Fo = (int) ps_variable->pd_Opiniones[0]; // Número de filas en la matriz de opiniones
	i_Co = (int) ps_variable->pd_Opiniones[1]; // Número de columnas en la matriz de opiniones
	i_Ca = (int) ps_variable->pi_Adyacencia[1]; // Número de columnas en la matriz de adyacencia
	
	
	// Armo un puntero a un vector en el cuál pondre la diferencia entre las opiniones del
	// agente i y el agente j.
	double *pd_Vector_Diferencia;
	pd_Vector_Diferencia = (double*) malloc((2+i_Co)*sizeof(double));
	*pd_Vector_Diferencia = 1;
	*(pd_Vector_Diferencia+1) = i_Co;
	
	// Hago la resta entre la opinión de mi agente i y el resto de los agentes
	
	for(register int i_agentej = 0; i_agentej<i_Fo; i_agentej++){
		if(ps_variable->pi_Adyacencia[ps_variable->i_agente*i_Ca+i_agentej+2] == 1){
			// Armo el vector que apunta del agente i al agente j en el espacio de opiniones
			for(register int i_topic=0; i_topic<i_Co; i_topic++){
				*(pd_Vector_Diferencia+i_topic+2) = ps_variable->pd_Opiniones[ps_variable->i_agente*i_Co+i_topic+2]-ps_variable->pd_Opiniones[i_agentej*i_Co+i_topic+2];
			}
			d_distancia = Norma_No_Ortogonal_d(pd_Vector_Diferencia, ps_variable->pd_Angulos);
			
			// Agrego el término del agente j a la sumatoria del denominador
			d_sumatoria += pow(d_distancia+ps_parametro->d_delta,-ps_parametro->d_beta);
		}
	}
	
	free(pd_Vector_Diferencia);
	
	return d_sumatoria;
}
*/

// Esta función calcula el numerador de los pesos en la ecuación dinámica.
// Estos pesos son los que determinan el valor que el agente i le da a la opinión del agente j según la homofilia.
double Numerador_homofilia(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables locales de mi función.
	double resultado = 0; // resultado es lo que returneo
	double distancia = 0; // distancia es la distancia en el espacio de opiniones entre el agente i y el agente j
	
	int Co;
	Co = (int) red->Opi[1]; // Número de columnas en la matriz de opiniones
	
	// Armo un puntero a un vector en el cuál pondre la diferencia entre las opiniones del
	// agente i y el agente j.
	double *Vec_Dif;
	Vec_Dif = (double*) malloc( ( 2+Co )*sizeof(double));
	*Vec_Dif = 1;
	*(Vec_Dif+1) = Co;
	
	// Armo el vector que apunta del agente i al agente vecino en el espacio de opiniones
	for(int topic=0; topic<Co; topic++) *(Vec_Dif +topic+2) = red->Opi[ red->agente*Co +topic+2 ] - red->Opi[ red->agente_vecino*Co +topic+2 ];
	
	// Calculo la distancia entre las opiniones de los agentes
	distancia = Norma_No_Ortogonal_d(Vec_Dif, red->Ang);
	
	// Agrego el término del agente j a la sumatoria del denominador
	resultado += pow(distancia + param->delta, -param->beta);
	
	free(Vec_Dif);
	
	return resultado;
}

