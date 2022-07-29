// En este archivo defino todas las funciones de que manipulan datos del sistema, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "general.h"
#include "inicializar.h"
#include "avanzar.h"


// Voy a separar la ecuación dinámica en dos funciones. Una que tome el término con la tanh pero que se aplique a 
// un sólo agente externo. Luego esa función la voy a usar en la otra dentro de una sumatoria.
// La gran pregunta es: ¿Me conviene hacer un juego como el que hicimos en Física Computacional, donde armábamos
// una tabla con valores pre calculados y a los valores intermedios los interpolábamos? Por ahora, probemos sin
// eso. Si resulta que tarda mucho el programa, consideraré agregarlo.


// Esta función resuelve un término de los de la sumatoria.
double Din1(s_Red var, s_Param par,int i_j){
	
	// Defino las variables locales de mi función. d_resultados es el return. 
	// d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	double d_resultado,d_opiniones_superpuestas = 0;
	
	// Obtengo el tamaño de columnas de mis tres matrices
	int i_Co,i_Cs,i_Ca;
	i_Co = (int) var.pd_Opi[1];
	i_Cs = (int) var.pd_Ang[1];
	i_Ca = var.pi_Ady[1];
	
	for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += var.pd_Ang[var.i_topico*i_Cs+i_p+2]*var.pd_Opi[i_j*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	d_resultado = var.pi_Ady[var.i_agente*i_Ca+i_j+2]*tanh(par.f_alfa*d_opiniones_superpuestas); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Din2(s_Red var, s_Param par){
	
	// Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	double d_resultado,d_sumatoria=0;
	
	// Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	for(register int i_j=0; i_j<var.i_agente; i_j++) d_sumatoria += Din1(var,par,i_j);
	for(register int i_j=var.i_agente+1; i_j<par.i_N; i_j++) d_sumatoria += Din1(var,par,i_j);
	
	// Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	int i_Columnas = (int) var.pd_Opi[1];
	d_resultado = -var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2]+par.f_K*d_sumatoria;
	return d_resultado;
}

