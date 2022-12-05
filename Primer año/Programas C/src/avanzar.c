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
double Din_sumatoria(ps_Red ps_var, ps_Param ps_par){
	// Defino las variables locales de mi función
	double d_opiniones_superpuestas = 0; // Es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	double d_resultado; // Es el valor que returnea la función
	double d_exponente; // Exponente de la función exponencial
	double d_denominador; // Denominador de la función logística
	
	// Obtengo el tamaño de columnas de mis dos matrices
	int i_Co,i_Cs;
	i_Co = (int) ps_var->pd_Opi[1]; // Número de columnas en la matriz de opiniones
	i_Cs = (int) ps_var->pd_Ang[1]; // Número de columnas en la matriz de superposición
	
	// Calculo el producto de la matriz con el vector.
	for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += ps_var->pd_Ang[ps_var->i_topico*i_Cs+i_p+2]*ps_var->pd_Opi[ps_var->i_agente2*i_Co+i_p+2];
	
	// Calculo el exponente de la exponencial
	d_exponente = ps_par->d_alfa*d_opiniones_superpuestas - ps_par->d_chi;
	
	// Calculo el denominador de la logística
	d_denominador = 1+exp(-d_exponente);
	
	// Ahora que tengo todo, calculo el resultado y returneo
	d_resultado = 1/d_denominador;
	return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Din_interes(ps_Red ps_var, ps_Param ps_par){
	// Defino las variables locales de mi función.
	double d_resultado; // d_resultado es lo que voy a returnear.
	double d_sumatoria=0; // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	int i_grado = 0;  // i_grado es el grado del agente 1
	
	
	// Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// Aprovecho este for y también calculo el grado del agente
	// La sumatoria es sobre todos los agentes conectados en la red de adyacencia
	for(ps_var->i_agente2=0; ps_var->i_agente2<ps_par->i_N; ps_var->i_agente2++){
		if(ps_var->pi_Ady[ps_var->i_agente*ps_var->pi_Ady[1]+ps_var->i_agente2+2] == 1){
			i_grado += 1; // Sumo un uno por cada agente con el cual el agente 1 está conectado
			d_sumatoria += Din_sumatoria(ps_var,ps_par); // Sumo los valores de las funciones logísticas
		}
	}
	
	
	// Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	int i_C = (int) ps_var->pd_Opi[1];
	d_resultado = ps_var->pd_Opi[ps_var->i_agente*i_C+ps_var->i_topico+2] *(-1-ps_var->pd_Sat[ps_var->i_agente*i_C+ps_var->i_topico+2]) + d_sumatoria/i_grado;
	return d_resultado;
}


double Din_saturacion(ps_Red ps_var, ps_Param ps_par){
	// Defino las variables locales de mi función
	double d_resultado; // d_resultado es lo que voy a returnear.
	int i_C = (int) ps_var->pd_Opi[1]; // Tamaño de columnas de mi vector de opiniones
	
	// Hago la cuenta de la ecuación dinámica
	d_resultado = ps_var->pd_Opi[ps_var->i_agente*i_C+ps_var->i_topico+2] - ps_par->d_gamma * ps_var->pd_Sat[ps_var->i_agente*i_C+ps_var->i_topico+2];
	
	return d_resultado;
}