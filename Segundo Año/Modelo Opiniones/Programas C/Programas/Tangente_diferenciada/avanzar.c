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
double Dinamica_sumatoria(ps_Red ps_variable, ps_Param ps_parametro){
	// Defino las variables locales de mi función
	double d_resultado; // Es el valor que returnea la función
	
	// Obtengo el tamaño de columnas de mis dos matrices
	int i_Co;
	i_Co = (int) ps_variable->pd_Opiniones[1]; // Número de columnas en la matriz de opiniones
		
	// Ahora que tengo todo, calculo el resultado y returneo
	d_resultado = tanh(ps_variable->pd_Opiniones[ps_variable->i_agente2*i_Co+ps_variable->i_topico+2]);
	return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Dinamica_opiniones(ps_Red ps_variable, ps_Param ps_parametro){
	// Defino las variables locales de mi función.
	double d_resultado; // d_resultado es lo que voy a returnear.
	double d_sumatoria = 0; // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	double d_numerador; // d_denominador es el numerador de los pesos por homofilia.
	double d_denominador; // d_denominador es el denominador de la normalización de los pesos por homofilia.
	
	//######################################################################
	// Voy a remover el cos(delta) de la tangente hiperbolica. Pero al mismo tiempo quiero que 
	// el cálculo de la distancia en los pesos siga realizándose correctamente. Por tanto no voy a
	// deshacer la matriz de superposición, sino que voy a quitar el for en la función de Dinámica_sumatoria
	//######################################################################
	
	d_denominador = Normalizacion_homofilia(ps_variable, ps_parametro);
	
	// Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// Aprovecho este for y también calculo el grado del agente
	// La sumatoria es sobre todos los agentes conectados en la red de adyacencia
	for(ps_variable->i_agente2=0; ps_variable->i_agente2<ps_parametro->i_N; ps_variable->i_agente2++){
		if(ps_variable->pi_Adyacencia[ps_variable->i_agente*ps_variable->pi_Adyacencia[1]+ps_variable->i_agente2+2] == 1){
			
			d_numerador = Numerador_homofilia(ps_variable,ps_parametro);
			d_sumatoria += d_numerador * Dinamica_sumatoria(ps_variable,ps_parametro); // Sumo los valores de las funciones logísticas
		}
	}
	
	
	// Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	int i_C = (int) ps_variable->pd_Opiniones[1];
	// d_resultado = -ps_var->pd_Opi[ps_var->i_agente*i_C+ps_var->i_topico+2] * ps_var->pd_Sat[ps_var->i_agente*i_C+ps_var->i_topico+2] + d_sumatoria/i_grado; // Término con saturación
	d_resultado = -ps_variable->pd_Opiniones[ps_variable->i_agente*i_C+ps_variable->i_topico+2] + ps_parametro->d_kappa*(d_sumatoria/d_denominador);
	return d_resultado;
}


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


// Esta función calcula el numerador de los pesos en la ecuación dinámica.
// Estos pesos son los que determinan el valor que el agente i le da a la opinión del agente j según la homofilia.
double Numerador_homofilia(ps_Red ps_variable, ps_Param ps_parametro){
	// Defino las variables locales de mi función.
	double d_resultado = 0; // d_resultado es lo que returneo
	double d_distancia = 0; // d_distancia es la distancia en el espacio de opiniones entre el agente i y el agente j
	
	int i_Co;
	i_Co = (int) ps_variable->pd_Opiniones[1]; // Número de columnas en la matriz de opiniones
	
	// Armo un puntero a un vector en el cuál pondre la diferencia entre las opiniones del
	// agente i y el agente j.
	double *pd_Vector_Diferencia;
	pd_Vector_Diferencia = (double*) malloc((2+i_Co)*sizeof(double));
	*pd_Vector_Diferencia = 1;
	*(pd_Vector_Diferencia+1) = i_Co;
	
	// Armo el vector que apunta del agente i al agente 2 en el espacio de opiniones
	for(register int i_topic=0; i_topic<i_Co; i_topic++){
		*(pd_Vector_Diferencia+i_topic+2) = ps_variable->pd_Opiniones[ps_variable->i_agente*i_Co+i_topic+2]-ps_variable->pd_Opiniones[ps_variable->i_agente2*i_Co+i_topic+2];
	}
	// Calculo la distancia entre las opiniones de los agentes
	d_distancia = Norma_No_Ortogonal_d(pd_Vector_Diferencia, ps_variable->pd_Angulos);
	
	// Agrego el término del agente j a la sumatoria del denominador
	d_resultado += pow(d_distancia+ps_parametro->d_delta,-ps_parametro->d_beta);
	
	free(pd_Vector_Diferencia);
	
	return d_resultado;
}

// double Dinamica_saturacion(ps_Red ps_variable, ps_Param ps_parametro){
	// // Defino las variables locales de mi función
	// double d_resultado; // d_resultado es lo que voy a returnear.
	// int i_C = (int) ps_variable->pd_Opiniones[1]; // Tamaño de columnas de mi vector de opiniones
	
	// // Hago la cuenta de la ecuación dinámica
	// d_resultado = ps_variable->pd_Opiniones[ps_variable->i_agente*i_C+ps_variable->i_topico+2] - ps_parametro->d_lambda * ps_variable->pd_Saturacion[ps_variable->i_agente*i_C+ps_variable->i_topico+2];
	
	// return d_resultado;
// }