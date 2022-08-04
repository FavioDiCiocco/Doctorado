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


// Esta función resuelve un término de los de la sumatoria.
double Din1(ps_Red ps_var, ps_Param ps_par){
	// Defino las variables locales de mi función. d_resultados es el return. 
	// d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	double d_resultado,d_opiniones_superpuestas = 0;
	
	// Obtengo el tamaño de columnas de mis tres matrices
	int i_Co,i_Cs;
	i_Co = (int) ps_var->pd_Opi[1];
	i_Cs = (int) ps_var->pd_Ang[1];
	// i_Ca = (int) ps_var->pi_Ady[1];
	
	for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += ps_var->pd_Ang[ps_var->i_topico*i_Cs+i_p+2]*ps_var->pd_Opi[ps_var->i_agente2*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	d_resultado = log(d_opiniones_superpuestas+1); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
}

// Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
double Din2(ps_Red ps_var, ps_Param ps_par){
	// Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	double d_resultado;
	
	// Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	int i_C = (int) ps_var->pd_Opi[1];
	d_resultado = -ps_par->d_mu*ps_var->pd_Opi[ps_var->i_agente*i_C+ps_var->i_topico+2]+(ps_par->f_K*ps_par->f_alfa*Din1(ps_var,ps_par));
	
	// Esta parte del código no la uso en una interacción de a pares
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// La sumatoria es sobre todos los agentes conectados en la red de adyacencia
	// for(ps_var->i_agente2=0; ps_var->i_agente2<ps_par->i_N; ps_var->i_agente2++) if(ps_var->pi_Ady[ps_var->i_agente*ps_var->pi_Ady[1]+ps_var->i_agente2+2] == 1) d_sumatoria += Din1(ps_var,ps_par);
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	return d_resultado;
}

// Esta función me itera todo el sistema. Está buena para simplemente reemplazarla en el main.
int Iteracion(double *pd_sistema,ps_Red ps_var, ps_Param ps_par, double (*pf_Dinamica)(ps_Red ps_var, ps_Param ps_par) ){
	// Voy a recorrer todos los agentes, de manera de evolucionarlos a todos
	for(ps_var->i_agente=0; ps_var->i_agente<ps_par->i_N; ps_var->i_agente++){
		do{
			// Elijo un segundo agente con el que interactuar.
			ps_var->i_agente2 = rand()%ps_par->i_N;
		}
		// Este agente tiene que ser distinto al primero
		while(ps_var->i_agente==ps_var->i_agente2);
		printf("El agente %d interactuó con el agente %d\n",ps_var->i_agente,ps_var->i_agente2);
		// Recorro todos los tópicos, para poder así evolucionar todas mis variables
		for(ps_var->i_topico=0; ps_var->i_topico<ps_par->i_T; ps_var->i_topico++){
			// En la posición del agente y del tópico correspondiente guardo el valor dado por la evolución.
			// El RK4 actual es tal que la matriz toma la matriz Opi, le hace una copia para tener el estado inicial y
			// luego va variando al vector Opi para calcular las nuevas pendientes. Cuando tiene todas las pendientes, usa
			// la copia inicial para reescribir Opi y que salga como entró. Luego usa las pendientes para calcular el valor de la 
			// opinión en el siguiente paso y eso es lo que guardo en el puntero OpiPosterior
			ps_var->pd_OpiPosterior[ps_var->i_agente*ps_par->i_T+ps_var->i_topico+2] = RK4(pd_sistema, ps_var, ps_par,pf_Dinamica);
		}
	}
	// Ahora que tengo todo el sistema evolucionado en OpiPosterior, lo paso a Opi para que la Opinión esté evolucionada
	for(register int i_j=0; i_j<ps_par->i_N*ps_par->i_T; i_j++) ps_var->pd_Opi[i_j+2] = ps_var->pd_OpiPosterior[i_j+2];
	return 0;
}

