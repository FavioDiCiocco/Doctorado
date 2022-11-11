// En este archivo me defino las funciones que inicializan, pero no las declaro

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "inicializar.h"


// Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
int GenerarOpi(ps_Red ps_var, ps_Param ps_par){
	// Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	int i_F,i_C;
	i_F = (int) ps_var->pd_Opi[0];
	i_C = (int) ps_var->pd_Opi[1];
	// double d_varianza;
	// d_varianza = sqrt(2.5);
	
	// Inicializo la "matriz" de opiniones de mis agentes.
	for(register int i_i=0;i_i<i_F*i_C;i_i++) ps_var->pd_Opi[i_i+2] = Random(); // Gaussiana(0,d_varianza);
	return 0;
}

// Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
int GenerarAng(ps_Red ps_var, ps_Param ps_par){
	// Obtengo las dimensiones de la matriz de Superposicion.
	int i_F,i_C;
	i_F = (int) ps_var->pd_Ang[0];
	i_C = (int) ps_var->pd_Ang[1];
	
	// Inicializo la matriz de Superposicion de mi sistema.
	for(register int i_i=0; i_i<i_F; i_i++) for(register int i_j=0; i_j<i_i; i_j++) ps_var->pd_Ang[i_i*i_C+i_j+2] = ps_par->d_Cosangulo; //
	for(register int i_i=0; i_i<i_F; i_i++) ps_var->pd_Ang[i_i*i_C+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	for(register int i_i=0; i_i<i_F; i_i++) for(register int i_j=i_i+1; i_j<i_C; i_j++) ps_var->pd_Ang[i_i*i_C+i_j+2] = ps_var->pd_Ang[i_j*i_C+i_i+2]; // Esta sola línea simetriza la matriz
	return 0;
}

// Esta función me arma una matriz de Adyacencia de una red totalmente conectada
int GenerarAdy_Conectada(ps_Red ps_var, ps_Param ps_par){
	// Obtengo las dimensiones de la matriz de Adyacencia.
	int i_F,i_C;
	i_F = ps_var->pi_Ady[0];
	i_C = ps_var->pi_Ady[1];
	
	// Escribo la matriz de Adyacencia
	for(register int i_i=1; i_i<i_F; i_i++) for(register int i_j=0; i_j<i_i; i_j++) ps_var->pi_Ady[i_i*i_C+i_j+2] = 1;  // Esto me pone 1 debajo de la diagonal
	for(register int i_i=0; i_i<i_F; i_i++) for(register int i_j=i_i+1; i_j<i_C; i_j++) ps_var->pi_Ady[i_i*i_C+i_j+2] = ps_var->pi_Ady[i_j*i_C+i_i+2]; // Esta sola línea simetriza la matriz
	return 0;
}


// // Esta función es la que lee un archivo y me arma la matriz de Adyacencia
int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo){
	// Defino los enteros que voy a usar para leer el archivo y escribir sobre el vector.	
	int i_indice = 2;
	int i_salida = 0;
	
	// Leo la matriz de Adyacencia del archivo y la guardo en el vector de Adyacencia.
	while(fscanf(pa_archivo,"%d",&i_salida) != EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		*(pi_vec+i_indice) = i_salida;
		i_indice++;
	}
	
	// Aviso si hubo algún problema.
	if(fscanf(pa_archivo,"%d",&i_salida) != EOF){
		printf("La matriz del archivo es mas grande que tu vector\n");
		return 1;
	}
	if(fscanf(pa_archivo,"%d",&i_salida) == EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		printf("La matriz del archivo es mas chica que el vector\n");
		return 1;
	}
	
	return 0;
}

// Me saco estas funciones de encima, total por ahora no las voy a usar, voy a trabajar con redes estáticas
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/*
// Programemos la función que debería devolver un valor de actividad
int Actividad(double* pd_vec, double d_epsilon, double d_potencia){
	// Defino las variables que voy a necesitar que son la probabilidad, la probabilidad mínima, La F del Epsilon
	double d_Feps;
	
	// Defino acá la cantidad de filas y de columnas de mi vector de double donde voy a guardar las actividades
	int i_F,i_C;
	i_F = (int) *pd_vec;
	i_C = (int) *(pd_vec+1);
	
	// Hago unas cuentas para tener de antemano
	d_Feps = pow(d_epsilon,d_potencia+1); // Esto lo hago para no rehacer esta cuenta mil veces en el camino
	
	for(register int i_i=0; i_i<i_F*i_C;i_i++){
		*(pd_vec+i_i+2) = pow((1-d_Feps)*Random()+d_Feps,1/(d_potencia+1));
	}
	
	return 0;
}





// Esta función va a recibir a la matriz de adyacencia y la va a armar según la actividad de los agentes
// Creo que voy a cambiar esto en la función haciendo que directamente reciba los punteros de struct de Red
// y de Parametros
int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos){
	// Primero armo las variables que voy a necesitar, como el tamaño de mis vectores
	int i_F;
	i_F = ps_red->pi_Ady[0];

	// Ahora reviso todos los agentes, los intento activar y si se activa lo conecto con m agentes.
	for(ps_red->i_agente=0; ps_red->i_agente<i_F; ps_red->i_agente++) if(Random()<ps_red->pd_Act[ps_red->i_agente+2]) Conectar_agentes(ps_red,ps_datos);
	
	return 0;
}



// Esta función recibe la matriz de Adyacencia y el agente, y lo conecta con m agentes.
int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos){
	// Defino las variables del tamaño de la matriz y otras que voy a usar
	
	// Obtengo los valores de filas y columnas de mi matriz de Adyacencia
	int i_F = ps_red->pi_Ady[0];
	int i_C = ps_red->pi_Ady[1];
	
	int i_indice = 0; // Uso esto para recorrer el vector de agentes libres
	int i_libres = 0; // Esto me indica cuántos agentes tengo "libres", es decir disponibles para conectarme
	
	// Estas son las variables de double que uso para cálculos y para obtener probabilidades.
	double d_distancia = 0; // Este es el valor escalar de distancia entre dos opiniones
	double d_probacumulada = 0; // Esta es la probabilidad acumulada de los agentes
	double d_proba = 0; // Esta es la probabilidad que lanzo una sola vez
	double d_normalizacionP = 0; // Este es el factor de normalización que es la suma de todas las distancias
	
	//###################################################################################################################
	
	// Acá defino e inicializo mis cuatro vectores.
	
	// Armo un vector que voy a usar para samplear elementos y lo inicializo
	int* pi_agentes_libres; // Defino el puntero al vector
	pi_agentes_libres = (int*) malloc((i_C+2)*sizeof(int)); // Malloqueo el espacio del vector
	*pi_agentes_libres = 1; // Defino el número de filas
	*(pi_agentes_libres+1) = i_C; // Defino el número de columnas
	for(register int i_i=0; i_i<i_C; i_i++) *(pi_agentes_libres+i_i+2) = 0; // Inicializo el resto del vector
	
	// Armo un vector para ver la diferencia entre opiniones. Este vector lo voy a usar primero para calcular la
	// distancia entre agentes y con eso hallar el factor de normalización y sus probabilidades de conexión
	double* pd_DifOpi; // Defino el puntero al vector
	pd_DifOpi = (double*) malloc((ps_datos->i_T+2)*sizeof(double)); // Malloqueo el espacio del vector
	*pd_DifOpi = 1; // Defino el número de filas
	*(pd_DifOpi+1) = ps_datos->i_T; // Defino el número de columnas
	for(register int i_i=0; i_i<ps_datos->i_T; i_i++) *(pd_DifOpi+i_i+2) = 0; // Inicializo el resto del vector
	
	// Armo el vector de Distbeta, el cual me guarda la distancia de cada agente con el agente ps_red->i_agente elevada a la menos beta.
	double* pd_distbeta; // Defino el puntero al vector
	pd_distbeta = (double*) malloc((ps_datos->i_N+2)*sizeof(double)); // Malloqueo el espacio del vector
	*pd_distbeta = 1; // Defino el número de filas
	*(pd_distbeta+1) = ps_datos->i_N; // Defino el número de columnas
	for(register int i_i=0; i_i<ps_datos->i_N; i_i++) *(pd_distbeta+i_i+2) = 0; // Inicializo el resto del vector
	
	// El vector probabilidades toma el valor de normalización y a cada una de estas distancias las normaliza para obtener la probabilidad de interacción por homofilia.
	double* pd_probabilidades; // Defino el puntero al vector
	pd_probabilidades = (double*) malloc((ps_datos->i_N+2)*sizeof(double)); // Malloqueo el espacio del vector
	*pd_probabilidades = 1; // Defino el número de filas
	*(pd_probabilidades+1) = ps_datos->i_N; // Defino el número de columnas
	for(register int i_i=0; i_i<ps_datos->i_N; i_i++) *(pd_probabilidades+i_i+2) = 0; // Inicializo el resto del vector
	
	//###################################################################################################################
	
	// Acá hago los cálculos preparatorios de las distancias y las probabilidades, así como revisar cuáles agentes
	// están libres para ser conectados al agente principal.
	
	// Reviso la fila de la matriz de adyacencia del agente $i_agente, y me guardo los números de los agentes
	// que no tienen conexión con $i_agente. Lo separo en dos fors, porque la idea es saltear el caso de i_i = ps_red->i_agente
	for(register int i_i=0; i_i<ps_red->i_agente; i_i++){
		if(ps_red->pi_Ady[ps_red->i_agente*i_C+i_i+2]==0){ // Esto mira que el agente $i_agente no esté conectado con el agente $i_i
			*(pi_agentes_libres+i_libres+2) = i_i; // En ese caso agrego al agente i_i a la lista de agentes libres
			i_libres++; // Avanzo el índice que marca la cantidad de agentes libres
		}
	}
	
	// Acá reviso los agentes de la fila de la matriz de adyacencia que están a la dercecha del agente $i_agente
	for(register int i_i=ps_red->i_agente+1; i_i<i_C; i_i++){
		if(ps_red->pi_Ady[ps_red->i_agente*i_C+i_i+2]==0){ // Esto mira que el agente $i_agente no esté conectado con el agente $i_i
			*(pi_agentes_libres+i_libres+2) = i_i; // En ese caso agrego al agente i_i a la lista de agentes libres
			i_libres++; // Avanzo el índice que marca la cantidad de agentes libres
		}
	}
	
	
	// Calculo todas las distancias y el factor de normalizacion. Empiezo recorriendo a todos los agentes libres
	for(register int i_i=0; i_i<i_libres; i_i++){
		i_indice = *(pi_agentes_libres+i_i+2); // Este es el número de agente libre
		for(register int i_j=0; i_j<ps_datos->i_T; i_j++) *(pd_DifOpi+i_j+2) = ps_red->pd_Opi[ps_red->i_agente*ps_datos->i_T+i_j+2]-ps_red->pd_Opi[i_indice*ps_datos->i_T+i_j+2];
		// Voy a armar el calculo de las distancias de vectores no ortogonales acá como una única cuenta
		// La suma es de la primer componente de la diferencia de vectores al cuadrado, más la segunda componente al cuadrado más el doble del producto
		// de la primer componente por la segunda componente por coseno(delta). Y a todo eso le tomo raíz
		d_distancia = sqrt(pow(*(pd_DifOpi+2),2)+pow(*(pd_DifOpi+1+2),2)+2*(*(pd_DifOpi+2))*(*(pd_DifOpi+1+2))*ps_datos->d_Cosangulo);
		if(d_distancia>0) *(pd_distbeta+i_indice+2) = pow(d_distancia,-ps_datos->d_beta); // Si la distancia es mayor a cero, calculo el valor de la distancia elevada a -beta
		else *(pd_distbeta+i_indice+2) = 0; // Si la distancia es cero, pongo un cero para que no haya problemas en la cuenta.
	}
	for(register int i_i=0; i_i<ps_datos->i_N; i_i++) d_normalizacionP += *(pd_distbeta+i_i+2); // Esta línea me calcula el factor de normalización
	
	// Ahora calculo las probabilidades. Vuelvo a recorrer a todos los agentes libres
	for(register int i_i=0; i_i<i_libres; i_i++){
		i_indice = *(pi_agentes_libres+i_i+2); // Este es el número de agente libre
		if(*(pd_distbeta+i_indice+2)>0) *(pd_probabilidades+i_indice+2) = *(pd_distbeta+i_indice+2)/d_normalizacionP;
		else *(pd_probabilidades+i_indice+2) = 1; // Si la distancia es cero, pongo un 1 en la probabilidad
	}
	
	
	//###################################################################################################################
	
	// Ahora sampleo m agentes de esta lista, si tengo una cantidad de agentes libres mayor a m. Sino simplemente conecto a todos los libres
	if(i_libres>ps_datos->i_m){
		int i_contador = 0; // El contador es para ir marcando cuántos agentes conecté hasta llegar a m.
		while(i_contador<ps_datos->i_m){
			i_indice = 0; // Este es el iterable de los agentes libres
			d_proba = Random(); // Establezco el valor de probabilidad a superar para armar la conexión
			d_probacumulada = *(pd_probabilidades+2); // Establezco probacumulada como la proba del primer agente en la lista.
			// Defino cuál es el agente que se conecta
			while(d_proba>d_probacumulada){
				i_indice++; // Actualizo mi iterable
				d_probacumulada += *(pd_probabilidades+i_indice+2); // Sumo la proba del siguiente agente a la acumulada
			}
			// Hago la conexión
			ps_red->pi_Ady[ps_red->i_agente*i_C+i_indice+2] = 1; // Marco la conexión con el sujeto 
			ps_red->pi_Ady[i_indice*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar la posición simétrica en la matriz de Adyacencia
			// Saco al agente del grupo de agentes posibles para conectarse al
			// poner su distancia como cero y su probabilidad como cero.
			*(pd_distbeta+i_indice+2) = 0;
			*(pd_probabilidades+i_indice+2) = 0;
			// Calculo el nuevo factor de normalización y las nuevas probabilidades, ignorando aquellos sujetos cuya distancia es cero.
			// Esto ignora tanto a los ya conectados como a los que tienen distancia cero.
			d_normalizacionP = 0;
			for(register int i_i=0; i_i<ps_datos->i_N; i_i++) d_normalizacionP += *(pd_distbeta+i_i+2);
			for(register int i_i=0; i_i<ps_datos->i_N; i_i++) if(*(pd_distbeta+i_i+2)!=0) *(pd_probabilidades+i_i+2) = *(pd_distbeta+i_i+2)/d_normalizacionP;
			// Avanzo el contador para marcar que hice una conexión
			i_contador++;
		}
	}
	else for(register int i_i=0; i_i<i_libres; i_i++){
		ps_red->pi_Ady[ps_red->i_agente*i_C+*(pi_agentes_libres+i_i+2)+2] = 1; // Marco la conexión con el sujeto 
		ps_red->pi_Ady[*(pi_agentes_libres+i_i+2)*i_C+ps_red->i_agente+2] = 1; // Esto se encarga de marcar la posición simétrica en la matriz de Adyacencia
	}
	
	//###################################################################################################################
	
	// Libero los punteros utilizados
	
	free(pi_agentes_libres);
	free(pd_DifOpi);
	free(pd_distbeta);
	free(pd_probabilidades);
	return 0;
}
*/
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


