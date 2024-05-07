// En este archivo me defino las funciones que inicializan, pero no las declaro

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "inicializar.h"
#include "avanzar.h"


// Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
void GenerarOpi(puntero_Matrices red, double kappa){
	
	// Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	int F,C;
	F = (int) red->Opi[0];
	C = (int) red->Opi[1];
	
	// Inicializo la "matriz" de opiniones de mis agentes.
	for(int i=0; i<F*C; i++) red->Opi[i+2] = (Random()-0.5)*2*kappa;
}

// Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
void GenerarAng(puntero_Matrices red, puntero_Parametros param){
	
	// Obtengo las dimensiones de la matriz de Superposicion.
	int F,C;
	F = (int) red->Ang[0];
	C = (int) red->Ang[1];
	
	// Inicializo la matriz de Superposicion de mi sistema.
	for(int i=0; i<F; i++) for(int j=0; j<i; j++) red->Ang[ i*C+j+2 ] = param->Cosd; //
	for(int i=0; i<F; i++) red->Ang[ i*C+i+2 ] = 1; // Esto me pone 1 en toda la diagonal
	for(int i=0; i<F; i++) for(int j=i+1; j<C; j++) red->Ang[ i*C+j+2] = red->Ang[ j*C+i+2 ]; // Esta sola línea simetriza la matriz
}

/*
// ##################################################################################
// Esta función me arma una matriz de Adyacencia de una red totalmente conectada
void GenerarAdy_Conectada(puntero_Matrices red){
	
	// Obtengo las dimensiones de la matriz de Adyacencia.
	int F,C;
	F = red->Ady[0];
	C = red->Ady[1];
	
	// Escribo la matriz de Adyacencia
	for(int i=1; i<F; i++) for(int j=0; j<i; j++) red->Ady[ i*C+j+2 ] = 1;  // Esto me pone 1 debajo de la diagonal
	for(int i=0; i<F; i++) for(int j=i+1; j<C; j++) red->Ady[ i*C+j+2 ] = red->Ady[ j*C+i+2 ]; // Esta sola línea simetriza la matriz
}
// ##################################################################################
*/

// Esta función me arma la matriz de Separación de agentes.
void Generar_Separacion(puntero_Matrices red, puntero_Parametros param){
	
	// Primero tomo las filas y columnas de mi matriz de Separacion
	int F,C;
	F = red->Sep[0];
	C = red->Sep[1];
	
	// int contador=0;
	
	// Esta es una matriz simétrica, construyo la mitad de arriba y después la copio abajo.
	
	for(red->agente=0; red->agente<F; red->agente++){
		for(int j=0; j<red->Ady[red->agente+2][1]; j++){
			red->agente_vecino = red->Ady[red->agente+2][j+2];
			if(red->agente < red->agente_vecino) red->Sep[ red->agente*C +red->agente_vecino+2 ] = Numerador_homofilia(red, param);
			// contador++;
			// printf("%d\t%d\n",red->agente,red->agente_vecino);
		}
	}
	
	// printf("Realicé %d iteraciones\n",contador);
	
	for(int i=0; i<F; i++) for(int j=i+1; j<C; j++) red->Sep[ j*C +i+2 ] = red->Sep[ i*C +j+2 ]; // Esta sola línea simetriza la matriz
}

// Esta función es la que lee un archivo y me arma la matriz de Adyacencia
int Lectura_Adyacencia(int *vec, FILE *archivo){
	
	// Defino los enteros que voy a usar para leer el archivo y escribir sobre el vector.	
	int indice = 2;
	int salida = 0;
	
	// Leo la matriz de Adyacencia del archivo y la guardo en el vector de Adyacencia.
	while( fscanf(archivo,"%d", &salida ) != EOF && indice < *vec * *(vec+1)+2){
		*(vec+indice) = salida;
		indice++;
	}
	
	// Aviso si hubo algún problema.
	if( fscanf(archivo,"%d", &salida) != EOF ){
		printf("La matriz del archivo es mas grande que tu vector\n");
		return 1;
	}
	if( fscanf(archivo,"%d", &salida) == EOF && indice < *vec * *(vec+1)+2){
		printf("La matriz del archivo es mas chica que el vector\n");
		return 1;
	}
	
	return 0;
}

// Esta función es la que lee un archivo y me arma la lista de vecinos en el puntero de punteros de pi_Adyacencia
int Lectura_Adyacencia_Ejes(puntero_Matrices red, FILE *archivo){
	//##########################################################################################
	
	// Defino las variables que voy a usar para leer el archivo y escribir sobre el vector.
	int N, L; // N es el número de agentes, L es el número de enlaces.
	while( fscanf(archivo,"%d %d",&N,&L ) != EOF) break;
	
	int n, m; // n y m son los agentes 
	
	// Construyo un vector que tenga los grados de todos los agentes
	int* grado;
	grado = (int*) calloc(N+2, sizeof(int) );
	*grado = 1;
	*( grado+1 ) = N;
	
	// Construyo un vector auxiliar que tenga los grados de todos los agentes
	int* grado_aux;
	grado_aux = (int*) calloc(N+2, sizeof(int));
	*grado_aux = 1;
	*( grado_aux+1 ) = N;
	
	//#########################################################################################
	
	// Construyo la lista de vecinos
	
	// Leo tantas lineas como enlaces haya en la red
	for(int i=0; i<L; i++){ 
		while( fscanf(archivo,"%d %d", &n, &m ) != EOF )  break;
        *( grado+n+2 ) += 1;
		*( grado+m+2 ) += 1;
    }
	
	// En cada componente de Adyacencia se declaran tantas componentes como enlaces tenga, no más
	for(int i=0; i<N; i++){
		red->Ady[i+2] = (int*) calloc( *(grado+i+2)+2, sizeof(int));
		red->Ady[i+2][0] = 1;
		red->Ady[i+2][1] = *( grado+i+2 );
	}
	
	if( fscanf(archivo,"%d %d", &n, &m) == EOF ){
		rewind(archivo);
		while( fscanf(archivo, "%d %d", &N, &L) !=EOF ) break;
	}
	else{
		printf("Leí mal la matriz");
		return 1;
	}

    for(int i=0; i<L; i++){
		while( fscanf(archivo,"%d %d", &n, &m) != EOF ) break;
        red->Ady[n+2][ *(grado_aux+n+2) +2 ] = m;
		red->Ady[m+2][ *(grado_aux+m+2) +2 ] = n;
        *(grado_aux+n+2) += 1;
		*(grado_aux+m+2) += 1;
    }

    free(grado_aux);
	
	//#########################################################################################
	
	// Construyo la lista de vecinos complementaria
	
	int l,vecino;

    for(int i=0; i<N; i++){
		red->Ady_vecinos[i+2] = (int*) calloc( *(grado+i+2)+2, sizeof(int));
		red->Ady_vecinos[i+2][0] = 1;
		red->Ady_vecinos[i+2][1] = *( grado+i+2 );

        for(int j=0; j<*( grado+i+2 ); j++){
            vecino = red->Ady[i+2][j+2];
            for(l=0; l< *(grado+vecino+2); l++)
                if( red->Ady[vecino+2][l+2] == i ) break;
            red->Ady_vecinos[i+2][j+2] = l;
        }
    }
	
	free(grado);
	
	return 0;
}



