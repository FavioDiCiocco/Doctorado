//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<stdbool.h>
#include <unistd.h> // Esto lo uso para poner un sleep

void Clasificacion(double *puntero, double *distribucion, double kappa, int bines);


int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tprin, tfin, semilla;
	time(&tprin);
	semilla = time(NULL);
	srand(semilla);
	int tardanza;
	double kappa=10;
	int bines=10;
	double* puntero;
	puntero = (double*) calloc(10+2,sizeof(double));
	double* distribucion;
	distribucion = (double*) calloc(bines*bines+2,sizeof(double));
	
	// Armo mi puntero de opiniones para clasificar
	*puntero = 5;
	*(puntero+1) = 2;
	*(puntero+2) = 7;
	*(puntero+3) = 5;
	*(puntero+4) = -2;
	*(puntero+5) = -6;
	*(puntero+6) = 3.5;
	*(puntero+7) = -8.001;
	*(puntero+8) = 10;
	*(puntero+9) = -10;
	*(puntero+10) = -3.99;
	*(puntero+11) = 3;
	
	// Armo mi puntero para guardar las distribuciones
	*distribucion = 10;
	*(distribucion+1) = 10;
	
	Clasificacion(puntero, distribucion, kappa, bines);
	
	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++) printf("%lf ", *(distribucion+i*10+j+2));
		printf("\n");
	}
	printf("\n");
	
	time(&tfin);
	tardanza = tfin - tprin;
	printf("Tarde %d segundos en terminar\n",tardanza);
	
	return 0;
}


//########################################################################################
//########################################################################################

void Clasificacion(double *puntero, double *distribucion, double kappa, int bines){
	
	// Defino las variables y vectores que voy a necesitar
	int fila,columna;
	int Fd = (int) *distribucion; // Este es el número de filas de la matriz de histograma
	int Cd = (int) *(distribucion+1); // Este es el número de columnas de la matriz de histograma
	int Fo = (int) *(puntero); // Este es el número de filas de la matriz de opiniones
	int Co = (int) *(puntero+1); // Este es el número de columnas de la matriz de opiniones
	double ancho = (double) 2/bines; // Este es el ancho de cada cajita en la que separo el espacio de opiniones.
	
	// Normalizo mi histograma y la corro a la región [0,2]
	for(int i =0; i< Fo*Co; i++) *(puntero+i+2) = *(puntero+i+2) / kappa + 1;
	
	for(int i = 0; i< Fo*Co; i++) printf("%lf ", *(puntero+i+2) );
	printf("\n");
	
	// Hago el conteo de agentes en cada una de las cajitas
	for(int agente = 0; agente < Fo; agente++ ){
		columna = fmin(floor(*(puntero+agente*Co+2)/ancho), bines-1);
		fila = fmin(floor(*(puntero+agente*Co+1+2)/ancho), bines-1);
		printf("La opinión sobre el ancho en la columna es: %lf \n", *(puntero+agente*Co+2)/ancho);
		printf("La opinión sobre el ancho en la fila es: %lf \n", *(puntero+agente*Co+1+2)/ancho);
		printf("La opinion %lf corresponde a la columna %d \n", *(puntero+agente*Co+2), columna);
		printf("La opinion %lf corresponde a la fila %d \n", *(puntero+agente*Co+1+2), fila);
		*(distribucion+fila*Cd+columna+2) += 1;
	}
	
	// Resuelto el conteo, ahora lo normalizo
	for(int i =0; i<Fd*Cd; i++) *(distribucion+i+2) = *(distribucion+i+2)/Fo;
}