//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<stdbool.h>
#include <unistd.h> // Esto lo uso para poner un sleep



int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tprin, tfin, semilla;
	time(&tprin);
	semilla = time(NULL);
	srand(semilla);
	int tardanza;
	double* puntero;
	puntero = (double*) calloc(5,sizeof(double));
	int c;
	
	
	
	// Primero abro el archivo
	char TextArchivo[355];
	sprintf(TextArchivo,"Opiniones_N=1000_kappa=10.0_beta=0.50_cosd=0.00_Iter=0.file");
	FILE *archivo = fopen(TextArchivo,"r"); // Con esto abro mi archivo y dirijo el puntero a él.
		
	// Salteo 5 filas
	for(int i=0; i<5; i++) while ((c = fgetc(archivo)) != EOF && c != '\n'); // Esto saltea una fila
	
	// Leo los primeros cinco números y los printeo
	printf("Estos son los primeros cinco números del archivo: \n");
	for(int i=0; i<5; i++) if(fscanf(archivo,"%lf", puntero+i ) != EOF) printf("%lf ", *(puntero+i));
	printf("\n");
	
	printf("Printeo de nuevo los números, para ver que están guardados correctamente\n");
	for(int i=0; i<5; i++) printf("%lf ", *(puntero+i));
	printf("\n");
	
	time(&tfin);
	tardanza = tfin - tprin;
	printf("Tarde %d segundos en terminar\n",tardanza);
	
	return 0;
}


//########################################################################################
//########################################################################################