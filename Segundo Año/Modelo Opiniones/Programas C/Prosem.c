//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<stdbool.h>
#include <unistd.h> // Esto lo uso para poner un sleep

double Varianza(double *pd_vector,double d_denominador);

// Voy a probar si la función que armé calcula bien la norma de un vector en un espacio no ortogonal


int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tt_prin,tt_fin,semilla;
	time(&tt_prin);
	semilla = time(NULL);
	srand(semilla);
	int i_tardanza;

	// Me armo un vector con valores 3 y 0, cosa de que el promedio sea 1.5 y la varianza sea 2.25.
	
	double* pd_prueba;
	pd_prueba = (double*) malloc((100+2)*sizeof(double));
	
	*pd_prueba = 1;
	*(pd_prueba+1) = 100;
	
	for(register int i_i=0; i_i<25;i_i++) *(pd_prueba+i_i+2) = 0;
	for(register int i_i=25; i_i<50;i_i++) *(pd_prueba+i_i+2) = 3;
	for(register int i_i=50; i_i<75;i_i++) *(pd_prueba+i_i+2) = 0;
	for(register int i_i=75; i_i<100;i_i++) *(pd_prueba+i_i+2) = 3;
	
	// Calculo la varianza sin normalizar
	printf("Hago los cálculos sin normalizar.\n");
	Varianza(pd_prueba,1);
	
	// Calculo la varianza normalizada
	printf("Hago los cálculos normalizados.\n");
	Varianza(pd_prueba,3);
	
	// Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(pd_Superposicion);
	// free(pd_Vector);
	
	free(pd_prueba);
	time(&tt_fin);
	i_tardanza = tt_fin-tt_prin;
	printf("Tarde %d segundos en terminar\n",i_tardanza);
	
	return 0;
}


//########################################################################################
//########################################################################################

// Función de cálculo de la varianza
double Varianza(double *pd_vector,double d_denominador){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pd_vector; // Filas del vector
	i_C = *(pd_vector+1); // Columnas del vector
	
	double d_norma = 1/d_denominador; // Este es el valor que normaliza mi vector.
	// Si quiero la varianza sin normalizar, elijo denominador = 1.
	double d_varianza = 0; // Acá voy guardando el acumulado de la suma de la varianza
	double d_promedio = 0; // Este es el promedio de los datos normalizados

	// Calculo el promedio
	for(register int i_i=0; i_i<i_F*i_C; i_i++) d_promedio += *(pd_vector+i_i+2) * d_norma;
	d_promedio = d_promedio/(i_F*i_C);
	
	printf("El promedio de mi vector es: %.2lf\n",d_promedio);
	
	// Hago el cálculo de la varianza
	for(register int i_i=0; i_i<i_F*i_C; i_i++) d_varianza += (*(pd_vector+i_i+2) * d_norma-d_promedio) * (*(pd_vector+i_i+2) * d_norma-d_promedio);
	d_varianza = d_varianza/(i_F*i_C);
	
	printf("La varianza de mi vector es: %.2lf\n",d_varianza);
	
	return d_varianza;
}