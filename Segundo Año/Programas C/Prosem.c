//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<stdbool.h>
#include <unistd.h> // Esto lo uso para poner un sleep

int Visualizar_d(double *pd_vector);
double Norma_No_Ortogonal_d(double *pd_Vector, double *pd_Superposicion);

// Voy a probar si la función que armé calcula bien la norma de un vector en un espacio no ortogonal


int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tt_prin,tt_fin,semilla;
	time(&tt_prin);
	semilla = time(NULL);
	srand(semilla);
	int i_tardanza;
	
	
	
	//################################################################################################################################
	
	// double d_resultado = 0;
	
	// // Armo una matriz de superposición
	// double *pd_Superposicion;
	// pd_Superposicion = (double*) malloc((4+2)*sizeof(double));
	// *pd_Superposicion = 2;
	// *(pd_Superposicion+1) = 2;
	
	// // Mis ejes están a 60 grados, por lo que el cos(delta) = 0.5
	// *(pd_Superposicion+0+2) = 1; // Elementos de la diagonal
	// *(pd_Superposicion+1+2) = 0.5; // Elementos fuera de la diagonal
	// *(pd_Superposicion+2+2) = 0.5; // Elementos fuera de la diagonal
	// *(pd_Superposicion+3+2) = 1; // Elementos de la diagonal
	
	// // Me construyo un vector que es la diferencia entre dos vectores
	// double *pd_Vector;
	// pd_Vector = (double*) malloc((2+2)*sizeof(double));
	// *pd_Vector = 1;
	// *(pd_Vector+1) = 2;
	
	// // Supongo el caso en que el vector diferencia es (1,1)
	// *(pd_Vector+0+2) = 0;
	// *(pd_Vector+1+2) = 1;
	
	// d_resultado = Norma_No_Ortogonal_d(pd_Vector, pd_Superposicion);
	
	// printf("El resultado final es: \n");
	// printf("%.3f \n", d_resultado);
	
	//################################################################################################################################
	
	// Aprovecho para probar una cosa más con el tema de las condiciones en los ifs o whiles
	// bool a = true;
	// bool b = false;
	// bool c = true;
	
	// if(a) printf("Sólo a es verdadero \n");
	// if(a && b) printf("Este mensaje no debería salir \n");
	// if(a && c) printf("a y c son verdaderos \n");
	// if(a && !b && c) printf("a y c son verdaderos, b es falso \n");
	
	// if(a || b) printf("a o b son verdaderos \n");
	
	
	//################################################################################################################################

	printf("La semilla inicial es: %d\n",(int) semilla);
	
	sleep(5);
	semilla = time(NULL);
	
	printf("La semilla final es: %d\n",(int) semilla);
	
	
	
	// Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(pd_Superposicion);
	// free(pd_Vector);
	
	time(&tt_fin);
	i_tardanza = tt_fin-tt_prin;
	printf("Tarde %d segundos en terminar\n",i_tardanza);
	
	return 0;
}


//########################################################################################
//########################################################################################

// Esta función es para observar los vectores double
int Visualizar_d(double *pd_vector){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pd_vector;
	i_C = *(pd_vector+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vector+i_i*i_C+i_j+2)); //pd_Vector[i_i*i_C+i_j+2]
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

// Esta función me calcula la norma de un vector
double Norma_No_Ortogonal_d(double *pd_Vector, double *pd_Superposicion){
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double d_norma = 0; // d_norma es la norma cuadrada del vector x
	double d_sumatoria = 0; // d_sumatoria es lo que iré sumando de los términos del denominador y después returneo
	
	int i_Cs,i_Fs; // Estas son el número de filas y de columnas de la matriz de superposición
	i_Fs = *pd_Superposicion;
	i_Cs = *(pd_Superposicion+1);
	
	// Yo voy a querer hacer el producto escalar en mi espacio no ortogonal. Para eso
	// uso mi matriz de Superposición, que contiene el ángulo entre todos los ejes
	// de mi espacio no ortogonal. Tengo que hacer el producto Vector*matriz*Vector.
	
	// Defino un puntero que guarde los valores del producto intermedio matriz*Vector.
	
	double *pd_Intermedios;
	pd_Intermedios = (double*) malloc((2+i_Fs)*sizeof(double));
	*pd_Intermedios = i_Fs;
	*(pd_Intermedios+1) = 1;
	for(register int i_i=0; i_i<i_Fs; i_i++) *(pd_Intermedios+i_i+2) = 0; // Inicializo el puntero
	
	printf("El vector que recibí es: \n");
	Visualizar_d(pd_Vector);
	
	// Armo el producto de matriz*Vector
	for(register int i_fila=0; i_fila<i_Fs; i_fila++){
		d_sumatoria = 0; // La seteo a 0 para volver a iniciar la sumatoria
		
		for(register int i_columna=0; i_columna<i_Cs; i_columna++) d_sumatoria += *(pd_Superposicion+i_fila*i_Cs+i_columna+2) * (*(pd_Vector+i_columna+2));
		*(pd_Intermedios+i_fila+2) = d_sumatoria;
	}
	
	printf("El cálculo intermedio es: \n");
	Visualizar_d(pd_Intermedios);
	
	// Armo el producto Vector*Intermedios
	d_sumatoria = 0;
	for(register int i_topico=0; i_topico<i_Fs; i_topico++) d_sumatoria += *(pd_Vector+i_topico+2) * (*(pd_Intermedios+i_topico+2));
	d_norma = sqrt(d_sumatoria);
	return d_norma;
}

