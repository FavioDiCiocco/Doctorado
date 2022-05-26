// En este archivo defino todas las funciones que trabajan con la tabla de valores de TANH.

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "TTanh.h"


// Voy a armar las funciones que voy a necesitar para poder realizar las interpolaciones.

// La siguiente función es la que me dice la cantidad de números que hay guardados en el
// archivo. La idea es que use este número para armar un vector que es el que guarda
// la posición de cada uno de los números de la tabla.
int Largo_Tabla_TANH(FILE *pa_file){
	// Defino la variable que voy a returnear
	int i_largo = 0;
	double d_salida = 0;
	
	fseek(pa_file,0,SEEK_SET);
	while(fscanf(pa_file,"%lf\t",&d_salida)!=EOF) i_largo++;
	fseek(pa_file,0,SEEK_SET);
	
	return i_largo;
}

// Con esta función ubico el índice que le corresponde a cada número que va entre -5 y 5
// en el vector de valores de la tabla. Es importante que pase los índices por referencia
// porque sino los valores calculados no se pasan por fuera de la función y es como si no
// hubiera hecho nada.
int Indices_Tabla(int* i_indice1, int* i_indice2, double d_argumento){
	// Defino las variables que voy a necesitar.
	double d_indice = 0; // Este es el valor "exacto" del índice que le correspondería al argumento
	
	d_indice = (d_argumento+5)/0.00001; // Esto sale de despejar la fórmula para calcular los valores de la tabla.
	// El 0,00001 es el paso con el que armé los valores de la tabla
	
	d_indice = Max(d_indice,0);
	d_indice = Min(d_indice,1000003-1);
	
	*i_indice1 = floor(d_indice);
	*i_indice2 = ceil(d_indice);
	
	return 0;
}


// Esta función empaqueta el resto para que sólo reciba los elementos básicos.
// La idea de esta función es que sea el reemplazo directo de la función tanh.
// De esa manera la pongo directo en Din1 y listo.
double IntrpTANH(double d_Arg, ps_Tab ps_val){
	// Defino las variables que necesito
	double d_Y1,d_Y2,d_X1,d_resultado;
	int i_i1, i_i2;
	i_i1 = 0;
	i_i2 = 0;
	
	Indices_Tabla(&i_i1, &i_i2,d_Arg);
	d_Y1 = ps_val->pd_valores[i_i1+2];
	d_Y2 = ps_val->pd_valores[i_i2+2];
	d_X1 = -5+0.00001*i_i1;
	
	d_resultado = Interpolacion(d_Y1,d_Y2,d_X1,d_Arg);
	
	
	return d_resultado;
}


// Esta función se encarga de leer la tabla de valores y colocarla en
// el vector de valores del struct
int Lectura_Tabla_TANH(double* pd_vec, FILE* pa_file){
	// Defino mi variable de salida y el entero que recorra los índices
	double d_salida = 0;
	int i_indice = 0;
	
	// Leo los datos de la tabla y los guardo en el vector.
	while(fscanf(pa_file,"%lf\t",&d_salida)!=EOF){
		*(pd_vec+i_indice+2)= d_salida;
		i_indice++;
	}
	
	return 0;
}