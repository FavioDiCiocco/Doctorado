// En este archivo defino todas las funciones generales, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "general.h"


// Esta función me genera un número random
double Random(){
	return ((double) rand()/(double) RAND_MAX);
}

// Esta función me da un valor tomado de una distribución gaussiana con valor medio mu y desviación sigma
double Gaussiana(float f_mu, float f_sigma){
	// Defino mis variables iniciales
	int i_n=10;
	double d_z=0;
	
	// Genero el número que voy a obtener de mi Gaussiana.
	// Para ser sincero, esto es un código legado del cual no comprendo la matemática involucrada.
	for(int i_i=0;i_i<i_n;i_i++) d_z += Random();
	d_z = sqrt(12*i_n) * (d_z/i_n-0.5);
	return d_z*f_sigma+f_mu;
}

// Esta función me calcula la norma de la distancia entre dos partículas
double Norma(double *pd_x){
	// Defino mis variables iniciales
	double d_norm;
	
	// Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada. (Hasta ahora considera vectores de tres coordenadas.)
	d_norm = sqrt((*pd_x)*(*pd_x)+(*(pd_x+1))*(*(pd_x+1))+(*(pd_x+2))*(*(pd_x+2)));
	return d_norm;
}

// Esta función me calcula el vector desplazamiento entre dos partículas
int delta_x(double *d_x1, double *d_x2, double d_L, double *d_Dx){
	// Defino mi vector intermedio de registro de datos
	double *d_diferencias;
	d_diferencias = (double*) malloc(3*sizeof(double));
	
	// Calculo la diferencia entre dos vectores de R^3
	for(int i_i=0;i_i<3;i_i++) *(d_diferencias+i_i) = *(d_x1+i_i)-*(d_x2+i_i);
	for(int i_i=0;i_i<3;i_i++) if(*(d_diferencias+i_i) < -0.5*d_L) *(d_diferencias+i_i) = *(d_diferencias+i_i)+d_L;
	for(int i_i=0;i_i<3;i_i++) if(*(d_diferencias+i_i) > 0.5*d_L) *(d_diferencias+i_i) = *(d_diferencias+i_i)-d_L;
	
	// Me anoto la diferencia en un vector que está en el main del programa, y luego libero el espacio usado.
	for(int i_i=0;i_i<3;i_i++) *(d_Dx+i_i) = *(d_diferencias+i_i);
	free(d_diferencias);
	return 0;
}

//Funciones de Visualización
//----------------------------------------------------------------------------------------
// Esta función es para observar los vectores double
int Visualizar_d(double *pd_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pd_vec;
	i_C = *(pd_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

// Esta función es para observar los vectores float
int Visualizar_f(float *pf_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pf_vec;
	i_C = *(pf_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

// Esta función es para observar los vectores int
int Visualizar_i(int *pi_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}
//--------------------------------------------------------------------------------------------

// Esta función me realiza una iteración del sistema a partir de un RK4.
// Incluso creo que no es necesario el uso del puntero de funciones, pero quiero aprender a usarlo
// En el futuro considerar cómo convertir a la función más general. Acordate, el pd_vec es el
// vector que voy a evolucionar en mi sistema
int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par)){
	// Defino las variables y vectores que voy a necesitar
	int i_filas = (int) *pd_vec; // Este es el número de filas del vector principal
	int i_columnas = (int) *(pd_vec+1); // Este es el número de columnas del vector principal
	double *pd_pendientes[5]; // Este array de punteros guarda los punteros kN
	float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	
	// Voy a armar vectores con las pendientes que calcule, para luego calcular el avance de todas las variables al final
	double *k0;
	k0 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	double *k1;
	k1 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	double *k2;
	k2 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	double *k3;
	k3 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	double *k4;
	k4 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	pd_inicial = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// Inicializo mis punteros
	pd_pendientes[0] = k0;
	pd_pendientes[1] = k1;
	pd_pendientes[2] = k2;
	pd_pendientes[3] = k3;
	pd_pendientes[4] = k4;
	for(register int i_i=0; i_i<i_filas*i_columnas+2; i_i++){
		*(pd_inicial+i_i) = *(pd_vec+i_i);
		for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+i_i) = 0; // De esta manera inicializo los 5 punteros, todos en una sola línea.
	}
	
	// Guardo registro del tamaño de mis vectores k
	for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]) = i_filas;
	for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+1) = i_columnas;
	
	// Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	DT[0] = 0;
	DT[1] = par.f_dt*0.5;
	DT[2] = par.f_dt*0.5;
	DT[3] = par.f_dt;
	
	//##############################################################################################
	//##############################################################################################
	
	// Acá hago las iteraciones del RK4 para hallar las pendientes k
	for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		for(var.i_agente=0; var.i_agente<*pd_vec; var.i_agente++){ // Itero para todos los agentes
			for(var.i_topico=0; var.i_topico<*(pd_vec+1); var.i_topico++){ // Itero para todos los tópicos
				// Calculo el elemento de la pendiente k(i_j+1)
				for(register int i_i=0; i_i<i_filas*i_columnas; i_i++) *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes[i_j]+i_i+2)*DT[i_j];
				*(pd_pendientes[i_j+1]+var.i_agente*i_columnas+var.i_topico+2) = (*fp_funcion)(var,par);
			}
		}
	}
	
	// Una nota para tomar en cuenta de esto es que en estas líneas diría yo se encuentra el punto clave que me genera dudas
	// respecto a la generalidad de esta función. Justamente acá estoy iterando respecto a var.i_agente y var.i_topico.
	// Estos son elementos específicos de estos structs usados y del problema en cuestión. Eso significa que esto es lo
	// que voy a tener que mirar específicamente para corregir. Aunque el problema creo que pasa por el hecho de que a mi
	// me gusta pasar la info usando structs, entonces siempre estás subordinado a los elementos que ponés en tu struct.
	// Por tanto, eso siempre te va a generar este problema, pero a cambio te soluciona el paso de variables a las funciones.
	// En conclusión, una de cal y una de arena. Y esto es el punto central al mirar cuando hagas otro programa.
	
	//##############################################################################################
	//##############################################################################################
	
	// Con TODAS las pendientes calculadas, ahora sí avanzo temporalmente mi sistema
	
	for(register int i_i=0; i_i<i_filas*i_columnas; i_i++){
	*(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+(par.f_dt/6)*(*(pd_pendientes[1]+i_i+2)+*(pd_pendientes[2]+i_i+2)*2+*(pd_pendientes[3]+i_i+2)*2+*(pd_pendientes[4]+i_i+2));
	}
	
	
	//###################################################################
	//###################################################################
	
	// Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// char c_npendientes[] = "k1234"; //
	
	// for(register int i_i=0; i_i<4; i_i++){
		// printf("\n Estas son las pendientes %c%c \n",c_npendientes[0],c_npendientes[i_i+1]);
		// Visualizar_d(pd_pendientes[i_i+1]);
	// }
	
	// También tengo que visualizar mi vector trabajado.
	
	// printf("Este es mi vector antes de evolucionarlo \n");
	// Visualizar_d(pd_inicial);
	
	// printf("Este es mi vector luego de evolucionarlo \n");
	// Visualizar_d(pd_vec);
	
	// Esto lo comento porque ya no es necesario, esta sección me permite visualizar el trabajo que hace este programa.
	
	//####################################################################
	//####################################################################

	// Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	free(pd_inicial);
	for(int register i_i=0; i_i<5; i_i++) free(pd_pendientes[i_i]);
	
	return 0;
}

//--------------------------------------------------------------------------------------------
// Las siguientes funciones son complementos para escribir datos en un archivo

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// Defino las variables del tamaño de mi vector
	int i_C,i_F;
	i_F = *pd_vec;
	i_C = *(pd_vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%lf",*(pd_vec+i_i+2));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// Defino las variables del tamaño de mi vector
	int i_C,i_F;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

