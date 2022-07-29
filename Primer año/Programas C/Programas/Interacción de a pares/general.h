// Este es el header para el módulo general

#ifndef General_H
#define General_H
#include <math.h>

// Acá vienen los structs.
// El struct Red tiene los datos que definen mi red, y la info de la red
typedef struct Red{
	double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	double *pd_Opi; // Vector de opinión de cada individuo
	int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
	int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
	double *pd_PreOpi; // Vector que guarda la matriz de opiniones del sistema en el paso temporal Previo.
	double *pd_Diferencia; // Vector que guarda las diferencias entre PreOpi y Opi.
	double d_Varprom; // Esto es la Variación promedio del sistema. Es cuanto cambia en promedio cada opinión
}s_Red;

typedef struct Parametros{
	float f_Beta; // Exponente que regula el grado de homofilia
	float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	float f_K; // Influencia social
	float f_alfa; // Controversialidad de los temas
	int i_N; // Número de agentes en la red
	int i_T; // Cantidad de tópicos
	int i_Mopi; // Máximo inicial de opiniones
	float f_Tint; // Esto es el tiempo total que integro a mi sistema
	float f_dt; // Paso temporal de iteración del sistema
	double d_NormDif; // Este es el factor de normalización de la Variación Promedio.
	double d_CritCorte; // Este es el Criterio de Corte de mi sistema. Define cuándo debería parar de evolucionar el sistema
	int i_Itextra; // Esta es la cantidad de iteraciones extra que el sistema debe hacer después de pasar el Criterio de Corte para asegurarse que se encuentra en un estado estable.
}s_Param;

//################################################################################################

double Random();
double Gaussiana(float f_mu, float f_sigma);
double Norma_d(double *pd_x);
int Delta_Vec_d(double *pd_x1, double *pd_x2, double *pd_Dx);
int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par));
int Visualizar_d(double *pd_vec);
int Visualizar_f(float *pf_vec);
int Visualizar_i(int *pi_vec);
int Escribir_d(double *pd_vec, FILE *pa_archivo);
int Escribir_i(int *pi_vec, FILE *pa_archivo);

#endif

