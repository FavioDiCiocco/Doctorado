// Este es el header para el módulo general

#ifndef General_H
#define General_H
#include <math.h>

#ifndef MPI
#define M_PI 3.14159265358979323846
#endif

// Acá vienen los structs.
// El struct Red tiene los datos que definen mi red, y la info de la red
typedef struct Red{
	double *pd_Diferencia; // Vector que guarda las diferencias entre PreOpi y Opi.
	double *pd_Opi; // Vector de opinión de cada individuo
	double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	double *pd_Sat; // Matriz de valores de la variable auxiliar de saturación
	double d_Varprom; // Esto es la Variación promedio del sistema. Es cuanto cambia en promedio cada opinión
	int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	int *pi_Activados; //Vector con los agentes que se activaron
	int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
	int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
	char s_Tred[50]; // Esto es el tipo de red, que puede ser de Barabasi, Erdos-Renyi o Random Regular
}s_Red;

typedef s_Red *ps_Red;

// El struct de Parametros tiene todos los datos sobre los parámetros del modelo, valga la redundancia
typedef struct Parametros{
	double d_NormDif; // Este es el factor de normalización de la Variación Promedio.
	double d_CritCorte; // Este número es el piso que tiene que cruzar el Varprom para que se corte la iteración
	double d_alfa; // Controversialidad de los temas
	double d_dt; // Paso temporal de iteración del sistema
	double d_Cosangulo; // Este es el coseno del ángulo entre los tópicos
	double d_chi; // Umbral que determina si el interés del vecino puede generarme más interés.
	double d_lambda; // Constante asociada a la evolución del término de saturación
	int i_Gradomedio; // Este es el grado medio de los agentes de la red.
	int i_m; // Esto sería el número de conexiones que haría para cada agente que se activa.
	int i_N; // Número de agentes en la red
	int i_T; // Cantidad de tópicos
	int i_Itextra; // Esta es la cantidad de iteraciones extra que el sistema tiene que hacer para asegurarme de que el estado alcanzado realmente es estable
	int i_ID; // Esto me va a servir para elegir las distintas redes entre el conjunto de redes estáticas armadas previamente.
	int i_pasosprevios; // Esto es la cantidad de pasos previos que voy a guardar para comparar la variación con el paso actual
}s_Param;

typedef s_Param *ps_Param;

//################################################################################################

double Random();
double Gaussiana(float f_mu, float f_sigma);
double Norma_d(double *pd_x);
double RK4(double *pd_sistema, double (*pf_funcion)(ps_Red ps_var, ps_Param ps_par) ,ps_Red ps_var, ps_Param ps_par);
double Max(double d_a, double d_b);
double Min(double d_a, double d_b);
double Interpolacion(double d_y1, double d_y2, double d_x1,double d_x);
int Visualizar_d(double *pd_vec);
int Visualizar_f(float *pf_vec);
int Visualizar_i(int *pi_vec);
int Escribir_d(double *pd_vec, FILE *pa_archivo);
int Escribir_i(int *pi_vec, FILE *pa_archivo);
int Tamano_Comunidad(int *pi_adyacencia,int i_inicial);
int Delta_Vec_d(double *pd_x1, double *pd_x2, double *pd_Dx);

#endif

