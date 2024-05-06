// Este es el header para el módulo general

#ifndef General_H
#define General_H
#include <math.h>

#ifndef MPI
#define M_PI 3.14159265358979323846
#endif

// Acá vienen los structs.
// El struct Matrices_Redes tiene los datos que definen mi red, y la info de la red
typedef struct Matrices_Redes{
	int **Ady; // Lista de vecinos que define mis conexiones. Tiene tamaño que no es rectangular. N filas, y cada fila tiene tamaño variable.
	int **Ady_vecinos; // Lista de la posición de los vecinos de cada agente. Si A[i][j] = l, entonces eso signfica que A[j][l] = i
	double *Dif; // Vector que guarda las diferencias entre PreOpi y Opi.
	double *Opi; // Vector de opinión de cada individuo
	double *Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	double *Sep; // Vector que guarda la inversa de la distancia entre agentes elevada a la -Beta
	double *Tanh; // Matriz con los valores de las tanh calculadas sobre los agentes. Tiene tamaño N*T
	double *Prom_Opi; // Vector de Promedios de Opiniones
	double *Hist; // Matriz de Histograma de agentes en el espacio de Opiniones
	double Variacion_promedio; // Esto es la Variación promedio del sistema. Es cuanto cambia en promedio cada opinión
	int agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	int agente_vecino; // Este es el segundo agente con el cual se pone en contacto el primero.
	int topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
}struct_Matrices;

typedef struct_Matrices * puntero_Matrices;


// El struct de Parametros tiene todos los datos sobre los parámetros del modelo, valga la redundancia
typedef struct Parametros{
	double NormDif; // Este es el factor de normalización de la Variación Promedio.
	double CritCorte; // Este número es el piso que tiene que cruzar el Varprom para que se corte la iteración
	double alfa; // Controversialidad de los temas
	double dt; // Paso temporal de iteración del sistema
	double Cosd; // Este es el coseno del ángulo entre los tópicos
	double delta; // Esto ayuda a que los pesos no diverjan
	double beta; // Este valor es la potencia del término que regula la homofilia
	double epsilon; // Umbral que determina si el interés del vecino puede generarme más interés.
	double kappa; // Esta amplitud regula la relación entre el término lineal y el término logístico
	int N; // Número de agentes en la red
	int T; // Cantidad de tópicos
	int bines; // Cantidad de cajas por eje con las que armo el histograma
	int Iteraciones_extras; // Esta es la cantidad de iteraciones extra que el sistema tiene que hacer para asegurarme de que el estado alcanzado realmente es estable
	int pasosprevios; // Esto es la cantidad de pasos previos que voy a guardar para comparar la variación con el paso actual
	int testigos; // Esta es la cantidad de agentes de cada distancia que voy registrar como máximo
}struct_Parametros;

typedef struct_Parametros * puntero_Parametros;

//################################################################################################

double Random();
double Gaussiana(float mu, float sigma);
double Norma_d(double *x);
double Norma_No_Ortogonal_d(double *vec, double *Sup);
void RK4(double *sistema, double (*func_din)(puntero_Matrices red,puntero_Parametros param), double (*func_act)(puntero_Matrices red,puntero_Parametros param), puntero_Matrices red, puntero_Parametros param);
double Max(double a, double b);
double Min(double a, double b);
double Interpolacion(double y1, double y2, double x1,double x);
void Visualizar_d(double *vec);
void Visualizar_f(float *vec);
void Visualizar_i(int *vec);
void Escribir_d(double *vec, FILE *archivo);
void Escribir_i(int *vec, FILE *archivo);
int Tamano_Comunidad(int *ady,int inicial);
void Delta_Vec_d(double *restado, double *restar, double *resultado);
void Clasificacion(puntero_Matrices red, puntero_Parametros param);
// int Distancia_agentes(int *pi_adyacencia, int *pi_separacion);
// int Lista_testigos(ps_Red ps_red, ps_Param ps_datos);

#endif

