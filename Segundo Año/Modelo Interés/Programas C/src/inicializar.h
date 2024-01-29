// Este es el header para el módulo general

#ifndef Inicializar_H
#define Inicializar_H
#include <math.h>

void GenerarOpi(puntero_Matrices red, int region, double kappa);
void GenerarAng(puntero_Matrices red, puntero_Parametros param);
void GenerarAdy_Conectada(puntero_Matrices red);
int Lectura_Adyacencia(int *vec, FILE *archivo);
int Lectura_Adyacencia_Ejes(puntero_Matrices red, FILE *archivo);
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos);

#endif

