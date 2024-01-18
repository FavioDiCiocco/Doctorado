// Este es el header para el módulo general

#ifndef Inicializar_H
#define Inicializar_H
#include <math.h>

int GenerarOpi(ps_Red ps_variable, int i_region, double d_kappa);
int GenerarAng(ps_Red ps_variable, ps_Param ps_parametro);
int Lectura_Adyacencia(int *pi_vector, FILE *pa_archivo);
int Lectura_Adyacencia_Ejes(ps_Red ps_variable, FILE *pa_archivo);
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos);
int GenerarAdy_Conectada(ps_Red ps_variable, ps_Param ps_parametro);

#endif

