// Este es el header para el módulo general

#ifndef Inicializar_H
#define Inicializar_H
#include <math.h>

int GenerarOpi(ps_Red ps_var, ps_Param ps_par);
int GenerarAng(ps_Red ps_var, ps_Param ps_par);
// int GenerarAdy(ps_Red ps_var, ps_Param ps_par);
int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo);
int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos);

#endif

