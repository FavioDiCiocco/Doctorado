// Este es el header para el módulo Avanzar

#ifndef Avanzar_H
#define Avanzar_H
#include <math.h>

double Din_sumatoria(ps_Red ps_var, ps_Param ps_par);
double Din_interes(ps_Red ps_var, ps_Param ps_par);
int Iteracion(double *pd_sistema,ps_Red ps_var, ps_Param ps_par, double (*pf_Dinamica)(ps_Red ps_var, ps_Param ps_par));

#endif