// Este es el header para el módulo Avanzar

#ifndef Avanzar_H
#define Avanzar_H
#include <math.h>

double Din1(ps_Red ps_var, ps_Param ps_par);
double Din2(ps_Red ps_var, ps_Param ps_par);
int Iteracion(ps_Red ps_var, ps_Param ps_par, double (*pf_Dinamica)(ps_Red ps_var, ps_Param ps_par));

#endif