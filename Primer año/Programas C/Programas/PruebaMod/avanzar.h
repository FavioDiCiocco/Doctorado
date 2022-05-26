// Este es el header para el módulo Avanzar

#ifndef Avanzar_H
#define Avanzar_H
#include <math.h>

double Din1(ps_Red ps_var, ps_Param ps_par, ps_Tab ps_tab);
double Din2(ps_Red ps_var, ps_Param ps_par, ps_Tab ps_tab);
int Iteracion(ps_Red ps_var, ps_Param ps_par, ps_Tab ps_tab, double (*pf_Dinamica)(ps_Red ps_var, ps_Param ps_par, ps_Tab ps_tab));

#endif