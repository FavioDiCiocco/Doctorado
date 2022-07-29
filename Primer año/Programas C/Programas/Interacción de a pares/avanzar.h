// Este es el header para el módulo Manipular

#ifndef Avanzar_H
#define Avanzar_H
#include <math.h>

double Din1(s_Red var, s_Param par);
double Din2(s_Red var, s_Param par);
int Iteracion(s_Red var, s_Param par, double (*pf_Dinamica)(s_Red var, s_Param par));

#endif