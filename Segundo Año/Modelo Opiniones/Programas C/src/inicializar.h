// Este es el header para el módulo general

#ifndef Inicializar_H
#define Inicializar_H
#include <math.h>

void GenerarOpi(puntero_Matrices red, double kappa);
void GenerarAng(puntero_Matrices red, puntero_Parametros param);
// void GenerarAdy_Conectada(puntero_Matrices red);
void Generar_Separacion(puntero_Matrices red, puntero_Parametros param);
int Lectura_Adyacencia(int *vec, FILE *archivo);
int Lectura_Adyacencia_Ejes(puntero_Matrices red, FILE *archivo);
void Lectura_Opiniones(double* vec, int* pasos_simulados , FILE *archivo);


#endif

