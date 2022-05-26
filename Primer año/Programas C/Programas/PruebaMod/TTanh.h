// Este es el header para el módulo Ttanh

#ifndef TTanh_H
#define TTanh_H


int Indices_Tabla(int* i_indice1, int* i_indice2, double d_argumento);
int Largo_Tabla_TANH(FILE* pa_file);
double IntrpTANH(double d_Arg, ps_Tab ps_val);
int Lectura_Tabla_TANH(double* pd_vec, FILE* pa_file);


#endif

