// En este archivo defino todas las funciones generales, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"


// Esta función me genera un número random entre 0 y 1
double Random(){
	return ((double) rand()/(double) RAND_MAX);
}

// Esta función me da un valor tomado de una distribución gaussiana con valor medio mu y desviación sigma
double Gaussiana(float f_mu, float f_sigma){
	// Defino mis variables iniciales
	int i_n=100;
	double d_z=0;
	
	// Genero el número que voy a obtener de mi Gaussiana.
	// Para ser sincero, esto es un código legado del cual no comprendo la matemática involucrada.
	for(int i_i=0;i_i<i_n;i_i++) d_z += Random();
	d_z = sqrt(12*i_n) * (d_z/i_n-0.5);
	return d_z*f_sigma+f_mu;
}

// Esta función me calcula la norma de un vector
double Norma_d(double *pd_x){
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double d_norm,d_sum = 0;
	int i_C,i_F;
	i_F = *pd_x;
	i_C = *(pd_x+1);
	
	// Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	for(register int i_i=0; i_i<i_C*i_F; ++i_i) d_sum += *(pd_x+i_i+2)*(*(pd_x+i_i+2));
	d_norm = sqrt(d_sum);
	return d_norm;
}


//Funciones de Visualización
//----------------------------------------------------------------------------------------
// Esta función es para observar los vectores double
int Visualizar_d(double *pd_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pd_vec;
	i_C = *(pd_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%.9lf\t",*(pd_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

// Esta función es para observar los vectores float
int Visualizar_f(float *pf_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pf_vec;
	i_C = *(pf_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

// Esta función es para observar los vectores int
int Visualizar_i(int *pi_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}
//--------------------------------------------------------------------------------------------

// La función RK4 evoluciona un sólo elemento de todo un sistema. La idea de esto es que la función sea completamente
// modular, y que en caso de querer evolucionar todo el sistema, desde fuera la llame múltiples veces y listo.
// Además, ahora lo que hace es devolver el cálculo de la evolución del número en vez de asignarlo al vector provisto.
// Esto ofrece mayor libertad a la hora de implementarlo. Justamente la idea de esto es que me permite pasarle una
// foto del sistema en un cierto paso temporal, y guardar los datos evolucionados en un segundo vector, de manera
// que la foto no varía al ser evolucionada. Eso me permite que con sólo dos vectores, uno del paso temporal actual
// y uno del paso temporal siguiente me alcance para poder hacer una evolución sincrónica del sistema.
// Fijate que la foto se ve intacta, porque en la línea antes de calcular el resultado final, vuelvo a reescribir
// mi foto con el valor inicial que tenía, de manera de que esta foto entra y sale igual.
double RK4(double *pd_sistema ,ps_Red ps_var, ps_Param ps_par, double (*pf_funcion)(ps_Red ps_var, ps_Param ps_par)){
	// Defino las variables y vectores que voy a necesitar
	int i_F = (int) *pd_sistema; // Este es el número de filas del vector principal
	int i_C = (int) *(pd_sistema+1); // Este es el número de columnas del vector principal
	float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	double d_resultado = 0; // Este número es el que voy a returnear a la salida de la función
	
	double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	pd_inicial = (double*) malloc((i_F*i_C+2)*sizeof(double));
	
	double *pd_pendientes; // Este puntero de doubles me guarda todos los valores de las pendientes k
	pd_pendientes = (double*) malloc((5+2)*sizeof(double));
	
	// Inicializo mis punteros
	for(register int i_i=0; i_i<i_F*i_C+2; i_i++) *(pd_inicial+i_i) = *(pd_sistema+i_i);
	
	*pd_pendientes = 1;
	*(pd_pendientes+1) = 5;
	for(register int i_i=0; i_i<5;++i_i) *(pd_pendientes+i_i+2)=0;
	
	// Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	DT[0] = 0;
	DT[1] = ps_par->f_dt*0.5;
	DT[2] = ps_par->f_dt*0.5;
	DT[3] = ps_par->f_dt;
		
	// Acá hago las iteraciones del RK4 para hallar las pendientes k
	for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// Calculo el elemento de la pendiente k(i_j+1)
		for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes+i_j+2)*DT[i_j];
		*(pd_pendientes+i_j+1+2) = (*pf_funcion)(ps_var,ps_par);
	}
	
	// Copio al sistema igual que el inicial para deshacer los cambios que hice en el vector principal al calcular los k
	for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2);
	
	// Ahora que tengo los 4 k calculados, avanzo al sujeto que quiero avanzar.
	d_resultado = *(pd_inicial+ps_var->i_agente*i_C+ps_var->i_topico+2)+(ps_par->f_dt/6)*(*(pd_pendientes+3)+*(pd_pendientes+4)*2+*(pd_pendientes+5)*2+*(pd_pendientes+6));
	
	
	// Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// printf("Estoy mirando el RK4 del agente %d y el tópico %d\n", ps_var->i_agente, ps_var->i_topico);
	
	printf("Estas son las pendientes\n");
	Visualizar_d(pd_pendientes);
	
	// También tengo que visualizar mi vector trabajado.
	
	// printf("Este es mi vector antes de evolucionarlo \n");
	// Visualizar_d(pd_inicial);
	
	// printf("Este es mi vector luego de evolucionarlo \n");
	// Visualizar_d(pd_sistema);
	

	// Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	free(pd_inicial);
	free(pd_pendientes);
	
	return d_resultado;
}

//--------------------------------------------------------------------------------------------
// Las siguientes funciones son complementos para escribir datos en un archivo

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// Defino las variables del tamao de mi vector
	int i_C,i_F;
	i_F = *pd_vec;
	i_C = *(pd_vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"%lf\t",*(pd_vec+i_i+2));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// Defino las variables del tamao de mi vector
	int i_C,i_F;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"%d\t",*(pi_vec+i_i+2));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

// Esta función me mide el tamao del grupo al cual pertenece el nodo inicial: i_inicial
int Tamano_Comunidad(int *pi_adyacencia, int i_inicial){
	// Defino la variable del tamao del grupo, el número de filas de la matriz de Adyacencia, el número de agentes
	// restantes por visitar; y los inicializo
	int i_tamao, i_F, i_restantes;
	i_tamao = 0;
	i_F = *pi_adyacencia;
	i_restantes = 0;
	
	// Defino un puntero que registre cuáles agentes están conectados y lo inicializo
	int *pi_Grupo;
	pi_Grupo = (int*) malloc((2+i_F)*sizeof(int));
	
	*pi_Grupo = 1;
	*(pi_Grupo+1) = i_F;
	for(register int i_i=0; i_i<i_F; i_i++) *(pi_Grupo+i_i+2) = 0;
	
	// Defino un puntero que me marque los nuevos sujetos que visitar. Lo hago de tamao i_F para poder asignar un 1 al visitar el agente en cada posición correcta.
	int *pi_Visitar;
	pi_Visitar = (int*) malloc((2+i_F)*sizeof(int));
	
	*pi_Visitar = 1;
	*(pi_Visitar+1) = i_F;
	for(register int i_i=0; i_i<i_F; i_i++) *(pi_Visitar+i_i+2) = 0;
	
	// Empiezo recorriendo la matriz desde un nodo inicial, que será el primero siempre.
	for(register int i_i=0; i_i<i_F; i_i++){
		*(pi_Grupo+i_i+2) = *(pi_adyacencia+i_i+i_inicial*i_F+2);
		*(pi_Visitar+i_i+2) = *(pi_adyacencia+i_i+i_inicial*i_F+2);
	}
	
	do{
		i_restantes = 0;
		// Primero reviso mi lista de gente por visitar
		for(register int i_i=0; i_i<i_F; i_i++){
			// Si encuentro un uno en la lista, reviso esa fila de la matriz de adyacencia. Es decir, la fila i_i
			if(*(pi_Visitar+i_i+2) == 1){
				// Si en esa fila encuentro un uno, tengo que agregar eso al grupo y a la lista de Visitar. Pero no siempre.
				// La idea es: Si el sujeto no estaba marcado en grupo, entonces lo visito y lo marco en el grupo.
				// Si ya estaba marcado, es porque lo visité o está en mi lista de visitar.
				// La idea de esto es no revisitar nodos ya visitados.
				for(register int i_j=0; i_j<i_F; i_j++){
					if(*(pi_adyacencia+i_j+i_i*i_F+2) == 1){
						if(*(pi_Grupo+i_j+2) == 0) *(pi_Visitar+i_j+2) = 1; // Esta línea me agrega el sujeto a visitar sólo si no estaba en el grupo
						*(pi_Grupo+i_j+2) = *(pi_adyacencia+i_j+i_i*i_F+2); // Esta línea me marca al sujeto en el grupo, porque al final si ya había un uno ahí, simplemente lo vuelve a escribir.
					}
				}
				*(pi_Visitar+i_i+2) = 0;
			}
		}
		for(int register i_i=0; i_i<i_F; i_i++) i_restantes += *(pi_Visitar+i_i+2);
	}
	while(i_restantes > 0);
	
	// Finalmente mido el tamao de mi grupo
	for(register int i_i=0; i_i<i_F; i_i++) i_tamao += *(pi_Grupo+i_i+2);
	
	// Libero las memorias malloqueadas
	free(pi_Grupo);
	free(pi_Visitar);
	
	return i_tamao;
}

// Esta función me calcula la diferencia entre dos vectores
int Delta_Vec_d(double *pd_x1, double *pd_x2, double *pd_Dx){
	// Compruebo primero que mis dos vectores sean iguales en tamao
	if(*pd_x1!=*pd_x2 || *(pd_x1+1)!=*(pd_x2+1) || *pd_x1!=*pd_Dx || *(pd_x1+1)!=*(pd_Dx+1)){
		printf("Los vectores son de tamaos distintos, no puedo restarlos\n");
		return 0;
	}
	
	// Defino las variables de tamao de mis vectores
	int i_C,i_F;
	i_F = *pd_x1;
	i_C = *(pd_x1+1);
	
	// Calculo la diferencia entre dos vectores
	for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(pd_Dx+i_i+2) = *(pd_x1+i_i+2)-*(pd_x2+i_i+2);
	
	// Me anoto la diferencia en un vector que está en el main del programa, y luego libero el espacio usado.
	return 0;
}


// // Me defino funciones de máximo y mínimo
double Max(double d_a, double d_b){
	// Defino la variable a usar
	double d_max = 0;
	
	d_max = (d_a > d_b)? d_a : d_b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// Sino se devuelve lo que está a la derecha
	
	return d_max;
}

double Min(double d_a, double d_b){
	// Defino la variable a usar
	double d_min = 0;
	
	d_min = (d_a < d_b)? d_a : d_b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// Sino se devuelve lo que está a la derecha
	
	return d_min;
}


// Esta función toma dos valores de "y" y uno de "x" y los usa para interpolar
// el valor y que le corresponde al argumento x que es el elemento final que se le
// pasa a esta función. En principio esta función sirve para interpolar funciones en
// general y no únicamente a la TANH, pero para usarla con otras funciones se necesita
// modificar el valor de d_deltax según el paso con el que fue armada la tabla de datos
// de la cual se extraen las "y" usadas para interpolar. El motivo de no usar dos valores
// de "x" para definir el d_deltax es porque podría ocurrir que los "x" sean iguales ya que
// el argumento d_x es un valor sobre el cual se calculo un dato de la tabla. Eso llevaría a
// que el programa haga una división por cero y eso sería un grave problema. En el futuro se
// pueden intentar buscar soluciones, del tipo de modificar el valor de d_x2 de manera artificial
// desde afuera, de forma de que el d_deltax no sea nunca cero. Total, una vez obtenido
// d_y2, no necesito respetar el valor de índice i_i2 obtenido antes. Al final, el d_x2
// sólamente tendría el propósito de definir correctametne el d_deltax
double Interpolacion(double d_y1, double d_y2,double d_x1,double d_x){
	// Defino las variables que voy a necesitar
	double d_resultado = 0;
	double d_deltax = 0.00001;
	double d_deltay = d_y2-d_y1;
	
	d_resultado = (d_deltay/d_deltax)*d_x+d_y1+(-d_deltay/d_deltax)*d_x1; // Esta es la cuenta de la interpolación
	
	return d_resultado;
}