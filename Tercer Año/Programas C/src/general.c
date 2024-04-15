// En este archivo defino todas las funciones generales, pero no las declaro.
// El <stdio.h>, o <math.h>, o <stdlib.h>, ¿Son necesarios?

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "general.h"
#include "inicializar.h"


// Esta función me genera un número random entre 0 y 1
double Random(){
	return ((double) rand()/(double) RAND_MAX);
}

// Esta función me da un valor tomado de una distribución gaussiana con valor medio mu y desviación sigma
double Gaussiana(float mu, float sigma){
	
	// Defino mis variables iniciales
	int n=100;
	double z=0;
	
	// Genero el número que voy a obtener de mi Gaussiana.
	// Para ser sincero, esto es un código legado del cual no comprendo la matemática involucrada.
	for(int i=0; i<n; i++) z += Random();
	z = sqrt(12*n) * (z/n-0.5);
	return z*sigma+mu;
}

// Esta función me calcula la norma de un vector
double Norma_d(double *x){
	
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double norm, sum = 0;
	int C, F;
	F = *x;
	C = *(x+1);
	
	// Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	for(int i=0; i< C*F; ++i) sum += *(x+i+2) * (*(x+i+2));
	norm = sqrt(sum);
	return norm;
}

// Esta función me calcula la norma de un vector
double Norma_No_Ortogonal_d(double *vec, double *Sup){
	
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double norma = 0; // norma es la norma cuadrada del vector x
	double sumatoria = 0; // sumatoria es lo que iré sumando de los términos del denominador y después returneo
	
	int Cs,Fs; // Estas son el número de filas y de columnas de la matriz de superposición
	Fs = *Sup;
	Cs = *(Sup+1);
	
	// Yo voy a querer hacer el producto escalar en mi espacio no ortogonal. Para eso
	// uso mi matriz de Superposición, que contiene el ángulo entre todos los ejes
	// de mi espacio no ortogonal. Tengo que hacer el producto Vector*matriz*Vector.
	
	// Defino un puntero que guarde los valores del producto intermedio matriz*Vector.
	
	double *Inter;
	Inter = (double*) malloc((2+Fs)*sizeof(double));
	*Inter = Fs;
	*(Inter+1) = 1;
	for(int i=0; i<Fs; i++) *(Inter+i+2) = 0; // Inicializo el puntero
	
	// Armo el producto de matriz*Vector
	for(int fila=0; fila<Fs; fila++){
		sumatoria = 0; // La seteo a 0 para volver a iniciar la sumatoria
		
		for(int columna=0; columna<Cs; columna++) sumatoria += *( Sup +fila*Cs +columna+2 ) * ( *(vec +columna+2 ) );
		*( Inter +fila+2 ) = sumatoria;
	}
	
	// Armo el producto Vector*Intermedios
	sumatoria = 0;
	for(int topico=0; topico<Fs; topico++) sumatoria += *( vec +topico+2 ) * ( * ( Inter+topico+2 ) );
	norma = sqrt(sumatoria);
	
	// Libero el puntero armado
	free(Inter);
	
	return norma;
}


//Funciones de Visualización
//---------------------------------------------------------------------------------------------------------------------------------------

// Esta función es para observar los vectores double
void Visualizar_d(double *vec){
	
	// Defino las variables que voy a necesitar.
	int F,C;
	F = *vec;
	C = *(vec+1);
	
	// Printeo mi vector
	for(int i=0; i<F; i++){
		for(int j=0; j<C; j++) printf("%lf\t", *(vec+i*C+j+2));
		printf("\n");
	}
	printf("\n");
}

// Esta función es para observar los vectores float
void Visualizar_f(float *vec){
	// Defino las variables que voy a necesitar.
	int F,C;
	F = *vec;
	C = *(vec+1);
	
	// Printeo mi vector
	for(int i=0; i<F; i++){
		for(int j=0; j<C; j++) printf("%lf\t", *(vec+i*C+j+2));
		printf("\n");
	}
	printf("\n");
}

// Esta función es para observar los vectores int
void Visualizar_i(int *vec){
	// Defino las variables que voy a necesitar.
	int F,C;
	F = *vec;
	C = *(vec+1);
	
	// Printeo mi vector
	for(int i=0; i<F; i++){
		for(int j=0; j<C; j++) printf("%d\t", *( vec+i*C+j+2 ));
		printf("\n");
	}
	printf("\n");
}

//------------------------------------------------------------------------------------------------------------------------------------------

// La función esta va a recibir un array que es el sistema a evolucionar, la función dinámica para realizar la evolución,
// los punteros a struct con la info relevante para pasar a la ecuación dinámica y luego con eso evoluciona mi sistema.
// La idea es evolucionar TODO el array en una sola llamada de RK4, a diferencia de implementaciones anteriores.

void RK4(double *sistema, double (*func_din)(puntero_Matrices red, puntero_Parametros param), double (*func_act)(puntero_Matrices red, puntero_Parametros param), puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables y vectores que voy a necesitar
	int F = (int) *sistema; // Este es el número de filas del vector principal
	int C = (int) *(sistema+1); // Este es el número de columnas del vector principal
	double DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	// Este me guarda las condiciones iniciales del sistema, que las voy a necesitar al calcular cada paso del RK4
	double *inicial;
	inicial = (double*) malloc(( F*C+2 )*sizeof(double));
	
	// Inicializo mi puntero inicial
	for(int i=0; i<F*C+2; i++) *(inicial+i) = *(sistema+i);
	
	// Armo un puntero a arrays para las pendientes
	double *pendientes[5];
	
	// Malloqueo e incializo los punteros
	for(int i=0; i<5; i++){
		pendientes[i] = (double*) calloc( F*C+2 , sizeof(double)); // Hago el malloc
		*(pendientes[i]) = F; // Defino el tamaño de las filas
		*( pendientes[i]+1 ) = C; // Defino el tamaño de las columnas
	}
	
	// Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	DT[0] = 0;
	DT[1] = param->dt * 0.5;
	DT[2] = param->dt * 0.5;
	DT[3] = param->dt;
	
	// Acá hago las iteraciones del RK4 para hallar las pendientes k
	for(int j=0; j<4; j++){
		
		// Avanzo el sistema para el cálculo de la siguiente pendiente.
		for(int i=0; i<F*C; i++) *( sistema+i+2 ) = *( inicial+i+2 )+ *( pendientes[j] +i+2 )*DT[j];
		// Recalculo los valores de las exponenciales y de la separación entre agentes
		for(red->agente_vecino=0; red->agente_vecino < F; red->agente_vecino++){
			for(red->topico=0; red->topico < C; red->topico++) red->Exp[red->agente_vecino*C +red->topico +2] = (*func_act) (red, param);
		}
		Generar_Separacion(red, param);
		
		// Avanzo en todos los agentes
		for(red->agente=0; red->agente < F; red->agente++){
			
			// Avanzo en todos los tópicos
			for(red->topico=0; red->topico < C; red->topico++){
				
				// Calculo el elemento de la pendiente k(i_j+1)
				*(pendientes[j+1] +red->agente*C +red->topico+2) = (*func_din) (red, param);
			}
		}
		
	}
	
	// Reescribo el vector de mi sistema con los valores luego de haber hecho la evolución dinámica
	for(int i=0; i< F*C; i++) *( sistema+i+2 ) = *( inicial+i+2 ) + (param->dt/6) * (*( pendientes[1]+i+2 ) +*( pendientes[2]+i+2 )*2 +*( pendientes[3]+i+2 )*2 +*( pendientes[4]+i+2 ));
	
	// Libero el espacio de memoria ocupada por los arrays.
	free(inicial);
	for(int i=0; i<5; i++) free(pendientes[i]);
}


// Esta función es la que toma la distribución de datos finales y a partir de esto construye un histograma
// de estos datos en el espacio de opiniones.

void Clasificacion(puntero_Matrices red, puntero_Parametros param){
	
	// Defino las variables y vectores que voy a necesitar
	int fila,columna;
	int Fd = (int) red->Hist[0]; // Este es el número de filas de la matriz de histograma
	int Cd = (int) red->Hist[1]; // Este es el número de columnas de la matriz de histograma
	int Fo = (int) red->Opi[0]; // Este es el número de filas de la matriz de opiniones
	int Co = (int) red->Opi[1]; // Este es el número de columnas de la matriz de opiniones
	double ancho = 2/param->bines; // Este es el ancho de cada cajita en la que separo el espacio de opiniones.
	
	// Normalizo mi histograma y la corro a la región [0,2]
	for(int i =0; i< Fo*Co; i++) red->Opi[i] = red->Opi[i] / param->kappa + 1;
	
	// Hago el conteo de agentes en cada una de las cajitas
	for(int agente = 0; agente < Fo; agente++ ){
		columna = fmin(floor(red->Opi[agente*Co+2]/ancho),param->bines-1);
		fila = fmin(floor(red->Opi[agente*Co+1+2]/ancho),param->bines-1);
		red->Hist[fila*Fd+columna+2] += 1;
	}
	
	// Resuelto el conteo, ahora lo normalizo
	for(int i =0; i<Fd*Cd; i++) red->Hist[i+2] = red->Hist[i+2]/param->N;
}


//--------------------------------------------------------------------------------------------
// Las siguientes funciones son complementos para escribir datos en un archivo

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
void Escribir_d(double *vec, FILE *archivo){
	
	// Defino las variables del tamaño de mi vector
	int C,F;
	F = *vec;
	C = *(vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(int i=0; i<C*F; i++) fprintf(archivo,"%.6lf\t",*( vec+i+2 ));
	fprintf(archivo,"\n");
}

// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
void Escribir_i(int *vec, FILE *archivo){
	
	// Defino las variables del tamao de mi vector
	int C,F;
	F = *vec;
	C = *(vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(int i=0; i<C*F; i++) fprintf(archivo,"%d\t",*( vec+i+2 ));
	fprintf(archivo,"\n");
}

// Esta función me mide el tamao del grupo al cual pertenece el nodo inicial: i_inicial
int Tamano_Comunidad(int *ady, int inicial){
	
	// Defino la variable del tamaño del grupo, el número de filas de la matriz de Adyacencia, el número de agentes
	// restantes por visitar; y los inicializo
	int tamano, F, restantes;
	tamano = 0;
	F = *ady;
	restantes = 0;
	
	// Defino un puntero que registre cuáles agentes están conectados y lo inicializo
	int *Grupo;
	Grupo = (int*) calloc((2+F), sizeof(int));
	
	*Grupo = 1;
	*(Grupo+1) = F;
	
	// Defino un puntero que me marque los nuevos sujetos que visitar. Lo hago de tamao i_F para poder asignar un 1 al visitar el agente en cada posición correcta.
	int *Visitar;
	Visitar = (int*) calloc( (2+F), sizeof(int));
	
	*Visitar = 1;
	*( Visitar+1 ) = F;
	
	// Empiezo recorriendo la matriz desde un nodo inicial, que será el primero siempre.
	for(int i=0; i<F; i++){
		*( Grupo+i+2 ) = *( ady+i+inicial*F+2 );
		*( Visitar+i+2 ) = *( ady+i+inicial*F+2 );
	}
	
	do{
		restantes = 0;
		
		// Primero reviso mi lista de gente por visitar
		for(int i=0; i<F; i++){
			
			// Si encuentro un uno en la lista, reviso esa fila de la matriz de adyacencia. Es decir, la fila i_i
			if( *( Visitar+i+2 ) == 1){
				
				// EXPLICACIÓN DE ESTA MARAÑA DE FUNCIONES
				// Si en esa fila encuentro un uno, tengo que agregar eso al grupo y a la lista de Visitar. Pero no siempre.
				// La idea es: Si el sujeto no estaba marcado en grupo, entonces lo visito y lo marco en el grupo.
				// Si ya estaba marcado, es porque lo visité o está en mi lista de visitar.
				// La idea de esto es no revisitar nodos ya visitados.
				
				for(int j=0; j<F; j++){
					if( *( ady+j+i*F+2 ) == 1){
						if( *(Grupo+j+2 ) == 0) *( Visitar+j+2 ) = 1; // Esta línea me agrega el sujeto a visitar sólo si no estaba en el grupo
						*( Grupo+j+2 ) = *( ady+j+i*F+2 ); // Esta línea me marca al sujeto en el grupo, porque al final si ya había un uno ahí, simplemente lo vuelve a escribir.
					}
				}
				*( Visitar+i+2 ) = 0;
			}
		}
		for(int i=0; i<F; i++) restantes += *( Visitar+i+2 );
	}
	while(restantes > 0);
	
	// Finalmente mido el tamao de mi grupo
	for(int i=0; i<F; i++) tamano += *( Grupo+i+2 );
	
	// Libero las memorias malloqueadas
	free(Grupo);
	free(Visitar);
	
	return tamano;
}

// Esta función me calcula la diferencia entre dos vectores
void Delta_Vec_d(double *restado, double *restar, double *resultado){
	
	// Compruebo primero que mis dos vectores sean iguales en tamao
	if(*restado != *restar || *( restado+1 ) != *( restar+1 ) || *restado != *resultado || *( restado+1 ) != *( resultado+1 )){
		printf("Los vectores son de tamaños distintos, no puedo restarlos\n");
	}
	
	// Defino las variables de tamao de mis vectores
	int C, F;
	F = *restado;
	C = *( restado+1 );
	
	// Calculo la diferencia entre dos vectores
	for(int i=0; i<C*F; ++i) *( resultado+i+2 ) = *( restado+i+2 ) - *( restar+i+2 );
}


// // Me defino funciones de máximo y mínimo
double Max(double a, double b){
	// Defino la variable a usar
	double max = 0;
	
	max = (a > b)? a : b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// Sino se devuelve lo que está a la derecha
	return max;
}


double Min(double a, double b){
	// Defino la variable a usar
	double min = 0;
	
	min = (a < b)? a : b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// Sino se devuelve lo que está a la derecha
	
	return min;
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
double Interpolacion(double y1, double y2,double x1,double x){
	// Defino las variables que voy a necesitar
	double resultado = 0;
	double deltax = 0.00001;
	double deltay = y2 - y1;
	
	resultado = (deltay / deltax)* x+y1+(- deltay / deltax)* x1; // Esta es la cuenta de la interpolación
	
	return resultado;
}


/*

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Esta función recibe la matriz de adyacencia y a partir de eso coloca en el vector pi_sep la distancia de cada agente al agente cero.
int Distancia_agentes(int *pi_adyacencia, int *pi_separacion){
	// Defino las variables necesarias para mi función
	int i_distmax,i_F,i_restantes, i_distancia;
	i_F = *pi_adyacencia; // Número de filas de la matriz de Adyacencia.
	i_distancia = 1; // Este valor lo uso para asignar la distancia de los agentes al nodo principal.
	
	//################################################################################################################################
	
	// Defino un puntero con los sujetos que voy a visitar. Lo hago de tamao i_F para poder asignar un 1 al visitar el agente en cada posición correcta.
	int *pi_Visitar;
	pi_Visitar = (int*) malloc((2+i_F)*sizeof(int));
	
	// Lo inicializo
	*pi_Visitar = 1;
	*(pi_Visitar+1) = i_F;
	for(register int i_i=0; i_i<i_F; i_i++) *(pi_Visitar+i_i+2) = 0;
	
	// Defino un puntero que me marque los sujetos a visitar en la próxima iteración. Lo hago de tamao i_F para poder asignar un 1 al visitar el agente en cada posición correcta.
	int *pi_Marcados;
	pi_Marcados = (int*) malloc((2+i_F)*sizeof(int));
	
	// Lo inicializo
	*pi_Marcados = 1;
	*(pi_Marcados+1) = i_F;
	for(register int i_i=0; i_i<i_F; i_i++) *(pi_Marcados+i_i+2) = 0;
	
	// Empiezo recorriendo la matriz desde un nodo inicial, que será el primero siempre.
	// Esto pondrá un 1 en los agentes que son los primeros vecinos del primer nodo.
	for(register int i_i=0; i_i<i_F; i_i++){
		*(pi_Marcados+i_i+2) = *(pi_adyacencia+i_i+2);
		*(pi_separacion+i_i+2) = *(pi_adyacencia+i_i+2);
	}
	
	// Marco al primer agente con un número, sólo para que no sea visitado al pedo
	*(pi_separacion+2) = -1;
	
	do{
		i_restantes = 0; // Lo vuelvo a cero para después contar los agentes restantes
		for(register int i_i=0; i_i<i_F; i_i++) *(pi_Visitar+i_i+2) = *(pi_Marcados+i_i+2);  // Paso todos los agentes marcados a la lista de Visitar
		for(register int i_i=0; i_i<i_F; i_i++) *(pi_Marcados+i_i+2) = 0; // Limpio mi lista de marcados
		i_distancia++; // Paso a revisar a los vecinos que se encuentran a un paso más de distancia
		
		// Primero reviso mi lista de gente por visitar
		for(register int i_agente=0; i_agente<i_F; i_agente++){
			// Si encuentro un uno en la lista, reviso esa fila de la matriz de adyacencia. Es decir, la fila i_agente
			if(*(pi_Visitar+i_agente+2) == 1){
				
				// Si en esa fila encuentro un uno, tengo que agregar eso a la lista de Visitar. Pero no siempre.
				// La idea es: Si el sujeto no estaba marcado en separación, entonces lo visito y lo marco en separación.
				// Si ya estaba marcado, es porque lo visité o está en mi lista de Marcados.
				// La idea de esto es no revisitar nodos ya visitados.
				for(register int i_vecino=0; i_vecino<i_F; i_vecino++){
					if(*(pi_adyacencia+i_vecino+i_agente*i_F+2) == 1){
						if(*(pi_separacion+i_vecino+2) == 0){ 
							*(pi_Marcados+i_vecino+2) = 1; // Esta línea me agrega el sujeto a visitar sólo si no estaba en el grupo
							*(pi_separacion+i_vecino+2) = i_distancia; // Esta línea me marca al sujeto en el grupo, porque al final si ya había un uno ahí, simplemente lo vuelve a escribir.
						}
					}
				}
				*(pi_Visitar+i_agente+2) = 0; // Visitado el agente, lo remuevo de mi lista
			}
		}
		for(int register i_i=0; i_i<i_F; i_i++) i_restantes += *(pi_Marcados+i_i+2);
	}
	while(i_restantes > 0);
	
	// Los agentes están catalogados según su distancia al centro. Queda liberar espacios de memoria y últimos detalles.
	free(pi_Visitar);
	free(pi_Marcados);
	*(pi_separacion+2) = 0; // La distancia del primer nodo a sí mismo es cero.
	i_distmax = i_distancia-1; // El while termina en distancia una unidad extra de la distancia recorrida.
	
	return i_distmax;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Esta función recibe el vector separación de los agentes al nodo inicial, el de cantidad de agentes
// a cada distancia y el de testigos, y me agarra los primeros tres agentes que se encuentran a esa
// distancia. Si no hay tres, agarra los que haya.
// int Lista_testigos(ps_Red ps_red, ps_Param ps_datos){
	// // Preparo las variables con las que inicio mi código
	// int i_agente_guardar=0; // Esta variable representa a los agentes que voy a anotar para guardar sus datos
	// int i_posicion_testigo=0; // Esta variable es la posición en el vector de Testigos a medida que voy completando el vector.
	
	// // Hago todo el proceso de anotar agentes de cada una de las distancias. Me anoto i_testigos o pi_cantidad de agentes, lo que sea menor.
	// for(register int i_distancia=0; i_distancia<ps_datos->i_distmax+1; i_distancia++){
		// i_agente_guardar = 0;
		// for(register int i_iteracion_testigo=0; i_iteracion_testigo<fmin(ps_datos->i_testigos, ps_red->pi_Cant[i_distancia+2]); i_iteracion_testigo++){
			// while(ps_red->pi_Sep[i_agente_guardar+2] != i_distancia) i_agente_guardar++;
			// ps_red->pi_Tes[i_posicion_testigo+2] = i_agente_guardar;
			// i_agente_guardar++;
			// i_posicion_testigo++;
		// }
	// }
	
	// return 0;
// }



//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Esta es la función de RK4 que usaba hasta la fase de Log_1D. La cambié por una que evolucione todo el
sistema de forma sincrónica en vez de evolucionar un elemento del array a la vez y hacer artilugios para que
el sistema evolucione de forma sincrónica.

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
	double DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
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
	DT[1] = ps_par->d_dt*0.5;
	DT[2] = ps_par->d_dt*0.5;
	DT[3] = ps_par->d_dt;
		
	// Acá hago las iteraciones del RK4 para hallar las pendientes k
	for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// Calculo el elemento de la pendiente k(i_j+1)
		for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes+i_j+2)*DT[i_j];
		*(pd_pendientes+i_j+1+2) = (*pf_funcion)(ps_var,ps_par);
	}
	
	// Copio al sistema igual que el inicial para deshacer los cambios que hice en el vector principal al calcular los k
	for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2);
	
	// Ahora que tengo los 4 k calculados, avanzo al sujeto que quiero avanzar.
	d_resultado = *(pd_inicial+ps_var->i_agente*i_C+ps_var->i_topico+2)+(ps_par->d_dt/6)*(*(pd_pendientes+3)+*(pd_pendientes+4)*2+*(pd_pendientes+5)*2+*(pd_pendientes+6));
	
	
	// Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// printf("Estoy mirando el RK4 del agente %d y el tópico %d\n", ps_var->i_agente, ps_var->i_topico);
	
	// printf("Estas son las pendientes\n");
	// Visualizar_d(pd_pendientes);
	
	// También tengo que visualizar mi vector trabajado.
	
	// printf("Este es mi vector de opinión actual \n");
	// Visualizar_d(ps_var->pd_Opi);
	
	// printf("Este es mi vector luego de evolucionarlo \n");
	// Visualizar_d(pd_sistema);
	

	// Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	free(pd_inicial);
	free(pd_pendientes);
	
	return d_resultado;
}

*/

