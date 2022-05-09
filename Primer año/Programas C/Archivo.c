// Este archivo es para guardar programas de pruebas armados previamente.

Para acceder a los programas rápido salteando todo el código que no querés ver
lo que tenés que hacer es apretar Ctrl+F, eso te abre el buscador de palabras dentro del código.
Lo que tenés que buscar es la palabra "NUEVO_PROGRAMA", con las mayúsculas y guión, todo correcto.
Luego vas saltando de uno al siguiente, y eso rápidamente te mueve entre programas


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA


Este es el programa que armé el 14/12/2020. La idea era probar cómo funcionan los arrays de punteros.
Esto es sólo para ver cómo funcionan, no hace nada especial. La sección de la evolución dinámica del
sistema no es necesaria para este programa.




// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// struct Red{
	// double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// double *pd_Opi; // Vector de opinión de cada individuo
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
// };

// struct Parametros{
	// float f_Beta; // Exponente que regula el grado de homofilia
	// float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	// float f_K; // Influencia social
	// float f_alfa; // Controversialidad de los temas
	// int i_N; // Número de agentes en la red
	// int i_T; // Cantidad de tópicos
	// int i_Mopi; // Máximo inicial de opiniones
	// float f_dt; // Paso temporal de iteración del sistema
// };

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// int GenerarOpi(struct Red var, struct Parametros par);
// int GenerarAng(struct Red var);
// int GenerarAdy(struct Red var, struct Parametros par);
// double Din1(struct Red var, struct Parametros par, int i_j);
// double Din2(struct Red var, struct Parametros par);
// double Duplicar(struct Red var, struct Parametros par);



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// //---------------------------------------------------------------------------------------------------

// // Vamos a intentar ver como se leen archivos y data desde estos
// // Creo que lo más útil va a ser usar el fscanf porque mis datos están separados por tabulaciones
// // Acá está la forma de leer el archivo. Sólo necesitaría poder saber que tan largo es antes de
// // subirlo, pero creo que eso sería mucho pedir.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	
	// // Defino los nombres de mis structs en este programa y los valores de los parámetros
	// struct Red red;
	// struct Parametros datos;
	// datos.f_Beta = 0.5;
	// datos.f_Pint = 1;
	// datos.f_K = 0.5;
	// datos.f_alfa = 2;
	// datos.i_N = 5;
	// datos.i_T = 4;
	// datos.i_Mopi = 3;
	// datos.f_dt = 0.5;
	
	// // Asigno memoria a mis matrices y las inicializo
	
	// // Matriz de Vectores de opinión. Tiene tamaño N*T
	// red.pd_Opi = (double*) malloc((datos.i_N*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Opi[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Opi[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Opi[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarOpi(red,datos); // Genero la red de Vectores de opinión propiamente dicha.

	// // Matriz de Superposición. Tiene tamaño T*T
	// red.pd_Ang = (double*) malloc((datos.i_T*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_T*datos.i_T+2; i_i++) red.pd_Ang[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Ang[0] = datos.i_T; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Ang[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarAng(red); // Genero la red de Vectores de opinión propiamente dicha.
	
	// // Matriz de Adyacencia. Tiene tamaño N*N
	// red.pi_Ady = (int*) malloc((datos.i_N*datos.i_N+2)*sizeof(int)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_N+2; i_i++) red.pi_Ady[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pi_Ady[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pi_Ady[1] = datos.i_N; // Guardo el número de columnas en la segunda coordenada del vector
	// GenerarAdy(red,datos); // Genero la red de Adyacencia propiamente dicha.
	
	// double *pa_proba[2];
	// pa_proba[0] = red.pd_Opi;
	// pa_proba[1] = red.pd_Ang;
	
	// // Visualizo las redes para ver que todo funca bien
	// /*Visualizar(red,0);
	// Visualizar(red,1);
	// Visualizar(red,2);*/
	
	// // Voy a probar visualizar a partir de mi array de punteros
	// for(register int i_i=0; i_i<2;i_i++){
		// int i_columnas = *(pa_proba[i_i]+1);
		// for(register int i_j=0; i_j<*pa_proba[i_i]; i_j++){
			// for(register int i_p=0; i_p<*(pa_proba[i_i]+1); i_p++) printf("%.03lf\t",*(pa_proba[i_i]+i_j*i_columnas+i_p+2));
			// printf("\n");
		// }
		// printf("\n");
	// }
	
	
	// // Evolución dinámica del sistema
	// /*
	// double d_prueba;
	// for(register int i_v=0; i_v<3; i_v++){
		// d_prueba = Din2(red,datos,0,i_v);
		// printf("La pendiente para el topico %d me dio %.5lf\n",i_v,d_prueba);
	// }*/
	
	// // Ejecuto los comandos finales para medir el tiempo.
	// free(red.pi_Ady);
	// free(red.pd_Ang);
	// free(red.pd_Opi);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }




// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(struct Red var, struct Parametros par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Opi[0];
	// i_Columnas = (int) var.pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(int i_i=0;i_i<i_Filas*i_Columnas;i_i++) var.pd_Opi[i_i+2] = par.i_Mopi*(Random()-0.5)*2;
	// return 0;
// }

// // Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
// int GenerarAng(struct Red var){
	// // Obtengo las dimensiones de la matriz de Superposicion.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Ang[0];
	// i_Columnas = (int) var.pd_Ang[1];
	
	// // Inicializo la matriz de Superposicion de mi sistema.
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=0; i_j<i_i; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = 0.5; // Decidí poner 0.5 entre todos los tópicos de mi modelo
	// for(int i_i=0; i_i<i_Filas; i_i++) var.pd_Ang[i_i*i_Columnas+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = var.pd_Ang[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// return 0;
// }


// // Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
// int GenerarAdy(struct Red var, struct Parametros par){
	// // Obtengo las dimensiones de la matriz de Adyacencia.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pi_Ady[0];
	// i_Columnas = (int) var.pi_Ady[1];
	
	// // Inicializo la matriz de Adyacencia de mi sistema
	// for(int i_i=0; i_i<i_Filas; i_i++) {
		// for(int i_j=0; i_j<i_i; i_j++) {
			// // Saco un número random y veo si asigno un 1 o un 0 a esa posición-
			// if(Random()<par.f_Pint) var.pi_Ady[i_i*i_Columnas+i_j+2] = 1;
			// else var.pi_Ady[i_i*i_Columnas+i_j+2] = 0;
		// }
	// }
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pi_Ady[i_i*i_Columnas+i_j+2] = var.pi_Ady[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// // for(int i=0; i<par.N; i++) var.Ady[i*par.N+i] = 1; // Esto me pone 1 en toda la diagonal
	// return 0;
// }

// // #####################################################################################

// // Estas son las funciones del avance
// //#######################################################################################

// // Esta función resuelve un término de los de la sumatoria.
// double Din1(struct Red var, struct Parametros par,int i_j){
	
	// // Defino las variables locales de mi función. d_resultados es el return. 
	// // d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	// double d_resultado,d_opiniones_superpuestas = 0;
	
	// // Obtengo el tamaño de columnas de mis tres matrices
	// int i_Co,i_Cs,i_Ca;
	// i_Co = (int) var.pd_Opi[1];
	// i_Cs = (int) var.pd_Ang[1];
	// i_Ca = var.pi_Ady[1];
	
	// for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += var.pd_Ang[var.i_topico*i_Cs+i_p+2]*var.pd_Opi[i_j*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	// d_resultado = var.pi_Ady[var.i_agente*i_Ca+i_j+2]*tanh(par.f_alfa*d_opiniones_superpuestas); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	// return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
// }

// // Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
// double Din2(struct Red var, struct Parametros par){
	
	// // Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	// double d_resultado,d_sumatoria=0;
	
	// // Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// for(register int i_j=0; i_j<var.i_agente; i_j++) d_sumatoria += Din1(var,par,i_j);
	// for(register int i_j=var.i_agente+1; i_j<par.i_N; i_j++) d_sumatoria += Din1(var,par,i_j);
	
	// // Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	// int i_Columnas = (int) var.pd_Opi[1];
	// d_resultado = -var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2]+par.f_K*d_sumatoria;
	// return d_resultado;
// }

// //#######################################################################################


// // Esta función me realiza una iteración del sistema a partir de un RK4.
// // Creo que no hay forma de que esta función sea 100% general para cualquier caso.
// // Incluso creo que no es necesario el uso del puntero de funciones, pero quiero aprender a usarlo
// int RK4(double *pd_vec ,struct Red var, struct Parametros par, double (*fp_funcion)(struct Red var, struct Parametros par)){
	// // Defino las variables y vectores que voy a necesitar
	// int i_filas = (int) *pd_vec; // Este es el número de filas del vector principal
	// int i_columnas = (int) *(pd_vec+1); // Este es el número de columnas del vector principal
	// double *pd_pendientes[5]; // Este array de punteros guarda los punteros kN
	// float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	
	// // Voy a armar vectores con las pendientes que calcule, para luego calcular el avance de todas las variables al final
	// double *k0;
	// k0 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k1;
	// k1 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k2;
	// k2 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k3;
	// k3 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k4;
	// k4 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	// pd_inicial = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// // Inicializo mis punteros
	// pd_pendientes[0] = k0;
	// pd_pendientes[1] = k1;
	// pd_pendientes[2] = k2;
	// pd_pendientes[3] = k3;
	// pd_pendientes[4] = k4;
	// for(register int i_i=0; i_i<i_filas*i_columnas+2; i_i++){
		// *(pd_inicial+i_i) = *(pd_vec+i_i);
		// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+i_i) = 0; // De esta manera inicializo los 5 punteros, todos en una sola línea.
	// }
	
	// // Guardo registro del tamaño de mis vectores k
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]) = i_filas;
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+1) = i_columnas;
	
	// // Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	// DT[0] = 0;
	// DT[1] = var.f_dt*0.5;
	// DT[2] = var.f_dt*0.5;
	// DT[3] = var.f_dt;
		
	// // Acá hago las iteraciones del RK4 para hallar las pendientes k
	// for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// for(var.i_agente=0; var.i_agente<*pd_vec; var.i_agente++){ // Itero para todos los agentes
			// for(var.i_topico=0; var.i_topico<*(pd_vec+1); var.i_topico++){ // Itero para todos los tópicos
				// // Calculo el elemento de la pendiente k(i_j+1)
				// for(register int i_i=0; i_i<i_filas*i_columnas;i_i++) *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes[i_j]+i_i+2)*DT[i_j];
				// *(pd_pendientes[i_j+1]+var.i_agente*i_columnas+var.i_topico+2) = (*fp_funcion)(var,par);
			// }
		// }
	// }
	
	// // Con TODAS las pendientes calculadas, ahora sí avanzo temporalmente mi sistema
	
	// for(register int i_i=0; i_i<i_filas*i_columnas; i_i++){
	// *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+(var.f_dt/6)*(*(pd_pendientes[1]+i_i+2)+*(pd_pendientes[2]+i_i+2)*2+*(pd_pendientes[3]+i_i+2)*2+*(pd_pendientes[4]+i_i+2));
	// }
	
	// // Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// // Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// char c_npendientes[200] = {'k','1','k','2','k','3','k','4'};
	
	// for(register int i_i=0; i_i<4; i_i++){
		// printf("\n Estas son las pendientes %c%c \n",c_npendientes[i_i*2],c_npendientes[i_i*2+1])
		// Visualizar_d(pd_pendientes[i_i]);
	// }
	
	// // También tengo que visualizar mi vector trabajado.
	
	// printf("Este es mi vector antes de evolucionarlo \n")
	// Visualizar_d(pd_inicial);
	
	// printf("Este es mi vector luego de evolucionarlo \n");
	// Visualizar_d(pd_vec);
	


	// // Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	// free(pd_inicial);
	// for(int register i_i=0; i_i<5; i_i++) free(pd_pendientes[i_i]);
	// free(pd_pendientes);
	
	// return 0;
// }

// // Esta es una función 
// double Duplicar(struct Red var, struct Parametros par){
	// // Defino mi variable a usar
	// double doble;
	
	// // Esta ecuación diferencial me define las pendientes como el doble del valor actual.
	// doble = 2*var.pd_Opi[var.i_agente*var.pd_Opi[1]+var.i_topico+2];
		
	// return doble,
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este programa lo usé el 21/12/2020 para probar mi función de RK4. La estoy probando
usando la función Duplicar. Esta función lo que hace es determinarme una ecuación
diferencial en la cual la derivada es igual a un múltiplo del valor actual de
la función. Si lo corrés lo que va a hacer esto es printearte los valores
calculados de los k y el valor inicial y final del vector a evolucionar.
Esta probado, funciona bárbaro la función RK4




// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// double *pd_Opi; // Vector de opinión de cada individuo
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
// } s_Red;

// typedef struct Parametros{
	// float f_Beta; // Exponente que regula el grado de homofilia
	// float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	// float f_K; // Influencia social
	// float f_alfa; // Controversialidad de los temas
	// int i_N; // Número de agentes en la red
	// int i_T; // Cantidad de tópicos
	// int i_Mopi; // Máximo inicial de opiniones
	// float f_dt; // Paso temporal de iteración del sistema
// } s_Param;

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// int GenerarOpi(s_Red var, s_Param par);
// int GenerarAng(s_Red var);
// int GenerarAdy(s_Red var,s_Param par);
// double Din1(s_Red var, s_Param par, int i_j);
// double Din2(s_Red var, s_Param par);
// double Duplicar(s_Red var, s_Param par);
// int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par));



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// // ---------------------------------------------------------------------------------------------------

// // Vamos a intentar ver como se leen archivos y data desde estos
// // Creo que lo más útil va a ser usar el fscanf porque mis datos están separados por tabulaciones
// // Acá está la forma de leer el archivo. Sólo necesitaría poder saber que tan largo es antes de
// // subirlo, pero creo que eso sería mucho pedir.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	
	// // Defino los nombres de mis structs en este programa y los valores de los parámetros
	// s_Red red;
	// s_Param datos;
	
	// datos.f_Beta = 0.5;
	// datos.f_Pint = 0.5;
	// datos.f_K = 0.5;
	// datos.f_alfa = 2;
	// datos.i_N = 3;
	// datos.i_T = 3;
	// datos.i_Mopi = 3;
	// datos.f_dt = 0.2;
	
	// // Defino el puntero a mi función y lo inicializo
	// double (*pf_Dup)(s_Red Red, s_Param Parametros) = &Duplicar;	
	
	// // Asigno memoria a mis matrices y las inicializo
	
	// // Matriz de Vectores de opinión. Tiene tamaño N*T
	// red.pd_Opi = (double*) malloc((datos.i_N*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Opi[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Opi[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Opi[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarOpi(red,datos); // Genero la red de Vectores de opinión propiamente dicha.

	// // Matriz de Superposición. Tiene tamaño T*T
	// red.pd_Ang = (double*) malloc((datos.i_T*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_T*datos.i_T+2; i_i++) red.pd_Ang[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Ang[0] = datos.i_T; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Ang[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarAng(red); // Genero la red de Vectores de opinión propiamente dicha.
	
	// // Matriz de Adyacencia. Tiene tamaño N*N
	// red.pi_Ady = (int*) malloc((datos.i_N*datos.i_N+2)*sizeof(int)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_N+2; i_i++) red.pi_Ady[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pi_Ady[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pi_Ady[1] = datos.i_N; // Guardo el número de columnas en la segunda coordenada del vector
	// GenerarAdy(red,datos); // Genero la red de Adyacencia propiamente dicha.
	
	// // Aplico el RK4
	// RK4(red.pd_Opi, red, datos, pf_Dup);
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(red.pi_Ady);
	// free(red.pd_Ang);
	// free(red.pd_Opi);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }




// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Opi[0];
	// i_Columnas = (int) var.pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(int i_i=0;i_i<i_Filas*i_Columnas;i_i++) var.pd_Opi[i_i+2] = par.i_Mopi*(Random()-0.5)*2;
	// return 0;
// }

// // Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
// int GenerarAng(s_Red var){
	// // Obtengo las dimensiones de la matriz de Superposicion.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Ang[0];
	// i_Columnas = (int) var.pd_Ang[1];
	
	// // Inicializo la matriz de Superposicion de mi sistema.
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=0; i_j<i_i; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = 0.5; // Decidí poner 0.5 entre todos los tópicos de mi modelo
	// for(int i_i=0; i_i<i_Filas; i_i++) var.pd_Ang[i_i*i_Columnas+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = var.pd_Ang[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// return 0;
// }


// // Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
// int GenerarAdy(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la matriz de Adyacencia.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pi_Ady[0];
	// i_Columnas = (int) var.pi_Ady[1];
	
	// // Inicializo la matriz de Adyacencia de mi sistema
	// for(int i_i=0; i_i<i_Filas; i_i++) {
		// for(int i_j=0; i_j<i_i; i_j++) {
			// // Saco un número random y veo si asigno un 1 o un 0 a esa posición-
			// if(Random()<par.f_Pint) var.pi_Ady[i_i*i_Columnas+i_j+2] = 1;
			// else var.pi_Ady[i_i*i_Columnas+i_j+2] = 0;
		// }
	// }
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pi_Ady[i_i*i_Columnas+i_j+2] = var.pi_Ady[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// // for(int i=0; i<par.N; i++) var.Ady[i*par.N+i] = 1; // Esto me pone 1 en toda la diagonal
	// return 0;
// }

// // #####################################################################################

// // Estas son las funciones del avance
// // #######################################################################################

// // Esta función resuelve un término de los de la sumatoria.
// double Din1(s_Red var, s_Param par,int i_j){
	
	// // Defino las variables locales de mi función. d_resultados es el return. 
	// // d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	// double d_resultado,d_opiniones_superpuestas = 0;
	
	// // Obtengo el tamaño de columnas de mis tres matrices
	// int i_Co,i_Cs,i_Ca;
	// i_Co = (int) var.pd_Opi[1];
	// i_Cs = (int) var.pd_Ang[1];
	// i_Ca = var.pi_Ady[1];
	
	// for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += var.pd_Ang[var.i_topico*i_Cs+i_p+2]*var.pd_Opi[i_j*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	// d_resultado = var.pi_Ady[var.i_agente*i_Ca+i_j+2]*tanh(par.f_alfa*d_opiniones_superpuestas); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	// return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
// }

// // Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
// double Din2(s_Red var, s_Param par){
	
	// // Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	// double d_resultado,d_sumatoria=0;
	
	// // Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// for(register int i_j=0; i_j<var.i_agente; i_j++) d_sumatoria += Din1(var,par,i_j);
	// for(register int i_j=var.i_agente+1; i_j<par.i_N; i_j++) d_sumatoria += Din1(var,par,i_j);
	
	// // Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	// int i_Columnas = (int) var.pd_Opi[1];
	// d_resultado = -var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2]+par.f_K*d_sumatoria;
	// return d_resultado;
// }

// // #######################################################################################


// // Esta función me realiza una iteración del sistema a partir de un RK4.
// // Creo que no hay forma de que esta función sea 100% general para cualquier caso.
// // Incluso creo que no es necesario el uso del puntero de funciones, pero quiero aprender a usarlo
// int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par)){
	// // Defino las variables y vectores que voy a necesitar
	// int i_filas = (int) *pd_vec; // Este es el número de filas del vector principal
	// int i_columnas = (int) *(pd_vec+1); // Este es el número de columnas del vector principal
	// double *pd_pendientes[5]; // Este array de punteros guarda los punteros kN
	// float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	
	// // Voy a armar vectores con las pendientes que calcule, para luego calcular el avance de todas las variables al final
	// double *k0;
	// k0 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k1;
	// k1 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k2;
	// k2 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k3;
	// k3 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k4;
	// k4 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	// pd_inicial = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// // Inicializo mis punteros
	// pd_pendientes[0] = k0;
	// pd_pendientes[1] = k1;
	// pd_pendientes[2] = k2;
	// pd_pendientes[3] = k3;
	// pd_pendientes[4] = k4;
	// for(register int i_i=0; i_i<i_filas*i_columnas+2; i_i++){
		// *(pd_inicial+i_i) = *(pd_vec+i_i);
		// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+i_i) = 0; // De esta manera inicializo los 5 punteros, todos en una sola línea.
	// }
	
	// // Guardo registro del tamaño de mis vectores k
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]) = i_filas;
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+1) = i_columnas;
	
	// // Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	// DT[0] = 0;
	// DT[1] = par.f_dt*0.5;
	// DT[2] = par.f_dt*0.5;
	// DT[3] = par.f_dt;
		
	// // Acá hago las iteraciones del RK4 para hallar las pendientes k
	// for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// for(var.i_agente=0; var.i_agente<*pd_vec; var.i_agente++){ // Itero para todos los agentes
			// for(var.i_topico=0; var.i_topico<*(pd_vec+1); var.i_topico++){ // Itero para todos los tópicos
				// // Calculo el elemento de la pendiente k(i_j+1)
				// for(register int i_i=0; i_i<i_filas*i_columnas; i_i++) *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes[i_j]+i_i+2)*DT[i_j];
				// *(pd_pendientes[i_j+1]+var.i_agente*i_columnas+var.i_topico+2) = (*fp_funcion)(var,par);
			// }
		// }
	// }
	
	// // Con TODAS las pendientes calculadas, ahora sí avanzo temporalmente mi sistema
	
	// for(register int i_i=0; i_i<i_filas*i_columnas; i_i++){
	// *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+(par.f_dt/6)*(*(pd_pendientes[1]+i_i+2)+*(pd_pendientes[2]+i_i+2)*2+*(pd_pendientes[3]+i_i+2)*2+*(pd_pendientes[4]+i_i+2));
	// }
	
	// // Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// // Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// char c_npendientes[] = "k1234"; //
	
	// for(register int i_i=0; i_i<4; i_i++){
		// printf("\n Estas son las pendientes %c%c \n",c_npendientes[0],c_npendientes[i_i+1]);
		// Visualizar_d(pd_pendientes[i_i+1]);
	// }
	
	// // También tengo que visualizar mi vector trabajado.
	
	// printf("Este es mi vector antes de evolucionarlo \n");
	// Visualizar_d(pd_inicial);
	
	// printf("Este es mi vector luego de evolucionarlo \n");
	// Visualizar_d(pd_vec);
	

	// // Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	// free(pd_inicial);
	// for(int register i_i=0; i_i<5; i_i++) free(pd_pendientes[i_i]);
	
	// return 0;
// }

// // Esta es una función 
// double Duplicar(s_Red var, s_Param par){
	// // Defino mi variable a usar
	// double doble;
	// int i_Columnas = var.pd_Opi[1];
	
	// // Esta ecuación diferencial me define las pendientes como el doble del valor actual.
	// doble = 5*var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2];
		
	// return doble;
// }

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

30/12/2020

Este programa lo hice hace unos días, lo hice para poder probar cómo se hacía para
guardar los datos que el programa iba calculando en un archivo externo. Tuve que
usar el puntero a un FILE, fopen, scanf, fprintf y fclose. Lo interesante es
que por algún motivo que desconozco, el programa ahora tiene el problema de que
no me permite ponerle a los archivos la extensión txt. Por algún motivo que desconozco,
ya eso no lo acepta. Igual a Python no le afecta que sean txt o no, así que eso por
no es un problema. También aproveché para crear unas funciones que reciben
un vector int o un vector double y el puntero al archivo y con eso te escriben
el vector en tu archivo.





// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// double *pd_Opi; // Vector de opinión de cada individuo
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
// } s_Red;

// typedef struct Parametros{
	// float f_Beta; // Exponente que regula el grado de homofilia
	// float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	// float f_K; // Influencia social
	// float f_alfa; // Controversialidad de los temas
	// int i_N; // Número de agentes en la red
	// int i_T; // Cantidad de tópicos
	// int i_Mopi; // Máximo inicial de opiniones
	// float f_dt; // Paso temporal de iteración del sistema
// } s_Param;

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// int GenerarOpi(s_Red var, s_Param par);
// int GenerarAng(s_Red var);
// int GenerarAdy(s_Red var,s_Param par);
// double Din1(s_Red var, s_Param par, int i_j);
// double Din2(s_Red var, s_Param par);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Escribir_i(int *pi_vec, FILE *pa_archivo);
// int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par));



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// // ---------------------------------------------------------------------------------------------------

// // Vamos a intentar ver como se leen archivos y data desde estos
// // Creo que lo más útil va a ser usar el fscanf porque mis datos están separados por tabulaciones
// // Acá está la forma de leer el archivo. Sólo necesitaría poder saber que tan largo es antes de
// // subirlo, pero creo que eso sería mucho pedir.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
		
	// // Defino los nombres de mis structs en este programa y los valores de los parámetros
	// s_Red red;
	// s_Param datos;
	
	// datos.f_Beta = 0.5;
	// datos.f_Pint = 0.5;
	// datos.f_K = 0.5;
	// datos.f_alfa = 2;
	// datos.i_N = 50;
	// datos.i_T = 2;
	// datos.i_Mopi = 3;
	// datos.f_dt = 0.2;
	
	// // Defino el puntero a mi función y lo inicializo
	// double (*pf_Dup)(s_Red Red, s_Param Parametros) = &Din2;	
	
	// // Asigno memoria a mis matrices y las inicializo
	
	// // Matriz de Vectores de opinión. Tiene tamaño N*T
	// red.pd_Opi = (double*) malloc((datos.i_N*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Opi[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Opi[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Opi[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarOpi(red,datos); // Genero la red de Vectores de opinión propiamente dicha.

	// // Matriz de Superposición. Tiene tamaño T*T
	// red.pd_Ang = (double*) malloc((datos.i_T*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_T*datos.i_T+2; i_i++) red.pd_Ang[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Ang[0] = datos.i_T; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Ang[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarAng(red); // Genero la red de Vectores de opinión propiamente dicha.
	
	// // Matriz de Adyacencia. Tiene tamaño N*N
	// red.pi_Ady = (int*) malloc((datos.i_N*datos.i_N+2)*sizeof(int)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_N+2; i_i++) red.pi_Ady[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pi_Ady[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pi_Ady[1] = datos.i_N; // Guardo el número de columnas en la segunda coordenada del vector
	// GenerarAdy(red,datos); // Genero la red de Adyacencia propiamente dicha.
	
	// // Defino las variables para mi archivo.txt que guarda los datos del sistema
	// char filename[255]; // Siempre lo hago lo más grande posible. Si sobra no es problema.
	// sprintf(filename,"Datos_Evolucion_Opinion_N=%d_T=%d",datos.i_N,datos.i_T); // Uso esto para asignarle el nombre a mi archivo.
	// FILE *pa_txt=fopen(filename,"w"); //Esto me crea un archivo con nombre filename y permiso de escritura. Es a partir del puntero pa_txt que accedo al archivo.
	
	// // Copio en el archivo mi matriz de Adyacencia
	// Escribir_i(red.pi_Ady, pa_txt);
	
	// // Copio en el archivo mi matriz de Superposicion
	// Escribir_d(red.pd_Ang, pa_txt);
	
	// // Copio en el archivo mis vectores de opinion
	// Escribir_d(red.pd_Opi, pa_txt);	
	
	// // Evoluciono mi sistema y luego voy guardando los datos.
	// for(register int i_p=0; i_p<20; i_p++){
		// RK4(red.pd_Opi, red, datos, pf_Dup);
		// Escribir_d(red.pd_Opi, pa_txt);
	// }
		
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_txt); // Con esto cierro mi archivo
	// free(red.pi_Ady);
	// free(red.pd_Ang);
	// free(red.pd_Opi);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }




// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Opi[0];
	// i_Columnas = (int) var.pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(int i_i=0;i_i<i_Filas*i_Columnas;i_i++) var.pd_Opi[i_i+2] = par.i_Mopi*(Random()-0.5)*2;
	// return 0;
// }

// // Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
// int GenerarAng(s_Red var){
	// // Obtengo las dimensiones de la matriz de Superposicion.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Ang[0];
	// i_Columnas = (int) var.pd_Ang[1];
	
	// // Inicializo la matriz de Superposicion de mi sistema.
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=0; i_j<i_i; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = 0.5; // Decidí poner 0.5 entre todos los tópicos de mi modelo
	// for(int i_i=0; i_i<i_Filas; i_i++) var.pd_Ang[i_i*i_Columnas+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = var.pd_Ang[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// return 0;
// }


// // Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
// int GenerarAdy(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la matriz de Adyacencia.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pi_Ady[0];
	// i_Columnas = (int) var.pi_Ady[1];
	
	// // Inicializo la matriz de Adyacencia de mi sistema
	// for(int i_i=0; i_i<i_Filas; i_i++) {
		// for(int i_j=0; i_j<i_i; i_j++) {
			// // Saco un número random y veo si asigno un 1 o un 0 a esa posición-
			// if(Random()<par.f_Pint) var.pi_Ady[i_i*i_Columnas+i_j+2] = 1;
			// else var.pi_Ady[i_i*i_Columnas+i_j+2] = 0;
		// }
	// }
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pi_Ady[i_i*i_Columnas+i_j+2] = var.pi_Ady[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// // for(int i=0; i<par.N; i++) var.Ady[i*par.N+i] = 1; // Esto me pone 1 en toda la diagonal
	// return 0;
// }

// // #####################################################################################

// // Estas son las funciones del avance
// // #######################################################################################

// // Esta función resuelve un término de los de la sumatoria.
// double Din1(s_Red var, s_Param par,int i_j){
	
	// // Defino las variables locales de mi función. d_resultados es el return. 
	// // d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	// double d_resultado,d_opiniones_superpuestas = 0;
	
	// // Obtengo el tamaño de columnas de mis tres matrices
	// int i_Co,i_Cs,i_Ca;
	// i_Co = (int) var.pd_Opi[1];
	// i_Cs = (int) var.pd_Ang[1];
	// i_Ca = var.pi_Ady[1];
	
	// for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += var.pd_Ang[var.i_topico*i_Cs+i_p+2]*var.pd_Opi[i_j*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	// d_resultado = var.pi_Ady[var.i_agente*i_Ca+i_j+2]*tanh(par.f_alfa*d_opiniones_superpuestas); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	// return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
// }

// // Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
// double Din2(s_Red var, s_Param par){
	
	// // Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	// double d_resultado,d_sumatoria=0;
	
	// // Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// for(register int i_j=0; i_j<var.i_agente; i_j++) d_sumatoria += Din1(var,par,i_j);
	// for(register int i_j=var.i_agente+1; i_j<par.i_N; i_j++) d_sumatoria += Din1(var,par,i_j);
	
	// // Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	// int i_Columnas = (int) var.pd_Opi[1];
	// d_resultado = -var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2]+par.f_K*d_sumatoria;
	// return d_resultado;
// }

// // #######################################################################################


// // Esta función me realiza una iteración del sistema a partir de un RK4.
// // Incluso creo que no es necesario el uso del puntero de funciones, pero quiero aprender a usarlo
// int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par)){
	// // Defino las variables y vectores que voy a necesitar
	// int i_filas = (int) *pd_vec; // Este es el número de filas del vector principal
	// int i_columnas = (int) *(pd_vec+1); // Este es el número de columnas del vector principal
	// double *pd_pendientes[5]; // Este array de punteros guarda los punteros kN
	// float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	
	// // Voy a armar vectores con las pendientes que calcule, para luego calcular el avance de todas las variables al final
	// double *k0;
	// k0 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k1;
	// k1 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k2;
	// k2 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k3;
	// k3 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *k4;
	// k4 = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	// pd_inicial = (double*) malloc((i_filas*i_columnas+2)*sizeof(double));
	
	// // Inicializo mis punteros
	// pd_pendientes[0] = k0;
	// pd_pendientes[1] = k1;
	// pd_pendientes[2] = k2;
	// pd_pendientes[3] = k3;
	// pd_pendientes[4] = k4;
	// for(register int i_i=0; i_i<i_filas*i_columnas+2; i_i++){
		// *(pd_inicial+i_i) = *(pd_vec+i_i);
		// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+i_i) = 0; // De esta manera inicializo los 5 punteros, todos en una sola línea.
	// }
	
	// // Guardo registro del tamaño de mis vectores k
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]) = i_filas;
	// for(register int i_j=0; i_j<5;i_j++) *(pd_pendientes[i_j]+1) = i_columnas;
	
	// // Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	// DT[0] = 0;
	// DT[1] = par.f_dt*0.5;
	// DT[2] = par.f_dt*0.5;
	// DT[3] = par.f_dt;
		
	// // Acá hago las iteraciones del RK4 para hallar las pendientes k
	// for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// for(var.i_agente=0; var.i_agente<*pd_vec; var.i_agente++){ // Itero para todos los agentes
			// for(var.i_topico=0; var.i_topico<*(pd_vec+1); var.i_topico++){ // Itero para todos los tópicos
				// // Calculo el elemento de la pendiente k(i_j+1)
				// for(register int i_i=0; i_i<i_filas*i_columnas; i_i++) *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes[i_j]+i_i+2)*DT[i_j];
				// *(pd_pendientes[i_j+1]+var.i_agente*i_columnas+var.i_topico+2) = (*fp_funcion)(var,par);
			// }
		// }
	// }
	
	// // Con TODAS las pendientes calculadas, ahora sí avanzo temporalmente mi sistema
	// for(register int i_i=0; i_i<i_filas*i_columnas; i_i++){
	// *(pd_vec+i_i+2) = *(pd_inicial+i_i+2)+(par.f_dt/6)*(*(pd_pendientes[1]+i_i+2)+*(pd_pendientes[2]+i_i+2)*2+*(pd_pendientes[3]+i_i+2)*2+*(pd_pendientes[4]+i_i+2));
	// }
	
	// // Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// // Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// // char c_npendientes[] = "k1234"; //
	
	// // for(register int i_i=0; i_i<4; i_i++){
		// // printf("\n Estas son las pendientes %c%c \n",c_npendientes[0],c_npendientes[i_i+1]);
		// // Visualizar_d(pd_pendientes[i_i+1]);
	// // }
	
	// // También tengo que visualizar mi vector trabajado.
	
	// // printf("Este es mi vector antes de evolucionarlo \n");
	// // Visualizar_d(pd_inicial);
	
	// // printf("Este es mi vector luego de evolucionarlo \n");
	// // Visualizar_d(pd_vec);
	

	// // Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	// free(pd_inicial);
	// for(int register i_i=0; i_i<5; i_i++) free(pd_pendientes[i_i]);
	
	// return 0;
// }

// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }

// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este programa lo terminé el 01/01/2021. Acá lo que hice fue modificar el RK4, la función Din2 y Din1.
El motivo es para poder realizar interacciones de agentes de a pares, en vez de hacerlos con todos los agentes
con los que tiene conexión. Además armé una nueva función llamada Iteración que comprime todos los for
y el proceso de selección del segundo agente para la evolución de cada sujeto. También agregué una nueva
variable en el struct de Red.
Recordá que antes de esta revisión la función Din1 recibía como input el segundo agente de interacción.
Ahora lo modifiqué para que ese agente se encuentre dentro del s_Red, como mencioné antes.

El RK4 está modificado ahora para que NO evolucione todo el vector que le pasas, sino que evolucione
sólamente una variable del vector. La variable a evolucionar tiene que ser definida por fuera del
RK4 ahora, lo cual es mejor en definitiva.


De paso, el proceso de selección del segundo agente lo hago con comparaciones a través de un while.
Este proceso puede realizar repeticiones innecesarias, si bien es un proceso muy corto. En caso de
tener tiempo, hay una función sin terminar al fondo de este código cuyo objetivo es poder seleccionar
un agente nuevo sin utilizar comparaciones, únicamente fors y algunos vectores. De ser necesario, se
podría intentar usar eso para modificar el proceso de selección y ahorrar algo de tiempo. No sé cuanto
se podría ahorrar igualmente. Esta mejora y otras más, se encuentran anotadas en el archivo de Optimización.txt


// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// double *pd_Opi; // Vector de opinión de cada individuo
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
	// int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
// } s_Red;

// typedef struct Parametros{
	// float f_Beta; // Exponente que regula el grado de homofilia
	// float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	// float f_K; // Influencia social
	// float f_alfa; // Controversialidad de los temas
	// int i_N; // Número de agentes en la red
	// int i_T; // Cantidad de tópicos
	// int i_Mopi; // Máximo inicial de opiniones
	// float f_dt; // Paso temporal de iteración del sistema
// } s_Param;

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// int GenerarOpi(s_Red var, s_Param par);
// int GenerarAng(s_Red var);
// int GenerarAdy(s_Red var,s_Param par);
// double Din1(s_Red var, s_Param par);
// double Din2(s_Red var, s_Param par);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Escribir_i(int *pi_vec, FILE *pa_archivo);
// int RK4(double *pd_vec ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par));
// double Multiplo(s_Red var, s_Param par);
// int Iteracion(s_Red var, s_Param par, double (*pf_Dinamica)(s_Red var, s_Param par));



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// //---------------------------------------------------------------------------------------------------

// // Vamos a intentar ver como se leen archivos y data desde estos
// // Creo que lo más útil va a ser usar el fscanf porque mis datos están separados por tabulaciones
// // Acá está la forma de leer el archivo. Sólo necesitaría poder saber que tan largo es antes de
// // subirlo, pero creo que eso sería mucho pedir.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
		
	// // Defino los nombres de mis structs en este programa y los valores de los parámetros
	// s_Red red;
	// s_Param datos;
	
	// datos.f_Beta = 0.5;
	// datos.f_Pint = 0.5; // Ahora mismo este valor no está haciendo nada
	// datos.f_K = 5;
	// datos.f_alfa = 2;
	// datos.i_N = 20;
	// datos.i_T = 4;
	// datos.i_Mopi = 2;
	// datos.f_dt = 0.2;
	
	// // Defino el puntero a mi función y lo inicializo
	// double (*pf_Dinamica)(s_Red Red, s_Param Parametros) = &Din2;	
	
	// // Asigno memoria a mis matrices y las inicializo
	
	// // Matriz de Vectores de opinión. Tiene tamaño N*T
	// red.pd_Opi = (double*) malloc((datos.i_N*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Opi[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Opi[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Opi[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarOpi(red,datos); // Genero la red de Vectores de opinión propiamente dicha.

	// // Matriz de Superposición. Tiene tamaño T*T
	// red.pd_Ang = (double*) malloc((datos.i_T*datos.i_T+2)*sizeof(double)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_T*datos.i_T+2; i_i++) red.pd_Ang[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pd_Ang[0] = datos.i_T; // Guardo el número de filas en la primer coordenada del vector
	// red.pd_Ang[1] = datos.i_T; // Guardo el número de Columnas en la segunda coordenada del vector
	// GenerarAng(red); // Genero la red de Vectores de opinión propiamente dicha.
	
	// // Matriz de Adyacencia. Tiene tamaño N*N
	// red.pi_Ady = (int*) malloc((datos.i_N*datos.i_N+2)*sizeof(int)); // Asigno espacio de memoria al vector
	// for(register int i_i=0; i_i<datos.i_N*datos.i_N+2; i_i++) red.pi_Ady[i_i] = 0; // Inicializo la matriz para que no haya "basura" en alguna posición
	// red.pi_Ady[0] = datos.i_N; // Guardo el número de filas en la primer coordenada del vector
	// red.pi_Ady[1] = datos.i_N; // Guardo el número de columnas en la segunda coordenada del vector
	// GenerarAdy(red,datos); // Genero la red de Adyacencia propiamente dicha.
	
	// // // Defino las variables para mi archivo.txt que guarda los datos del sistema
	// // char filename[255]; // Siempre lo hago lo más grande posible. Si sobra no es problema.
	// // sprintf(filename,"Datos_Evolucion_Opinion_N=%d_T=%d",datos.i_N,datos.i_T); // Uso esto para asignarle el nombre a mi archivo.
	// // FILE *pa_txt=fopen(filename,"w"); //Esto me crea un archivo con nombre filename y permiso de escritura. Es a partir del puntero pa_txt que accedo al archivo.
	
	// // // Copio en el archivo mi matriz de Adyacencia
	// // Escribir_i(red.pi_Ady, pa_txt);
	
	// // // Copio en el archivo mi matriz de Superposicion
	// // Escribir_d(red.pd_Ang, pa_txt);
	
	// // // Copio en el archivo mis vectores de opinion
	// // Escribir_d(red.pd_Opi, pa_txt);
	
	// printf("Este es mi sistema antes de evolucionarlo\n");
	// Visualizar_d(red.pd_Opi);
	
	// // Evoluciono mi sistema y luego voy guardando los datos.
	// for(register int i_i=0; i_i<5000; ++i_i) Iteracion(red,datos,pf_Dinamica);

	
	// printf("Este es mi sistema final\n");
	// Visualizar_d(red.pd_Opi);
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// // fclose(pa_txt); // Con esto cierro mi archivo
	// free(red.pi_Ady);
	// free(red.pd_Ang);
	// free(red.pd_Opi);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }




// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Opi[0];
	// i_Columnas = (int) var.pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(int i_i=0;i_i<i_Filas*i_Columnas;i_i++) var.pd_Opi[i_i+2] = par.i_Mopi*(Random()-0.5)*2;
	// return 0;
// }

// // Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
// int GenerarAng(s_Red var){
	// // Obtengo las dimensiones de la matriz de Superposicion.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pd_Ang[0];
	// i_Columnas = (int) var.pd_Ang[1];
	
	// // Inicializo la matriz de Superposicion de mi sistema.
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=0; i_j<i_i; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = 0.5*(((-i_i-i_j)%2)*2+1); // Decidí poner 0.5 entre todos los tópicos de mi modelo
	// for(int i_i=0; i_i<i_Filas; i_i++) var.pd_Ang[i_i*i_Columnas+i_i+2] = 1; // Esto me pone 1 en toda la diagonal
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pd_Ang[i_i*i_Columnas+i_j+2] = var.pd_Ang[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// return 0;
// }


// // Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
// int GenerarAdy(s_Red var, s_Param par){
	// // Obtengo las dimensiones de la matriz de Adyacencia.
	// int i_Filas,i_Columnas;
	// i_Filas = (int) var.pi_Ady[0];
	// i_Columnas = (int) var.pi_Ady[1];
	
	// // Inicializo la matriz de Adyacencia de mi sistema
	// for(int i_i=0; i_i<i_Filas; i_i++) {
		// for(int i_j=0; i_j<i_i; i_j++) {
			// // Saco un número random y veo si asigno un 1 o un 0 a esa posición-
			// if(Random()<par.f_Pint) var.pi_Ady[i_i*i_Columnas+i_j+2] = 1;
			// else var.pi_Ady[i_i*i_Columnas+i_j+2] = 0;
		// }
	// }
	// for(int i_i=0; i_i<i_Filas; i_i++) for(int i_j=i_i+1; i_j<i_Columnas; i_j++) var.pi_Ady[i_i*i_Columnas+i_j+2] = var.pi_Ady[i_j*i_Columnas+i_i+2]; // Esta sola línea simetriza la matriz
	// // for(int i=0; i<par.N; i++) var.Ady[i*par.N+i] = 1; // Esto me pone 1 en toda la diagonal
	// return 0;
// }

// // #####################################################################################

// // Estas son las funciones del avance
// //#######################################################################################

// // Esta función resuelve un término de los de la sumatoria.
// double Din1(s_Red var, s_Param par){
	// // Defino las variables locales de mi función. d_resultados es el return. 
	// // d_opiniones_superpuestas es el producto de la matriz de superposición de tópicos con el vector opinión de un agente.
	// double d_resultado,d_opiniones_superpuestas = 0;
	
	// // Obtengo el tamaño de columnas de mis tres matrices
	// int i_Co,i_Cs;
	// i_Co = (int) var.pd_Opi[1];
	// i_Cs = (int) var.pd_Ang[1];
	
	// for(register int i_p=0; i_p<i_Cs; i_p++) d_opiniones_superpuestas += var.pd_Ang[var.i_topico*i_Cs+i_p+2]*var.pd_Opi[var.i_agente2*i_Co+i_p+2]; // Calculo previamente este producto de la matriz con el vector.
	// d_resultado = tanh(par.f_alfa*d_opiniones_superpuestas); // Esto es lo que está dentro de la sumatoria en la ecuación dinámica.
	// return d_resultado; // La función devuelve el número que buscás, no te lo asigna en una variable.
// }

// // Esta es la segunda parte de la ecuación dinámica, con esto puedo realizar una iteración del sistema.
// double Din2(s_Red var, s_Param par){
	// // Defino las variables locales de mi función. d_resultado es lo que voy a returnear.
	// // d_sumatoria es el total de la sumatoria del segundo término de la ecuación diferencial.
	// double d_resultado,d_sumatoria=0;
	
	// // Calculo la sumatoria de la ecuación diferencial. Para esto es que existe la función Din1.
	// // La sumatoria es sobre un sólo agente ahora. La idea es ver que pasa cuando las interacciones se hacen de a pares de manera aleatoria
	// // El sujeto lo voy a definir por afuera de esta función, cosa de que cada iteración del RK4 no me lo cambie.
	// d_sumatoria += Din1(var,par);
	
	// // Obtengo el tamaño de Columnas de mi matriz de Vectores de opinión y calculo el valor del campo que define mi ecuación diferencial
	// int i_Columnas = (int) var.pd_Opi[1];
	// d_resultado = -var.pd_Opi[var.i_agente*i_Columnas+var.i_topico+2]+par.f_K*d_sumatoria;
	// return d_resultado;
// }

// //#######################################################################################


// // Esta función me realiza una iteración del sistema a partir de un RK4.
// // Incluso creo que no es necesario el uso del puntero de funciones, pero quiero aprender a usarlo
// int RK4(double *pd_sistema ,s_Red var, s_Param par, double (*fp_funcion)(s_Red var, s_Param par)){
	// // Defino las variables y vectores que voy a necesitar
	// int i_F = (int) *pd_sistema; // Este es el número de filas del vector principal
	// int i_C = (int) *(pd_sistema+1); // Este es el número de columnas del vector principal
	// float DT[4]; // Esto me ayuda a meter el paso temporal que se usa para calcular cada pendiente.
	
	// double *pd_inicial; // Este me guarda las condiciones iniciales del vector, que las voy a necesitar al calcular cada paso del RK4
	// pd_inicial = (double*) malloc((i_F*i_C+2)*sizeof(double));
	
	// double *pd_pendientes; // Este puntero de doubles me guarda todos los valores de las pendientes k
	// pd_pendientes = (double*) malloc((5+2)*sizeof(double));
	
	// // Inicializo mis punteros
	// for(register int i_i=0; i_i<i_F*i_C+2; i_i++) *(pd_inicial+i_i) = *(pd_sistema+i_i);
	
	// *pd_pendientes = 1;
	// *(pd_pendientes+1) = 5;
	// for(register int i_i=0; i_i<5;++i_i) *(pd_pendientes+i_i+2)=0;
	
	// // Armo mi vector DT. Este hay que armarlo uno por uno, si o si.
	// DT[0] = 0;
	// DT[1] = par.f_dt*0.5;
	// DT[2] = par.f_dt*0.5;
	// DT[3] = par.f_dt;
		
	// // Acá hago las iteraciones del RK4 para hallar las pendientes k
	// for(register int i_j=0; i_j<4; i_j++){ // Esto itera para mis 4 k
		// // Calculo el elemento de la pendiente k(i_j+1)
		// for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2)+*(pd_pendientes+i_j+2)*DT[i_j];
		// *(pd_pendientes+i_j+1+2) = (*fp_funcion)(var,par);
	// }
	
	// // Copio al sistema igual que el inicial para deshacer los cambios que hice en el vector principal al calcular los k
	// for(register int i_i=0; i_i<i_F*i_C; i_i++) *(pd_sistema+i_i+2) = *(pd_inicial+i_i+2);
	// // Ahora que tengo los 4 k calculados, avanzo al sujeto que quiero avanzar.
	// *(pd_sistema+var.i_agente*i_C+var.i_topico+2) = *(pd_inicial+var.i_agente*i_C+var.i_topico+2)+(par.f_dt/6)*(*(pd_pendientes+2)+*(pd_pendientes+3)*2+*(pd_pendientes+4)*2+*(pd_pendientes+5));
	
	
	// // Ahora hagamos algún mecanismo de visualización, para ver que todo esté correctamente calculado. Dios que esto va a ser un bardo.
	// // Primero visualicemos las pendientes. Para eso voy a armar unos strings que poner en el printeo
	
	// // printf("\n Estas son las pendientes\n");
	// // Visualizar_d(pd_pendientes);
	
	// // También tengo que visualizar mi vector trabajado.
	
	// // printf("Este es mi vector antes de evolucionarlo \n");
	// // Visualizar_d(pd_inicial);
	
	// // printf("Este es mi vector luego de evolucionarlo \n");
	// // Visualizar_d(pd_sistema);
	

	// // Libero el espacio de memoria asignado a los punteros de las pendientes y al pd_inicial
	// free(pd_inicial);
	// free(pd_pendientes);
	
	// return 0;
// }

// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }

// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta es una función define la ecuación deiferencial como un múltiplo del valor actual
// double Multiplo(s_Red var, s_Param par){
	// // Defino mis variables a usar
	// int i_C;
	// i_C = var.pd_Opi[1];
	// double multiplo;
	
	// // Esta ecuación diferencial me define las pendientes como el doble del valor actual.
	// multiplo = 2*var.pd_Opi[var.i_agente*i_C+var.i_topico+2];
	
	// return multiplo;
// }


// // Esta función me itera todo el sistema. Está buena para simplemente reemplazarla en el main.
// int Iteracion(s_Red var, s_Param par, double (*pf_Dinamica)(s_Red var, s_Param par)){
	// // Simplemente copio lo que estaba usando antes
	// // Primero itero en los agentes, luego defino el segundo agente y evoluciono al primero interactuando con el segundo.
	// for(var.i_agente=0; var.i_agente<par.i_N; ++var.i_agente){
			// do{
				// var.i_agente2 = rand()%par.i_N;
			// }
			// while(var.i_agente==var.i_agente2);
			// for(var.i_topico=0; var.i_topico<par.i_T; ++var.i_topico) RK4(var.pd_Opi, var, par, pf_Dinamica);
		// }
		
	// return 0;
// }


// Esta función toma un número y me da un número aleatorio DISTINTO al número recibido en un cierto rango.
// Esta es la versión con ints. Para que este programa funcione correctamente tiene que pasar que i_min <= i_input <= i_max
// Además, el min y el max pertenecen al rango.
// Seré sincero, toda esta función es para no meter un if justo en el corazón de la iteración de mi programa.
// Esta función no está terminada, otro día la retomo.
// int Aleatorio_distinto_i(int i_input, int i_min, int i_max){
	// // Defino mis variables a usar
	// unsigned int i_rango = i_max - i_min; // Fijate que el i_rango tiene que ser siempre positivo.
	// // 
	// // Espero que esto no me genere problemas a futuro. Problemas del tipo: "Segmentation Fault" que nunca más descubro dónde ocurren.
	// unsigned int i_aleatorio = (Random()%i_rango);
	
	// // Armo la tira de datos que va a tener todos los números posibles dentro del rango, MENOS el número i_input
	// int *pi_Tira;
	// pi_Tira = (int*) malloc((i_rango-1+2)*sizeof(int));
	// *pi_Tira = 1; // Este es el número de filas de mi vector
	// *(pi_Tira+1) = i_rango-1; // Este es el número de columnas de mi vector
	// for(register int i_i=0; i_i<i_input; ++i_i) *(pi_Tira+i_i+2) = i_i+i_min;
	// for(register int i_i=i_input+1; i_i<i_rango; ++i_i) *(pi_Tira+i_i-1+2) = i_i+i_min;
	
	
	
	
	
	
	// return ;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este programa lo hice el 03/01/2021. La idea era modificar las funciones de Deltax y
la que calculaba la norma de los vectores de manera de poder usarla de forma más general.
Para esto hice que puedan tomar el tamaño de los vectores de las primeras coordenadas
de los vectores. Aproveché para cambiarles los nombres siguiendo un poco las nuevas
reglas de nomenclatura. Les agregué una d al final para señalar que son las versiones
que funcionan sobre vectores double.

Este trabajito lo hice porque como quería implementar un mecanismo de corte en mi
programa principal, quería que este mecanismo use como criterio la diferencia entre
el paso siguiente y el anterior, de manera de que si al iterar el sistema deja de 
evolucionar, el programa corte y avance a otra cosa. Por eso necesitaba calcular
diferencias entre vectores y sus normas.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Escribir_i(int *pi_vec, FILE *pa_archivo);
// int Delta_Vec_d(double *pd_x1, double *pd_x2, double *pd_Dx);
// double Norma_d(double *pd_x);



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// //---------------------------------------------------------------------------------------------------

// // Vamos a intentar ver como se leen archivos y data desde estos
// // Creo que lo más útil va a ser usar el fscanf porque mis datos están separados por tabulaciones
// // Acá está la forma de leer el archivo. Sólo necesitaría poder saber que tan largo es antes de
// // subirlo, pero creo que eso sería mucho pedir.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino los vectores que voy a restar y el que voy a usar para guardar la diferencia. También el tamaño de los vectores y la variable de la norma final.
	// int i_F = 3,i_C = 3; // Esto está bueno para reducir filas, pero no siempre es bueno para la lectura del código
	// double d_norma = 0;
	// double *pd_x1, *pd_x2, *pd_diferencia; // Pensé que no me iba a tomar esto.
	// // Voy a poner los tres punteros en un solo array
	// double *apd_punteros[3];
	// apd_punteros[0] = pd_x1;
	// apd_punteros[1] = pd_x2;
	// apd_punteros[2] = pd_diferencia;
	
	// // Asigno memoria a mis vectores.
	// for(register int i_i=0; i_i<3; ++i_i) apd_punteros[i_i] = (double*) malloc((i_F*i_C+2)*sizeof(double));
	
	// // Inicializo mis vectores
	// for(register int i_i=0; i_i<3; ++i_i) *(apd_punteros[i_i])= i_F;
	// for(register int i_i=0; i_i<3; ++i_i) *(apd_punteros[i_i]+1)= i_C;
	// for(register int i_i=0; i_i<3; ++i_i) for(register int i_j=0; i_j<i_F*i_C; ++i_j) *(apd_punteros[i_i]+i_j+2) = 0;
	
	// // Ahora sí. Voy a poner todos 1 en el primer vector, números que crecen en progresión de a 1 en el segundo vector, y el tercero queda para ser la diferencia
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(apd_punteros[0]+i_i+2) = 1;
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(apd_punteros[1]+i_i+2) = i_i;
	
	// // Voy a visualizarlos para garantizar que están bien
	// printf("Este es el primer vector. Tiene todos 1\n");
	// Visualizar_d(apd_punteros[0]);
	
	// printf("Este es el segundo vector. Tiene numeros que van del 0 al %d\n",i_C*i_F-1);
	// Visualizar_d(apd_punteros[1]);
	
	// printf("Este es el vector de diferencias. Por ahora solo tiene ceros porque simplemente esta inicializado\n");
	// Visualizar_d(apd_punteros[2]);
	
	// // Ahora aplico mis funciones
	// Delta_Vec_d(apd_punteros[0],apd_punteros[1],apd_punteros[2]);
	// d_norma = Norma_d(apd_punteros[2]);
	
	// printf("La norma del vector de diferencias es: %lf\n", d_norma);
	
	// printf("El vector de diferencias ahora cambio a esto\n");
	// Visualizar_d(apd_punteros[2]);
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// for(register int i_i=0; i_i<3; ++i_i) free(apd_punteros[i_i]);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }

// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamaño de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }

// // Esta función me calcula la norma de un vector
// double Norma_d(double *pd_x){
	// // Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamaño de mi vector
	// double d_norm,d_sum = 0;
	// int i_C,i_F;
	// i_F = *pd_x;
	// i_C = *(pd_x+1);
	
	// // Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) d_sum += *(pd_x+i_i+2)*(*(pd_x+i_i+2));
	// d_norm = sqrt(d_sum);
	// return d_norm;
// }

// // Esta función me calcula la diferencia entre dos vectores
// int Delta_Vec_d(double *pd_x1, double *pd_x2, double *pd_Dx){
	// // Compruebo primero que mis dos vectores sean iguales en tamaño
	// if(*pd_x1!=*pd_x2 || *(pd_x1+1)!=*(pd_x2+1) || *pd_x1!=*pd_Dx || *(pd_x1+1)!=*(pd_Dx+1)){
		// printf("Los vectores son de tamaños distintos, no puedo restarlos\n");
		// return 0;
	// }
	
	// // Defino las variables de tamaño de mis vectores
	// int i_C,i_F;
	// i_F = *pd_x1;
	// i_C = *(pd_x2);
	
	// // Defino mi vector intermedio de registro de datos y lo inicializo
	// double *pd_diferencias;
	// pd_diferencias = (double*) malloc((i_C*i_F+2)*sizeof(double));
	// *pd_diferencias = i_F;
	// *(pd_diferencias+1) = i_C;
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(pd_diferencias+i_i+2) = 0;
	
	// // Calculo la diferencia entre dos vectores
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(pd_diferencias+i_i+2) = *(pd_x1+i_i+2)-*(pd_x2+i_i+2);
	
	// // Me anoto la diferencia en un vector que está en el main del programa, y luego libero el espacio usado.
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) *(pd_Dx+i_i+2) = *(pd_diferencias+i_i+2);
	// free(pd_diferencias);
	// return 0;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este programa lo hice el 07/01/2021. La idea era ver si se podía escribir en dos
archivos de manera simultánea. Efectivamente se puede, se necesitan abrir dos punteros
a FILE diferenciados y abrir los distintos programas por separado. Después de eso, todo
funca bárbaro. Es muy corto, saqué todo lo que no fuera necesario. Así que no tiene
ninguna de las otras funciones.


// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>


// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Voy a abrir un archivo y guardar algo en él
	// char s_Nombre1[255];
	// sprintf(s_Nombre1,"Primer Archivo");
	// FILE *pf_archivo1=fopen(s_Nombre1,"w");
	// fprintf(pf_archivo1,"Estoy escribiendo en el primer archivo\n");
	
	// // Ahora abro un segundo archivo en el mismo código
	// char s_Nombre2[255];
	// sprintf(s_Nombre2,"Segundo Archivo");
	// FILE *pf_archivo2=fopen(s_Nombre2,"w");
	// fprintf(pf_archivo2,"Estoy escribiendo en el segundo archivo\n");
	
	// for(register int i_i=0; i_i<10; i_i++){
		// fprintf(pf_archivo1,"%d\n",i_i);
		// fprintf(pf_archivo2,"%d\n",i_i);
	// }
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pf_archivo1);
	// fclose(pf_archivo2);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este programa lo armé el 09/01/2021. La idea era probar el funcionamiento del
argc y del argv para ver cómo se pasan datos a la función a través de línea
de comando. También acá probé el uso de la función strtol. Estas dos cosas
están muy bien explicadas al final de este programa.

// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;

	// // Defino dos variables a las que les voy a pasar input desde línea de comando, y las voy a sumar.
	// int i_var1,i_var2;
	// i_var1 = strtol(argv[1],NULL,10);
	// i_var2 = strtol(argv[2],NULL,10);
	
	// printf("La suma de los numeros que me pasaste es: %d \n",i_var1+i_var2);
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// Voy a explicar acá cómo funca el strtol(const char *str, char **endptr, int base).
// La idea es que uno le pasa el string str. Este string puede tener una parte numérica seguida de una parte
// literal. O podría ser sólo numérica. La cosa es que uno le pasa el str a la primera coordenada. A la segunda
// coordenada le pasa la dirección de un puntero donde la función va a guardar la parte literal sobrante de str.
// Siempre se puede poner NULL ahí en caso de que uno no quiera guardar esa info. El tercer elemento es la
// base numérica a la cual quiere convertir el número. Claramente yo me muevo en base 10, así que eso
// siempre puede ser 10 tranquilamente.

// Expliquemos también argc y *argv. argc es el número de elementos pasados en por input de línea de
// comando al main. En particular el nombre del archivo SIEMPRE se pasa, así que argc siempre es 1 como mínimo.
// Por otro lado, en cada coordenada argv guarda cada uno de los argumentos pasados. argv[0] es el nombre del
// archivo. argv[1] es el primer argumento. argv[2] es el segundo argumento, y así.


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------

NUEVO_PROGRAMA

Este archivo es del 17/01/2021. La idea era simplemente probar cómo era el tema para que
el programa guarde en los archivos de datos mis valores con 12 decimales de precisión.
La cosa es que el sistema venía guardando sólo hasta 6 decimales, lo cual era una cagada
porque si ya tengo hecho el cálculo con datos más precisos, guardar menos decimales es
un desperdicio. Cuestión que ahora estoy guardando más datos, ya vi que simplemente tengo
que poner el punto con el número de decimales y listo.


// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;

	// // Defino unas variables int y doubles para hacer unas cuentas.
	
	// double d_resultado;
	// double d_num1 = 0.453;
	// int i_num1 = 250, i_num2 = 1000;
	
	// char s_archivo1[255];
	// sprintf(s_archivo1,"Prueba");
	// FILE *pa_archivo1=fopen(s_archivo1,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// d_resultado = d_num1/(i_num1*i_num2);
	
	// fprintf(pa_archivo1,"%.12lf\t", d_resultado);
	
	// d_resultado = d_num1/ (double) (i_num1*i_num2);
	
	// fprintf(pa_archivo1,"%lf\t", d_resultado);
	
	// d_resultado = 0.000000001;
	
	// fprintf(pa_archivo1,"%lf\t",d_resultado);
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// Estoy queriendo ver si descubro porqué mis datos de error me guardan sólo hasta
// seis decimales, cuando mis variables son doubles.

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este programa es del 19/01/2021. Lo que hice acá fue probar el cómo armar una función
que implemente un método de corte automático, de manera que cuando la el sistema se
encuentre cerca de un estado estable, deje de iterar. Esto es para reducir el número
de iteraciones innecesarias. La idea es que tiene un parámetro, d_parámetro, que lo
compara con un valor de corte. Si el parámetro es menor que el valor de corte,
el sistema sigue operando una cantidad arbitaria de veces, como para asegurarse
de que no incumpla el valor de corte en el futuro. Algo así como decir que no
sea una estabilidad temporal y el sistema luego vuelva a dispararse. Si no se
vuelve a disparar durante esas iteraciones extra, el sistema deja de correr.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// int Operacion(double* d_parametro);
// int Corte(double* d_parametro);
// double Random();


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;

	// // Defino mi contador de iteraciones, el parámetro de corte y una variable para la cantidad total de iteraciones
	// double d_parametro = rand();
	// int i_resultado, i_iteracionesT;
	
	// // Como el valor inicial de parámetro es random, lo printeo para verlo
	// printf("El numero inicial es %lf\n",d_parametro);
	
	// // Printeo también la cantidad de pasos mínimos que voy a necesitar para que el sistema finalice su proceso
	// i_resultado = (int) ceil(log(1000*d_parametro)/log(5))+50;
	// printf("Voy a tener que iterar mínimo %d pasos\n", i_resultado);
	
	// // Realizo las operaciones aplicando la función de Corte y muestro cuántas veces iteré.
	// i_iteracionesT = Corte(&d_parametro);
	// printf("En total itere %d veces\n", i_iteracionesT);
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }

// // Esta función realiza un proceso que modifica mi parámetro
// int Operacion(double* pd_parametro){
	
	// // Defino un mecanismo para que con una baja probabilidad el parámetro a veces suba en vez de bajar
	// if(Random() < 0.7) *pd_parametro = *pd_parametro/5;
	// else *pd_parametro = *pd_parametro*10;
	// return 0;
// }

// // Esta función gestiona la iteración del sistema y se encarga de cortarlo cuando considera que el sistema llegó a un estado estable.
// int Corte(double* pd_parametro){
	
	// // Defino mi cantidad de pasos totales realizados
	// int i_iteraciones=0, i_contador=0;
	
	// while(i_contador<50){
		// if(i_iteraciones > 0) printf("Se rompio la condicion de corte al hacer %d iteraciones extra\n",i_contador);
		// // Inicializo el contador
		// i_contador = 0;
		
		// // Evoluciono el sistema hasta que se cumpla el criterio de corte
		// do{
			// Operacion(pd_parametro);
			// i_iteraciones +=1;
		// }
		// while(*pd_parametro > 0.001);
		
		// // Ahora evoluciono el sistema una cantidad arbitaria de veces, 500 en este caso. Le pongo como condición que si el sistema
		// // deja de cumplir la condición de corte, deje de evolucionar
		// while(i_contador < 50 && *pd_parametro <= 0.001 ){
			// Operacion(pd_parametro);
			// i_contador +=1;
			// i_iteraciones +=1;
		// }
		// // Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// // Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// // Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	// }
	
	// return i_iteraciones;
// }

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este programa lo terminé de formatear el 29/01/2021. La idea era usarlo para probar las funciones de
lectura y escritura y ver cómo funcionaban. Probé el fgets(), al cual le puse una nota extra al fondo,
probé los modos de apertura de archivos, el ftell(), el fseek(), el fscanf() y algunas cosas más.

Lo último que hice en este archivo fue separar lo mejor posible toda la parte utilizada para crear
el archivo de la parte utilizada para leer el archivo. La idea es por si querés revisarlo después
puedas abrir una parte u otra sin hacer quilombos. Quilombos del tipo: Primero abro el archivo
para escritura y después lo abro para lectura.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Voy a separar la parte de escritura del programa de la parte de lectura. No actives las dos partes juntas
	
	// //#######################################################################################
	// // Escritura
	
	// // Defino un array que tenga unos números y listo. Estos son los números que voy a meter en mi matriz de 3x3.
	// // Armé dos listas para probar guardar ints y para guardar doubles. La idea no es que se armen las dos juntas, pero no generaría un crash eso.
	// int ia_Lista[9];
	// for(int register i_i=0; i_i<9; i_i++) ia_Lista[i_i] = i_i+1;
	// double da_Lista[9];
	// for(int register i_i=0; i_i<9; i_i++) da_Lista[i_i] = 0.3*(i_i+1);
	
	// // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Tabla_Valores_Prueba");
	// FILE *pa_archivo=fopen(s_archivo,"w+x"); // Con esto abro mi archivo y dirijo el puntero a él. Con el +x lo que hago es que no me reescriba el archivo ya creado
	
	// // Primero voy a escribir un nuevo archivo que voy a usar para leer y ver cómo funcionan las funciones de lectura
	// for(int register i_i=0; i_i<3; i_i++){
		// for(int register i_j=0; i_j<3; i_j++) fprintf(pa_archivo,"%lf\t",da_Lista[i_i*3+i_j]);
		// fprintf(pa_archivo,"\n");
	// }
	// //##########################################################################################
	// // Lectura
	
	// // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Tabla_Valores_Prueba");
	// FILE *pa_archivo=fopen(s_archivo,"r"); // Con esto abro mi archivo y dirijo el puntero a él
	
	// // Defino un int y un double para guardar los datos que voy levantando del archivo y un long int para la posición del indicador.
	// // Los *_salida son los que guardan los valores leídos de la matriz.
	// int i_salida = 0;
	// double d_salida = 0;
	// long int li_posicion = -5;
	
	// // Acá lo que hago es ir probando las funciones de ftell() y viendo cuanto avanza el puntero cada vez que leo un número.
	// // También probé el fseek() para ir moviendo el indicador de posición y el uso del while con el fscanf para leer todo el archivo.
	// li_posicion = ftell(pa_archivo);
	// printf("La posicion del indicador es %li\n", li_posicion);
	// // while(fscanf(pa_archivo,"%d",&i_salida) != EOF)	printf("Lei %d \n", i_salida);
	// // fseek(pa_archivo,2,SEEK_SET);
	// fscanf(pa_archivo,"%lf",&d_salida);
	// printf("Lei %.12lf\n",d_salida);
	// li_posicion = ftell(pa_archivo);
	// printf("La posicion del indicador es %li\n", li_posicion);
	
	// //##############################################################################################
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_archivo);
	// // free(pc_Lectura);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }


// Defino ahora un puntero de char para guardar mis datos. Esto lo usé con el fgets(), pero ahora ya no es necesario en este código.
// char * pc_Lectura;
// pc_Lectura = (char*) malloc(20*sizeof(char));

// Haré una explicación acá. La idea es la siguiente. El fgets toma una cantidad de caracteres igual a los definidos
// O hasta encontrarse con un newline o fin de archivo. En ese caso es que el fgets returnea un puntero nulo.
// Por eso es que uso un while para levantarlo, porque de esa manera leo todas las filas hasta llegar al final. Y cada fila
// la leo una vez. AHORA, hasta donde vi, no importa si el puntero a char que yo le paso es más chico que el número de
// caracteres a leer, el tipo escribe en el puntero sin ningún drama. Por tanto, eso es bueno porque te guarda los datos
// que querés, pero es malo porque puede escribir cosas que no querés y armarte alto quilombo. Por otro lado, esto levanta
// chars, no sé si sirva para lo que quiero hacer, porque yo voy a querer levantar doubles. Probemos de nuevo con el
// fscanf.

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este programa lo usé el 31/01/2021. Lo usé para construir dos conjuntos de datos. Estos dos conjuntos
me sirven para medir que tan pesado computacionalmente es realizar ciertas operaciones en el programa.
Lo que hice fue darle una tarea de unas cuentas tontas al programa, hacer que la haga muchas veces y 
separar esa tarea en dos casos. El primer caso, la tarea era un conjunto de cuentas aritméticas simples:
Sumas, restas, multiplicaciones y divisiones. El segundo caso es el mismo tipo de cuentas, pero esta vez
le agrego uan tanh. Mi idea es que como mi precisión en la medición del tiempo es muy mala, los hago 
hacer muchas veces la tarea, cuento cuántas veces pueden hacer la misma tarea en un segundo y me guardo ese
número. Luego, como el proceso tiene variabilidad de tiempo, guardo muchos números para poder hacer un histograma
con eso. Y dicho y hecho, lo hice correr 2000 veces para ambos casos, lo cual claramente tomó 4000 segundos.
Guardé los datos y me armé los histogramas. Lo que se vió es que de media las cuentas simples realizaban 100 ciclos
en un segundo, mientras que las cuentas complejas con tanh realizaban 25. Por tanto, esto me da la confianza que la 
implementación de un programa que interpole datos pre calculados en un archivo externo debería reducir tiempos
de cálculo. Algo que necesito fuertemente.




// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// //##################################################################################
	// // // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Histograma_Ciclos");
	// FILE *pa_archivo=fopen(s_archivo,"w+x"); // Con esto abro mi archivo y dirijo el puntero a él. Con el +x lo que hago es que no me reescriba el archivo ya creado
	
	
	// //##################################################################################
	
	// // En un ratito continuo con el guardado de datos. Primero quiero probar unos temas de cómputo
	// // Voy a hacer que el programa realice la misma tarea muchas veces, guardarme la cantidad de ciclos
	// // que realiza en cada segundo y luego con eso armar un histograma. Por eso voy a separar la parte
	// // del cálculo complejo del cálculo simple
	
	// int i_contador = 0; // Este lo uso para ver la cantidad de ciclos que el programa realiza
	// int i_a = 10;
	// int i_b = -3;
	// int i_c = 4;
	// int i_resultado = 0;
	
	// // Ahora realizo unas cuentas con tanh.
	// // double d_resultado = 0; // Defino el double para guardar en algún lado la cuenta
	// // for(int register i_i=0; i_i<10000000; i_i++) d_resultado = tanh(i_a*(i_b+i_i)+i_c); // Hago la cuenta 100000 veces para que me tome un tiempo medible
	// for(int register i_j=0; i_j<2000; i_j++){
		// i_contador = 0;
		// time(&tt_prin);
		// do{
			// // Primero realizo unas simples cuentas con ints muchas veces
			
			// for(int register i_i=0; i_i<10000000; i_i++) i_resultado = (i_a*(i_b+i_i)+i_c)/2;
			// time(&tt_fin);
			// i_tardanza = tt_fin-tt_prin;
			// i_contador++;
		// }
		// while(i_tardanza<1);
		// fprintf(pa_archivo,"\t%d",i_contador);
	// }
	// fprintf(pa_archivo,"\n"); // Separo los ciclos de mi primer cálculo de los ciclos del segundo
	// //################################################################################
	
	// // Defino mis variables para tomar registro y hacer las cuentas
	// i_contador = 0; // Este lo uso para ver la cantidad de ciclos que el programa realiza
	// i_a = 10;
	// i_b = -3;
	// i_c = 4;
	// double d_resultado = 0; // Defino el double para guardar en algún lado la cuenta
	
	// for(int register i_j=0; i_j<2000; i_j++){
		// i_contador = 0;
		// time(&tt_prin);
		// do{
			// // Segundo realizo la misma cuenta sumando una tanh muchas veces
			
			// for(int register i_i=0; i_i<10000000; i_i++) d_resultado = tanh((i_a*(i_b+i_i)+i_c)/2);
			// time(&tt_fin);
			// i_tardanza = tt_fin-tt_prin;
			// i_contador++;
		// }
		// while(i_tardanza<1);
		// fprintf(pa_archivo, "\t%d",i_contador);
	// }
	
	
	// //#####################################################################################
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_archivo);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar de calcular las tanh\n",i_tardanza);
	
	
	// return 0;
// }

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Este archivo lo armé el 09/02/2021. La idea era armar un archivo de prueba para guardar
doubles y ver hasta cuantos decimales puedo guardar sin problemas. Interesantemente
me guardó hasta 32 decimales y no tuvo errores que yo haya notado. Le hice tomar un 
número y dividirlo por 2 iterativamente y guardar ese resultado. Por lo visto, funcionó bien
guardó los 32 decimales correctamente.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	
	// // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Prueba_Double");
	// FILE *pa_archivo=fopen(s_archivo,"w"); // Con esto abro mi archivo y dirijo el puntero a él. Con el +x lo que hago es que no me reescriba el archivo ya creado
	
	// double d_valor = 1;
	
	// for(register int i=0; i<100; i++){
		// d_valor = d_valor/2;
		// fprintf(pa_archivo,"%.32lf\n",d_valor);
	// }
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_archivo);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
	
	
	// return 0;
// }

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este archivo lo habré terminado el 10/02/2021. Este código crea un archivo llamado
Tabla_Valores_TANH el cual tiene los valores pre calculados de tanh para después
ser levantado y permitir usar una interpolación en vez de un cálculo explícito de la
tanh. El archivo guarda datos para valores del argumento que varían entre -5 y 5
con un paso de 10^(-5). Los datos guardados tienen agregados un -1 al principio
y un 1 al final a propósito, la idea era para aproximar correctamente en caso
de que el argumento a interpolar en el programa final estuviera fuera del
rango [-5,5].



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Tabla_Valores_TANH");
	// FILE *pa_archivo=fopen(s_archivo,"w"); // Con esto abro mi archivo y dirijo el puntero a él. Con el +x lo que hago es que no me reescriba el archivo ya creado
	
	// // Ahora con esto me armo los datos para la interpolación. Voy a guardar 12 decimales.
	// fprintf(pa_archivo,"%lf\t",-1.0);
	// for(register int i_i=0; i_i<1000001; i_i++) fprintf(pa_archivo,"%lf\t",tanh(-5+0.00001*i_i));
		
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_archivo);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Esta función la cree el 17/02/2021 para probar cómo se usan los punteros a struct. Es
bastante directa la cosa, sin mucho drama. Hay que agregar un typedef para el puntero
de struct, sólo por simplicidad. Después al crear el puntero a struct hay que malloquearlo.
Lo lindo es que con el typedef después escribir en las funciones para que reciban el puntero
a struct es una pavada. Y pasarlo es como pasar el struct original.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>

// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Ang; // Matriz de superposición entre tópicos. Tiene tamaño T*T
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// double *pd_Opi; // Vector de opinión de cada individuo
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_topico; // Entero que representa el tópico que estoy mirando. Es un valor que va entre 0 y T-1
// } s_red;

// typedef s_red *ps_red;

// // Acá voy a declarar las funciones.------------------------------------

// double Random();
// int Visualizar_d(double *pd_vec);
// int Visualizar_f(float *pf_vec);
// int Visualizar_i(int *pi_vec);
// double Duplicar(ps_red var);



// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */


// //---------------------------------------------------------------------------------------------------

// // Voy a probar cómo se usan los punteros a struct, escribiendo en ellos, en un puntero de struct
// // Y pasando el puntero a una función simple.

// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	
	// // Defino el puntero a mi struct y lo inicializo.
	// ps_red ps_inicial;
	// ps_inicial = malloc(sizeof(s_red));
	
	// // Malloqueo los punteros dentro de mis structs y los inicializo.
	// ps_inicial->pd_Ang = (double*) malloc((15+2)*sizeof(double));
	// ps_inicial->pd_Opi = (double*) malloc((15+2)*sizeof(double));
	// ps_inicial->pi_Ady = (int*) malloc((15+2)*sizeof(int));
	
	// ps_inicial->pd_Ang[0] = 1;
	// ps_inicial->pd_Opi[0] = 1;
	// ps_inicial->pi_Ady[0] = 1;
	
	// ps_inicial->pd_Ang[1] = 15;
	// ps_inicial->pd_Opi[1] = 15;
	// ps_inicial->pi_Ady[1] = 15;
	
	// for(register int i_i=0; i_i<15; i_i++){
		// ps_inicial->pd_Ang[i_i+2] = i_i;
		// ps_inicial->pi_Ady[i_i+2] = i_i;
		// ps_inicial->pd_Opi[i_i+2] = i_i;
	// }
	
	// ps_inicial->i_agente = 1;
	// ps_inicial->i_topico = 1;
	
	// // Veamos que los punteros están bien escritos
	// printf("Puntero Uno \n");
	// Visualizar_d(ps_inicial->pd_Ang);
	// printf("Puntero Dos  \n");
	// Visualizar_d(ps_inicial->pd_Opi);
	// printf("Puntero Tres \n");
	// Visualizar_i(ps_inicial->pi_Ady);
	
	// // Veamos cómo funca una función que recibe un puntero a struct
	// printf("Doble del puntero Dos\n");
	// for(ps_inicial->i_agente = 1; ps_inicial->i_agente <10; ps_inicial->i_agente++) printf("%d ",(int) Duplicar(ps_inicial));
	// printf("\n");

	
	// // Ejecuto los comandos finales para medir el tiempo.
	// free(ps_inicial->pd_Opi);
	// free(ps_inicial->pd_Ang);
	// free(ps_inicial->pi_Ady);
	// free(ps_inicial);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar\n",i_tardanza);
	// return 0;
// }

// /*####################################################################################
// ####################################################################################
// ####################################################################################
// ####################################################################################
// */

// // De acá en adelante van las funciones a declarar

// // Esta función me genera un número random
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Estas son las funciones de Inicialización.
// // #########################################################################################################


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }

// // Esta función es para observar los vectores float
// int Visualizar_f(float *pf_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pf_vec;
	// i_C = *(pf_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pf_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	
	// return 0;
// }


// // Esta es una función 
// double Duplicar(ps_red var){
	// // Defino mi variable a usar
	// double doble;
	
	// // Esta ecuación diferencial me define las pendientes como el doble del valor actual.
	// doble = 2*var->pd_Opi[var->i_agente+var->i_topico+2];
		
	// return doble;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Este programa lo armé el 19/02/2021 creo. Cuestión que el objetivo del programa era armar
la función Tamaño_Comunidad, la cual mide el tamaño de la componente en la cual un cierto agente
es parte. El programa ya está recontra probado y funca bárbaro. Incluso, tiene una segunda
funcionalidad escrita. La primera funcionalidad era usando la función de Inicializar_Adyacencia,
la cual armaba una matriz para una red con tres componentes claramente separadas. Una de tamaño
7, otra de tamaño 5 y otra de tamaño 8. La idea era probar la función Tamaño_Comunidad con estas
comunidades de tamaños conocidos, y para esto la función trabajaba perfecto
Lo otro que hice con este código, que es lo que quedó armado, es probar cuánto tiempo tardaba el
sistema en armar una red conexa con grado medio 8 y número de agentes 10,100 y 1000. La idea era
ver si la probabilidad de armar redes conexas era baja, y por tanto, el esperar a que el sistema
arme una red sería un factor que ralentizara la ejecución del programa principal. Cuestión que
hasta N=1000 no hay problemas. Ya en N=10000 la cosa se hace muy complicada.

Por último, hablando de la función Tamaño_Comunidad, es una función que recibe el puntero a la
matriz de adyacencia y el numero de uno de los agentes, entonces el programa entra a la matriz
de adyacencia desde ese agente, revisa a todos sus vecinos, luego revisa al resto de los vecinos
y así sin repetir hasta que no queden vecinos sin revisar. Luego va tomando nota de quiénes son
estos vecinos y a la salida devuelve el tamaño de la comunidad. Lo que sí, en el futuro podría
modificar esta función para que me de el puntero de la comunidad y no sólo el tamaño.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>


// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
// }s_Red;

// typedef s_Red *ps_Red;

// typedef struct Parametros{
	// float f_Pint; // Probabilidad de que se forme el enlace de manera aleatoria
	// float f_Gradomedio; // Este es el grado medio de los agentes de la red.
	// int i_N; // Número de agentes en la red
// }s_Param;

// typedef s_Param *ps_Param;


// int Visualizar_i(int *pi_vec);
// double Random();
// int Inicializar_Adyacencia(int *pi_adyacencia);
// int GenerarAdy(ps_Red ps_var, ps_Param ps_par);
// int Tamaño_Comunidad(int *pi_adyacencia,int i_inicial);


// int main(){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino los punteros a Structs
	// ps_Red ps_red1;
	// ps_red1 = malloc(sizeof(s_Red));
	
	// ps_Param ps_datos;
	// ps_datos = malloc(sizeof(s_Param));
	
	// // Defino los parámetros de la red
	// ps_datos->f_Gradomedio = 8;
	// ps_datos->i_N = 1000;
	// ps_datos->f_Pint = ps_datos->f_Gradomedio/(ps_datos->i_N-1);
	// int i_tamaño = 0; // Esto me mide el tamaño del grupo medido en la red.
	// int i_contador = 0; // Esto me sirve para contar cuantas veces se cumple que la red es conexa.
	
	
	// // Defino el puntero a mi matriz, lo malloqueo y lo inicializo
	// ps_red1->pi_Ady = (int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int));
	// ps_red1->pi_Ady[0] = ps_datos->i_N;
	// ps_red1->pi_Ady[1] = ps_datos->i_N;
	// for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N; i_i++) ps_red1->pi_Ady[i_i+2] = 0;
	
	// // Inicializo la matriz de manera aleatoria para ver si se genera una única componente gigante
	// // Calculo el tamaño de un grupo en la red. Si el tamaño es menor que el total de agentes, arranco de nuevo.
	// do{
		// GenerarAdy(ps_red1,ps_datos);
		// i_tamaño = Tamaño_Comunidad(ps_red1->pi_Ady, 0);
		// if(i_tamaño == ps_datos->i_N){
			// i_contador +=1;
			// printf("El tamano de la red es: %d\n", i_tamaño);
			// break;
		// }
		// printf("El número del contador es: %d\n",i_contador);
	// }
	// while(i_contador < 10);
	// printf("El número del contador es: %d\n",i_contador);

	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(ps_red1->pi_Ady);
	// free(ps_red1);
	// free(ps_datos);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);

	// return 0;
// }




// //###########################################################
// //###########################################################
// //###########################################################


// // Funciones

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }

// // Esta función me inicializa una red de 20 agentes conectados en tres grupos, uno de 7 agentes, uno de 5 y uno de 8.
// // Los primeros 7 agentes son un grupo, del 8 al 12 son otro grupo y el último son los agentes del 13 al 20
// int Inicializar_Adyacencia(int *pi_adyacencia){
	// // Defino el tamaño de mi vector y lo guardo en variables
	// int i_C;
	// i_C = *(pi_adyacencia+1);
	
	// // Este es el primer grupo
	
		// // Armo la primer fila
		// *(pi_adyacencia+1+2) = 1; // Conecta con 2
		// *(pi_adyacencia+4+2) = 1; // Conecta con 5
		// *(pi_adyacencia+6+2) = 1; // Conecta con 7
		
		// // Armo la segunda fila
		// *(pi_adyacencia+i_C+2) = 1; // Conecta con 1
		// *(pi_adyacencia+2+i_C+2) = 1; // Conecta con 3
		// *(pi_adyacencia+3+i_C+2) = 1; // Conecta con 4
		// *(pi_adyacencia+4+i_C+2) = 1; // Conecta con 5
		// *(pi_adyacencia+5+i_C+2) = 1; // Conecta con 6
		// *(pi_adyacencia+6+i_C+2) = 1; // Conecta con 7
		
		// // Armo la tercera fila
		// *(pi_adyacencia+1+2*i_C+2) = 1; // Conecta con 2
		// *(pi_adyacencia+3+2*i_C+2) = 1; // Conecta con 4
		// *(pi_adyacencia+4+2*i_C+2) = 1; // Conecta con 5
		
		// // Armo la cuarta fila
		// *(pi_adyacencia+1+3*i_C+2) = 1; // Conecta con 2
		// *(pi_adyacencia+2+3*i_C+2) = 1; // Conecta con 3
		// *(pi_adyacencia+4+3*i_C+2) = 1; // Conecta con 5
		
		// // Armo la quinta fila
		// *(pi_adyacencia+4*i_C+2) = 1; // Conecta con 1
		// *(pi_adyacencia+1+4*i_C+2) = 1; // Conecta con 2
		// *(pi_adyacencia+2+4*i_C+2) = 1; // Conecta con 3
		// *(pi_adyacencia+3+4*i_C+2) = 1; // Conecta con 4
		// *(pi_adyacencia+5+4*i_C+2) = 1; // Conecta con 6
		
		// // Armo la sexta fila
		// *(pi_adyacencia+1+5*i_C+2) = 1; // Conecta con 2
		// *(pi_adyacencia+4+5*i_C+2) = 1; // Conecta con 5
		// *(pi_adyacencia+5+5*i_C+2) = 1; // Conecta con 6
		
		// // Armo la septima fila
		// *(pi_adyacencia+6*i_C+2) = 1; // Conecta con 1
		// *(pi_adyacencia+1+6*i_C+2) = 1; // Conecta con 2
		// *(pi_adyacencia+5+6*i_C+2) = 1; // Conecta con 6
	
	// // Este es el segundo grupo
	
		// // Armo la octava fila
		// *(pi_adyacencia+9+7*i_C+2) = 1; // Conecta con 10
		// *(pi_adyacencia+10+7*i_C+2) = 1; // Conecta con 11
		
		// // Armo la novena fila
		// *(pi_adyacencia+10+8*i_C+2) = 1; // Conecta con 11
		// *(pi_adyacencia+11+8*i_C+2) = 1; // Conecta con 12
		
		// // Armo la decima fila
		// *(pi_adyacencia+7+9*i_C+2) = 1; // Conecta con 8
		// *(pi_adyacencia+10+9*i_C+2) = 1; // Conecta con 11
		
		// // Armo la onceava fila
		// *(pi_adyacencia+7+10*i_C+2) = 1; // Conecta con 8
		// *(pi_adyacencia+8+10*i_C+2) = 1; // Conecta con 9
		// *(pi_adyacencia+9+10*i_C+2) = 1; // Conecta con 10
		// *(pi_adyacencia+11+10*i_C+2) = 1; // Conecta con 12
		
		// // Armo la doceava fila
		// *(pi_adyacencia+8+11*i_C+2) = 1; // Conecta con 9
		// *(pi_adyacencia+10+11*i_C+2) = 1; // Conecta con 11
		
	// // Este es el tercer grupo
	
		// // Armo la treceava fila
		// *(pi_adyacencia+13+12*i_C+2) = 1; // Conecta con 14
		// *(pi_adyacencia+15+12*i_C+2) = 1; // Conecta con 16
		// *(pi_adyacencia+16+12*i_C+2) = 1; // Conecta con 17
		
		// // Armo la catorceava fila
		// *(pi_adyacencia+12+13*i_C+2) = 1; // Conecta con 13
		
		// // Armo la quinceava fila
		// *(pi_adyacencia+15+14*i_C+2) = 1; // Conecta con 16
		
		// // Armo la decimo sexta fila
		// *(pi_adyacencia+12+15*i_C+2) = 1; // Conecta con 13
		// *(pi_adyacencia+14+15*i_C+2) = 1; // Conecta con 15
		// *(pi_adyacencia+16+15*i_C+2) = 1; // Conecta con 17
		
		// // Armo la decimo séptima fila
		// *(pi_adyacencia+12+16*i_C+2) = 1; // Conecta con 13
		// *(pi_adyacencia+15+16*i_C+2) = 1; // Conecta con 16
		// *(pi_adyacencia+17+16*i_C+2) = 1; // Conecta con 18
		// *(pi_adyacencia+18+16*i_C+2) = 1; // Conecta con 19
		// *(pi_adyacencia+19+16*i_C+2) = 1; // Conecta con 20
		
		// // Armo la decimo octava fila
		// *(pi_adyacencia+16+17*i_C+2) = 1; // Conecta con 17
		// *(pi_adyacencia+18+17*i_C+2) = 1; // Conecta con 19
		// *(pi_adyacencia+19+17*i_C+2) = 1; // Conecta con 20
		
		// // Armo la decimo novena fila
		// *(pi_adyacencia+16+18*i_C+2) = 1; // Conecta con 17
		// *(pi_adyacencia+17+18*i_C+2) = 1; // Conecta con 18
		
		// // Armo la vigésima fila
		// *(pi_adyacencia+16+19*i_C+2) = 1; // Conecta con 17
		// *(pi_adyacencia+17+19*i_C+2) = 1; // Conecta con 18
		
	// // Con eso están armados los tres grupos
	
	// return 0;
// }

// // Esta función me inicializa la matriz de Adyacencia. Esto es una matriz de N*N
// int GenerarAdy(ps_Red ps_var, ps_Param ps_par){
	// // Obtengo las dimensiones de la matriz de Adyacencia.
	// int i_F,i_C;
	// i_F = (int) ps_var->pi_Ady[0];
	// i_C = (int) ps_var->pi_Ady[1];
	
	// // Inicializo la matriz de Adyacencia de mi sistema
	// for(register int i_i=0; i_i<i_F; i_i++) {
		// for(register int i_j=0; i_j<i_i; i_j++) {
			// // Saco un número random y veo si asigno un 1 o un 0 a esa posición
			// if(Random()< ps_par->f_Pint) ps_var->pi_Ady[i_i*i_C+i_j+2] = 1;
			// else ps_var->pi_Ady[i_i*i_C+i_j+2] = 0;
		// }
	// }
	// for(register int i_i=0; i_i<i_F; i_i++) for(register int i_j=i_i+1; i_j<i_C; i_j++) ps_var->pi_Ady[i_i*i_C+i_j+2] = ps_var->pi_Ady[i_j*i_C+i_i+2]; // Esta sola línea simetriza la matriz
	// for(register int i_i=0; i_i<i_F; i_i++) ps_var->pi_Ady[i_i*i_C+i_i+2] = 0; // Esto me pone 0 en toda la diagonal
	// return 0;
// }

// // Esta función me mide el tamaño del grupo al cual pertenece el nodo inicial i_inicial
// int Tamaño_Comunidad(int *pi_adyacencia, int i_inicial){
	// // Defino la variable del tamaño del grupo, el número de filas de la matriz de Adyacencia, el número de agentes
	// // restantes por visitar; y los inicializo
	// int i_tamaño, i_F, i_restantes;
	// i_tamaño = 0;
	// i_F = *pi_adyacencia;
	// i_restantes = 0;
	
	// // Defino un puntero que registre cuáles agentes están conectados y lo inicializo
	// int *pi_Grupo;
	// pi_Grupo = (int*) malloc((2+i_F)*sizeof(int));
	
	// *pi_Grupo = 1;
	// *(pi_Grupo+1) = i_F;
	// for(register int i_i=0; i_i<i_F; i_i++) *(pi_Grupo+i_i+2) = 0;
	
	// // Defino un puntero que me marque los nuevos sujetos que visitar. Lo hago de tamaño i_F para poder asignar un 1 al visitar el agente en cada posición correcta.
	// int *pi_Visitar;
	// pi_Visitar = (int*) malloc((2+i_F)*sizeof(int));
	
	// *pi_Visitar = 1;
	// *(pi_Visitar+1) = i_F;
	// for(register int i_i=0; i_i<i_F; i_i++) *(pi_Visitar+i_i+2) = 0;
	
	// // Empiezo recorriendo la matriz desde un nodo inicial, que será el primero siempre.
	// for(register int i_i=0; i_i<i_F; i_i++){
		// *(pi_Grupo+i_i+2) = *(pi_adyacencia+i_i+i_inicial*i_F+2);
		// *(pi_Visitar+i_i+2) = *(pi_adyacencia+i_i+i_inicial*i_F+2);
	// }
	
	// do{
		// i_restantes = 0;
		// // Primero reviso mi lista de gente por visitar
		// for(register int i_i=0; i_i<i_F; i_i++){
			// // Si encuentro un uno en la lista, reviso esa fila de la matriz de adyacencia. Es decir, la fila i_i
			// if(*(pi_Visitar+i_i+2) == 1){
				// // Si en esa fila encuentro un uno, tengo que agregar eso al grupo y a la lista de Visitar. Pero no siempre.
				// // La idea es: Si el sujeto no estaba marcado en grupo, entonces lo visito y lo marco en el grupo.
				// // Si ya estaba marcado, es porque lo visité o está en mi lista de visitar.
				// // La idea de esto es no revisitar nodos ya visitados.
				// for(register int i_j=0; i_j<i_F; i_j++){
					// if(*(pi_adyacencia+i_j+i_i*i_F+2) == 1){
						// if(*(pi_Grupo+i_j+2) == 0) *(pi_Visitar+i_j+2) = 1; // Esta línea me agrega el sujeto a visitar sólo si no estaba en el grupo
						// *(pi_Grupo+i_j+2) = *(pi_adyacencia+i_j+i_i*i_F+2); // Esta línea me marca al sujeto en el grupo, porque al final si ya había un uno ahí, simplemente lo vuelve a escribir.
					// }
				// }
				// *(pi_Visitar+i_i+2) = 0;
			// }
		// }
		// for(int register i_i=0; i_i<i_F; i_i++) i_restantes += *(pi_Visitar+i_i+2);
	// }
	// while(i_restantes > 0);
	
	// // Finalmente mido el tamaño de mi grupo
	// for(register int i_i=0; i_i<i_F; i_i++) i_tamaño += *(pi_Grupo+i_i+2);
	
	// // Libero las memorias malloqueadas
	// free(pi_Grupo);
	// free(pi_Visitar);
	
	// return i_tamaño;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este programa lo terminé el 06/05/2021. La idea era para armar una función que levante
un archivo de texto y de ahí pase el contenido que sería la matriz de Adyacencia. Además
estuve revisando cómo manipular los strings de manera de armar los nombres de mis archivos
desde adentro del programa en C, cosa de que los archivos no tengan que entrar como input
desde el Bash. En conclusión, usé el fscanf para leer el archivo, simplemente leo hasta final
de archivo o hasta el tamaño total del vector. Y tiene implementado un recurso para detectar si
la matriz de Adyacencia del archivo es más grande o más chica que el vector del programa.
Por otra parte, usé el sprintf para armar el nombre del archivo, entendiendo que simplemente
uso el nombre de tipo de red como input desde Bash, la cantidad de agentes y el Grado
medio también salen de Input y el número de identidad de red lo saco en el programa de C.





// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>


// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
// }s_Red;

// typedef s_Red *ps_Red;

// typedef struct Parametros{
	// int i_N; // Número de agentes en la red
// }s_Param;

// typedef s_Param *ps_Param;


// int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo);
// int Visualizar_i(int *pi_vec);
// double Random();


// int main(int argc, char *argv[]){
	
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino los punteros a Structs
	// ps_Red ps_red1;
	// ps_red1 = malloc(sizeof(s_Red));
	
	// ps_Param ps_datos;
	// ps_datos = malloc(sizeof(s_Param));
	
	// // Defino los parámetros de la red
	// ps_datos->i_N = 1000;
	
	// // Defino el string que recibo del título. Lo puedo armar usando sprintf sobre el s_archivo. La idea es que en
	// // formato escribo el formato de mis archivos y listo. Después le agrego los valores de N, Grado Medio e Identidad
	// // de la red y con eso puedo leerla
	// // char s_archivo[255];
	// // sprintf(s_archivo,"%s_N=%d_Gm=%d.txt",argv[1],ps_datos->i_N,8);
	// // printf("%s\n", s_archivo);
	
	// // Defino el puntero a mi matriz, lo malloqueo y lo inicializo
	// ps_red1->pi_Ady = (int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int));
	// ps_red1->pi_Ady[0] = ps_datos->i_N;
	// ps_red1->pi_Ady[1] = ps_datos->i_N;
	// for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N; i_i++) ps_red1->pi_Ady[i_i+2] = 3;
	
	// // Ahora voy a querer leer un archivo. Para eso armo un puntero a file y lo dirijo al archivo.
	// // Luego con scanf leo mi archivo. También defino un entero para recorrer mi vector y uno para guardar mis
	// // datos leídos
	// char s_archivo[255];
	// sprintf(s_archivo,"./../Programas Python/Prueba.txt");
	// FILE *pa_archivo=fopen(s_archivo,"r"); // Con esto abro mi archivo y dirijo el puntero a él

	// Lectura_Adyacencia(ps_red1->pi_Ady,pa_archivo);
	
	// // Visualizar_i(ps_red1->pi_Ady);
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(ps_red1->pi_Ady);
	// free(ps_red1);
	// free(ps_datos);
	// // fclose(pa_archivo);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);

	// return 0;
// }

// // ###########################################################
// // ###########################################################
// // ###########################################################


// // Funciones

// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }

// // Esta función es la que lee un archivo y me arma la matriz de Adyacencia
// int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo){
	// // Deefino los enteros que voy a usar para leer el archivo y escribir sobre el vector.	
	// int i_indice = 2;
	// int i_salida = 0;
	
	// // Leo la matriz de Adyacencia del archivo y la guardo en el vector de Adyacencia.
	// while(fscanf(pa_archivo,"%d",&i_salida) != EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		// *(pi_vec+i_indice) = i_salida;
		// i_indice++;
	// }
	
	// // Aviso si hubo algún problema.
	// if(fscanf(pa_archivo,"%d",&i_salida) != EOF){
		// printf("La matriz del archivo es mas grande que tu vector\n");
		// return 1;
	// }
	// if(fscanf(pa_archivo,"%d",&i_salida) == EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		// printf("La matriz del archivo es mas chica que el vector\n");
		// return 1;
	// }
	
	// return 0;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Este programa lo armé el 14/05/2021. Es un programa sencillo cuya función es simplemente 
ver cómo comparar strings para algo que necesitaba hacer en el main de la función principal.
Cuestión que terminé teniendo que usar la función de C strcmp, que compara dos strings
y si son iguales devuelve un cero. A todo esto entonces, agregué la biblioteca
string.h que tiene la función strcmp.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>


// int main(int argc, char *argv[]){
	
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;

	// // Probemos que puede comparar correctamente el argv con un string.
	// char s_input[30];
	// sprintf(s_input,"%s",argv[1]);
	// if(strcmp("Exito",argv[1])==0) printf("Esto funciona correctamente\n");
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);

	// return 0;
// }

//###########################################################

//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Esta función la terminé el 22/05/2021. Acá el objetivo era armar una función que sirva
para interpolar el cálculo de la TANH, de manera de que el programa no tenga que realmente
calcular una TANH, sino que obtenga el valor a partir de una tabla de valores calculada previamente.
Para esto tuve que armar varias funciones, no fue sólo una. Y además mi idea es que para implementar
esto en el programa principal debería agregar nuevos archivos y cosas de manera de que esté bien organizado.

Las cosas que se introducen en este código son:
.) Un struct específico para la tabla de valores de la Tanh.
.) La función: Interpolación. La idea es que es un poco general, por lo que en el futuro
debería poder usarse para otras interpolaciones en caso de ser necesario. Esta va a ir en 
el archivo de general.c.
.) La función Largo_Tabla_TANH. Esta lo que hace es medir la cantidad de números que tengo
guardados en mi tabla, de manera de que si cambio el paso del cálculo de valores de la tabla,
el programa ajuste el tamaño del vector donde se van a guardar de manera automática. La forma en que está
armada es un poco específica según la tabla que tengo, pero podría usarse para otras cosas con algunos ajustes.
La voy a colocar en el archivo TTanh.c (Tabla Tanh).
.) La función Indices_Tabla. Esta lo que hace es tomar el valor del argumento que yo quisiera evaluar en la
tanh y sobreescribir sobre dos ints que yo le pase los valores de los índices del vector del struct en los
cuales se encuentran los valores calculados entre los cuales ese argumento se encuentra encerrado. Esta función
depende de cómo fue armada la tabla, así que tiene que ajustarse a cada tabla nueva que se arme. Va a ir a TTanh.c
.) La función Lectura_Tabla_TANH se encarga de pasar los valores que están en la tabla al vector de doubles del struct.
Esta función también está armada tomando en consideración la tabla de valores de tanh, así que la voy a poner
en TTanh.c, pero podría usarse en un futuro para levantar datos de otras tablas con un pequeño cambio.
.) La función IntrpTANH empaqueta el uso de las funciones Indices_Tabla e Interpolacion de manera de
poder usarla en la función Din1 reemplazando directamente a la función tanh. Esta función por supuesto que
va también en TTanh.c
.) De yapa, me terminé armando dos funciones que calculan máximo y mínimo entre dos valores doubles y devuelve
un double. Creo que se le puede igualmente pasar valores int y la función los auto casteara. Sino se pueden
castear en la entrada a double, y luego castear la salida a int en caso de ser necesario. Estas van a ir
a general.c



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>

// typedef struct Tabla{
	// double* pd_valores; // Este es el vector donde guardo los valores de la tabla
	// int i_largo; // Este es el largo de la tabla.
// } s_Tabla;

// typedef s_Tabla * ps_Tab;

// double Interpolacion(double d_y1, double d_y2, double d_x1,double d_x);
// int Indices_Tabla(int* i_indice1, int* i_indice2, double d_argumento);
// int Largo_Tabla_TANH(FILE *pa_file);
// double IntrpTANH(double d_Arg, ps_Tab ps_val, FILE* pa_file);
// int Lectura_Tabla_TANH(double* pd_vec, FILE* pa_file);
// double Max(double d_a, double d_b);
// double Min(double d_a, double d_b);




// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino el puntero a mi archivo y lo abro
	// char s_archivo[255];
	// sprintf(s_archivo,"Tabla_Valores_TANH");
	// FILE *pa_archivo=fopen(s_archivo,"r"); // Con esto abro mi archivo y dirijo el puntero a él. Con el +x lo que hago es que no me reescriba el archivo ya creado
	
	// // Armo mi puntero a struct
	// ps_Tab ps_tabla;
	// ps_tabla = malloc(sizeof(s_Tabla));
	
	// // Defino el tamaño de mi vector de índices, lo malloqueo y lo inicializo
	// ps_tabla->i_largo = Largo_Tabla_TANH(pa_archivo);
	// ps_tabla->pd_valores = (double*) malloc((ps_tabla->i_largo+2)*sizeof(double));
	// for(int i_i=0; i_i<ps_tabla->i_largo+2;i_i++) ps_tabla->pd_valores[i_i] = 0;
	// ps_tabla->pd_valores[0] = 1; // Cantidad de filas
	// ps_tabla->pd_valores[1] = ps_tabla->i_largo; // Cantidad de columnas
	
	// // Levanto los valores de la Tabla_Valores_TANH y los pongo en mi vector
	// Lectura_Tabla_TANH(ps_tabla->pd_valores,pa_archivo);
	
	// // Comparemos algunos cálculos de tanh con cálculos de Interpolación
	// double da_array[8] = {-5,-4.213485,-3.457286,-2.123457,1.244567,2.124518,3.555555,5};
	
	// for(int i_i=0; i_i<8; i_i++){
		// printf("El resultado de la interpolacion para %lf es %.5lf\n",da_array[i_i],IntrpTANH(da_array[i_i],ps_tabla,pa_archivo));
		// printf("El resultado de la tanh para %lf es %.5lf\n",da_array[i_i],tanh(da_array[i_i]));
	// }
	
	// // Ahora con esto me armo los datos para la interpolación. Voy a guardar 12 decimales.
	// // fprintf(pa_archivo,"%lf\t",-1.0);
	// // for(register int i_i=0; i_i<1000001; i_i++) fprintf(pa_archivo,"%lf\t",tanh(-5+0.00001*i_i));
	// // fprintf(pa_archivo,"%lf\t",1.0);
		
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// free(ps_tabla->pd_valores);
	// free(ps_tabla);
	// fclose(pa_archivo);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }

// //###########################################################

// // Me defino funciones de máximo y mínimo
// double Max(double d_a, double d_b){
	// // Defino la variable a usar
	// double d_max = 0;
	
	// d_max = (d_a > d_b)? d_a : d_b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// // signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// // Sino se devuelve lo que está a la derecha
	
	// return d_max;
// }

// double Min(double d_a, double d_b){
	// // Defino la variable a usar
	// double d_min = 0;
	
	// d_min = (d_a < d_b)? d_a : d_b; // Uso un operador ternario. La idea es que se evalúa la función antes del
	// // signo de pregunta. Si es verdadera, se devuelve lo que está a la izquierda de los dos puntos.
	// // Sino se devuelve lo que está a la derecha
	
	// return d_min;
// }


// // Voy a armar las funciones que voy a necesitar para poder realizar las interpolaciones.
// // Eso sería la función de Interpolación y la función que calcula la posición

// double Interpolacion(double d_y1, double d_y2,double d_x1,double d_x){
	// // Defino las variables que voy a necesitar
	// double d_resultado = 0;
	// double d_deltax = 0.00001;
	// double d_deltay = d_y2-d_y1;
	
	// d_resultado = (d_deltay/d_deltax)*d_x+d_y1+(-d_deltay/d_deltax)*d_x1; // Esta es la cuenta de la interpolación
	
	// return d_resultado;
// }


// // La siguiente función es la que me dice la cantidad de números que hay guardados en el
// // archivo. La idea es que use este número para armar un vector que es el que guarda
// // la posición de cada uno de los números de la tabla

// int Largo_Tabla_TANH(FILE *pa_file){
	// // Defino la variable que voy a returnear
	// int i_largo = 0;
	// double d_salida = 0;
	
	// fseek(pa_file,0,SEEK_SET);
	// while(fscanf(pa_file,"%lf\t",&d_salida)!=EOF) i_largo++;
	// fseek(pa_file,0,SEEK_SET);
	
	// return i_largo;
// }

// // Con esta función ubico el índice que le corresponde a cada número que va entre -5 y 5
// // en el vector de valores de la tabla.

// int Indices_Tabla(int* i_indice1, int* i_indice2, double d_argumento){
	// // Defino las variables que voy a necesitar.
	// double d_indice = 0; // Este es el valor "exacto" del índice que le correspondería al argumento
	
	// d_indice = (d_argumento+5)/0.00001; // Esto sale de despejar la fórmula para calcular los valores de la tabla.
	// // El 0,00001 es el paso con el que armé los valores de la tabla
	
	// d_indice = Max(d_indice,0);
	// d_indice = Min(d_indice,1000003-1);
	
	// *i_indice1 = floor(d_indice);
	// *i_indice2 = ceil(d_indice);
	
	// return 0;
// }


// // Esta función empaqueta el resto para que sólo reciba los elementos básicos, y
// // ya poder pasarlo al main
// double IntrpTANH(double d_Arg, ps_Tab ps_val, FILE* pa_file){
	// // Defino las variables que necesito
	// double d_Y1,d_Y2,d_X1,d_resultado;
	// int i_i1, i_i2;
	// i_i1 = 0;
	// i_i2 = 0;
	
	// Indices_Tabla(&i_i1, &i_i2,d_Arg);
	// d_Y1 = ps_val->pd_valores[i_i1+2];
	// d_Y2 = ps_val->pd_valores[i_i2+2];
	// d_X1 = -5+0.00001*i_i1;
	
	// d_resultado = Interpolacion(d_Y1,d_Y2,d_X1,d_Arg);
	
	
	// return d_resultado;
// }


// // Esta función se encarga de leer la tabla de valores y colocarla en
// // el vector de valores del struct
// int Lectura_Tabla_TANH(double* pd_vec, FILE* pa_file){
	// // Defino mi variable de salida y el entero que recorra los índices
	// double d_salida = 0;
	// int i_indice = 0;
	
	// // Leo los datos de la tabla y los guardo en el vector.
	// while(fscanf(pa_file,"%lf\t",&d_salida)!=EOF){
		// *(pd_vec+i_indice+2)= d_salida;
		// i_indice++;
	// }
	
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Esto lo armé con la ayuda de Seba. La idea es que armé una función llamada Actividad,
la cual recibe un puntero de double, en el cual van a ir los valores de actividad de
cada uno de los agentes. Luego, a ese vector le escribe en cada una de su posiciones
el valor de actividad de cada agente, obteniendo esta actividad de una distribución
de valores de una ley de potencias.


// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>

// double Random();
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);

// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Armo un vector que voy a usar para guardar estos datos y después armar un histograma
	// double* pd_vector;
	// pd_vector = (double*) malloc((2+1000000)*sizeof(double));
	// for(register int i_i=0; i_i<2+1000000;i_i++) *(pd_vector+i_i) = 0;
	// *pd_vector = 1;
	// *(pd_vector+1) = 1000000;
	
	// // Defino mis parámetros y los inicializo
	// double d_epsilon, d_gamma;
	// d_epsilon = 0.01;
	// d_gamma = -2.1;
	
	// // Pruebo de hacer esto un millon de veces y me guardo los datos en un vector.
	// Actividad(pd_vector, d_epsilon, d_gamma);
	
	// // Ahora abro un archivo y guardo estos datos así los levanto con Python y armo un histograma con ellos
	// char s_archivo[255];
	// sprintf(s_archivo,"Distribucion_Powerlaw.txt");
	// FILE* pa_archivo=fopen(s_archivo,"w");
	
	// Escribir_d(pd_vector,pa_archivo);
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// free(pd_vector);
	// fclose(pa_archivo);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }

// // ###########################################################

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Programemos la función que debería devolver un valor de actividad
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia){
	// // Defino las variables que voy a necesitar que son la probabilidad, la probabilidad mínima, La F del Epsilon
	// double d_Feps;
	
	// // Defino acá la cantidad de filas y de columnas de mi vector de double donde voy a guardar las actividades
	// int i_F,i_C;
	// i_F = (int) *pd_vec;
	// i_C = (int) *(pd_vec+1);
	
	// // Hago unas cuentas para tener de antemano
	// d_Feps = pow(d_epsilon,d_potencia+1); // Esto lo hago para no rehacer esta cuenta mil veces en el camino
	
	// for(register int i_i=0; i_i<i_F*i_C;i_i++){
		// *(pd_vec+i_i+2) = pow((1-d_Feps)*Random()+d_Feps,1/(d_potencia+1));
	// }
	
	// return 0;
// }


// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%.6lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Terminé esto el 07/06/2021, este código tiene las funciones necesarias para crear las
matrices de Adyacencia según el modelo de Actividad. Las funciones son:
.) Actividad: A un vector de doubles le asigna los valores de actividad
obtenidos a partir de una distribución de probabilidad que sigue una ley
de potencias. La idea es que el vector de doubles contenga las actividades de cada agente.
.) Adyacencia_Actividad: Limpia la matriz de Adyacencia para eliminar todos
los enlaces. Luego revisa uno por uno a los agentes y los intenta activar.
En caso de activarlos, los conecta usando Conectar_agentes
.) Conectar_agentes: Esta función lo que hace es revisar las uniones que tiene
un agente, hace una lista con todos los agentes con los que no está conectado
y luego samplea una cantidad m de agentes para conectarse con ellos.
Obviamente conectarse significa que escribo sobre la matriz de Adyacencia.

El código básicamente arma las matrices e inicializa los datos de los structs,
luego usa las funciones para asignar actividades y finalmente armar la matriz 
de Adyacencia. Es importante destacar que al activar un agente siempre realiza
m conexiones, de manera que no se permite la opción de que un agente repita
una conexión previamente realizada por otro nodo que se activó. Esto es algo
que no vi aclarado en los papers sobre el modelo. También es importante mencionar
que no está implementado el uso de la homofilia para definir las conexiones.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>


// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_actividad; // Este vector contiene los valores de actividad de cada uno de los agentes.
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
// }s_Red;

// typedef s_Red *ps_Red;

// // El struct de Parametros tiene todos los datos sobre los parámetros del modelo, valga la redundancia
// typedef struct Parametros{
	// int i_N; // Número de agentes en la red
	// int i_m; // Esto sería el número de conexiones que haría para cada agente que se activa.
	// double d_epsilon; // Actividad mínima de los agentes
	// double d_gamma; // Potencia de la distribución de probabilidades de la actividad. Es el valor positivo.
// }s_Param;

// typedef s_Param *ps_Param;




// double Random();
// int Visualizar_i(int *pi_vec);
// int Visualizar_d(double *pd_vec);
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
// int Conectar_agentes(ps_Red ps_red, int i_m);


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino mis punteros a structs
	// ps_Red ps_red;
	// ps_red = malloc(sizeof(s_Red));
	
	// ps_Param ps_datos;
	// ps_datos = malloc(sizeof(s_Param));
	
	// // Inicializo mis datos
	// ps_datos->i_N = 20;
	// ps_datos->i_m = 5;
	// ps_datos->d_epsilon = 0.01;
	// ps_datos->d_gamma = 2.1;
	
	// // Inicializo mis punteros
	// ps_red->pi_Ady =(int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int));
	// for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N+2; i_i++) ps_red->pi_Ady[i_i] = 0;
	// ps_red->pi_Ady[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	// ps_red->pi_Ady[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// ps_red->pd_actividad = (double*) malloc((2+ps_datos->i_N)*sizeof(double));
	// for(register int i_i=0; i_i<2+ps_datos->i_N; i_i++) ps_red->pd_actividad[i_i] = 0;
	// ps_red->pd_actividad[0] = 1;
	// ps_red->pd_actividad[1] = ps_datos->i_N;
	
	// // Asigno las actividades a mis agentes
	// Actividad(ps_red->pd_actividad, ps_datos->d_epsilon, -ps_datos->d_gamma);
	
	// // Visualizo mi red antes de generarla
	// // Visualizar_i(ps_red->pi_Ady);
	
	// // Armo mi matriz de actividad y luego la visualizo
	// Adyacencia_Actividad(ps_red,ps_datos);
	// Visualizar_i(ps_red->pi_Ady);
	// printf("\n");
	
	// // Lo repito sólo para ver que efectivamente la red se desarma en cada nueva iteración
	// // Adyacencia_Actividad(ps_red,ps_datos);
	// // Visualizar_i(ps_red->pi_Ady);
	// // printf("\n");
	
	// // Finalmente escribo mis valores de actividad, como para ver si alguno tenía una actividad alta y eso tiene sentido con lo observado.
	// Visualizar_d(ps_red->pd_actividad);
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// free(ps_red->pi_Ady);
	// free(ps_red->pd_actividad);
	// free(ps_red);
	// free(ps_datos);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }

// //###########################################################

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Programemos la función que debería devolver un valor de actividad
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia){
	// // Defino las variables que voy a necesitar que son la probabilidad, la probabilidad mínima, La F del Epsilon
	// double d_Feps;
	
	// // Defino acá la cantidad de filas y de columnas de mi vector de double donde voy a guardar las actividades
	// int i_F,i_C;
	// i_F = (int) *pd_vec;
	// i_C = (int) *(pd_vec+1);
	
	// // Hago unas cuentas para tener de antemano
	// d_Feps = pow(d_epsilon,d_potencia+1); // Esto lo hago para no rehacer esta cuenta mil veces en el camino
	
	// for(register int i_i=0; i_i<i_F*i_C;i_i++){
		// *(pd_vec+i_i+2) = pow((1-d_Feps)*Random()+d_Feps,1/(d_potencia+1));
	// }
	
	// return 0;
// }


// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%.6lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir a la matriz de adyacencia y la va a armar según la actividad de los agentes
// // Creo que voy a cambiar esto en la función haciendo que directamente reciba los punteros de struct de Red
// // y de Parametros
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos){
	// // Primero armo las variables que voy a necesitar, como el tamaño de mis vectores
	// int i_F,i_C;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	
	// // Desarmo todos los enlaces de la red
	// for(register int i_i=0; i_i<i_F*i_C; i_i++) ps_red->pi_Ady[i_i+2] = 0;
	
	// // Ahora reviso todos los agentes, los intento activar y si se activa lo conecto con m agentes.
	// for(ps_red->i_agente=0; ps_red->i_agente<i_F; ps_red->i_agente++) if(Random()<ps_red->pd_actividad[ps_red->i_agente+2]){
		// printf("Se activo el agente %d\n",ps_red->i_agente);
		// Conectar_agentes(ps_red, ps_datos->i_m);
	// }
	
	// return 0;
// }


// // Esta función recibe la matriz de Adyacencia y el agente, y lo conecta con m agentes.
// int Conectar_agentes(ps_Red ps_red, int i_m){
	// // Defino las variables del tamaño de la matriz y un vector que voy a usar para samplear elementos. Inicializo el vector
	// int i_F,i_C, i_indice;
	// i_indice = 0;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	// int* pi_agentes_libres;
	// pi_agentes_libres = (int*) malloc((i_C+2)*sizeof(int));
	// *pi_agentes_libres = 1;
	// *(pi_agentes_libres+1) = i_C;
	// for(register int i_i=0; i_i<i_C; i_i++) *(pi_agentes_libres+i_i+2) = 0;
	
	// // Reviso la fila de la matriz de adyacencia del agente i_agente, y me guardo los números de los agentes
	// // que no tienen conexión con i_agente. Lo separo en dos fors, porque la idea es saltear el caso
	// // de i_i = ps_red->i_agente
	// for(register int i_i=0; i_i<ps_red->i_agente; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_indice+2) = i_i;
			// i_indice++;
		// }
	// }
	
	// for(register int i_i=ps_red->i_agente+1; i_i<i_C; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_indice+2) = i_i;
			// i_indice++;
		// }
	// }
	
	// // Ahora sampleo m agentes de esta lista. ¿Podría pasar que un agente no tenga 10 sujetos con los
	// // cuales conectarse? Eso requeriría que tenga un grado de 990 para arrancar. Muy poco probable.
	// // Igual después armo una solución para ese caso.
	// if(i_indice>9){
		// for(register int i_i=0; i_i<i_m; i_i++){
			// printf("Los agentes disponibles son: ");
			// Visualizar_i(pi_agentes_libres);
			// ps_red->i_agente2 = rand()%i_indice;
			// ps_red->pi_Ady[ps_red->i_agente*i_F+*(pi_agentes_libres+ps_red->i_agente2+2)+2] = 1;
			// ps_red->pi_Ady[*(pi_agentes_libres+ps_red->i_agente2+2)*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
			// printf("Lo conecte con el agente %d\n",*(pi_agentes_libres+ps_red->i_agente2+2));
			// for(register int i_j=ps_red->i_agente2; i_j<*(pi_agentes_libres+1)-1; i_j++) *(pi_agentes_libres+i_j+2) =  *(pi_agentes_libres+i_j+1+2);
			// i_indice--;
		// }
	// }
	// else for(register int i_i=0; i_i<i_indice; i_i++){
		// ps_red->pi_Ady[ps_red->i_agente*i_F+*(pi_agentes_libres+i_i+2)+2] = 1;
		// ps_red->pi_Ady[*(pi_agentes_libres+i_i+2)*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
	// }
		
	
	// free(pi_agentes_libres);
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Este código lo terminé el 20/06/2021. Acá implementé el uso de la homofilia a la hora de armar
la red de Actividad. Esto no requirió una función nueva, sino el modificar la función de Conectar_agentes.
La idea es simplemente que en la función cuando un agente se activa primero calculo el factor de normalización
de la probabilidad de conexión de agentes según homofilia. Ese factor es la suma de las distancias entre el
agente activado y el resto de los agentes, elevando cada una de las distancias a la -3. Luego hago como antes
y me armo una lista de agentes que voy a usar para las nuevas conexiones. Elijo un agente al azar de la lista y calculo distancia entre
las opiniones de ese agente que quiero conectar y el actual y con eso calculo la probabilidad de homofilia de
que esos agentes se conecten. Hago una tirada random contra esa probabilidad. Si tiene éxito los conecto,
sino busco otro agente y vuelvo a repetir el proceso hasta que alguno se conecte. Así con las m conexiones
para el agente activo.

El programa acá tiene anotadas TODOS los mensajes que usé para poder ir revisando que todo estuviera bien.
Cuando lo implemente voy a borrar esos printf.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>


// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Act; // Este vector contiene los valores de actividad de cada uno de los agentes.
	// double *pd_Opi; // Este es el vector de opiniones del sistema
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
// }s_Red;

// typedef s_Red *ps_Red;

// // El struct de Parametros tiene todos los datos sobre los parámetros del modelo, valga la redundancia
// typedef struct Parametros{
	// int i_N; // Número de agentes en la red
	// int i_T; // Número de tópicos del sistema
	// int i_m; // Esto sería el número de conexiones que haría para cada agente que se activa.
	// int i_Mopi; // Este es el máximo valor de las opiniones iniciales
	// double d_beta; // Este parámetro regula la homofilia del sistema.
	// double d_epsilon; // Actividad mínima de los agentes
	// double d_gamma; // Potencia de la distribución de probabilidades de la actividad. Es el valor positivo.
// }s_Param;

// typedef s_Param *ps_Param;




// double Random();
// int Visualizar_i(int *pi_vec);
// int Visualizar_d(double *pd_vec);
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos);
// int GenerarOpi(ps_Red ps_var, ps_Param ps_par);
// double Norma_d(double *pd_x);


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino mis punteros a structs
	// ps_Red ps_red;
	// ps_red = malloc(sizeof(s_Red));
	
	// ps_Param ps_datos;
	// ps_datos = malloc(sizeof(s_Param));
	
	// // Inicializo mis datos
	// ps_datos->i_N = 20;
	// ps_datos->i_T = 2;
	// ps_datos->i_m = 3;
	// ps_datos->d_epsilon = 0.01;
	// ps_datos->d_gamma = 2.1;
	// ps_datos->d_beta = 3;
	// ps_datos->i_Mopi = 20;
	
	// // Inicializo mis punteros
	// ps_red->pi_Ady =(int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int));
	// for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N+2; i_i++) ps_red->pi_Ady[i_i] = 0;
	// ps_red->pi_Ady[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	// ps_red->pi_Ady[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// ps_red->pd_Act = (double*) malloc((2+ps_datos->i_N)*sizeof(double));
	// for(register int i_i=0; i_i<2+ps_datos->i_N; i_i++) ps_red->pd_Act[i_i] = 0;
	// ps_red->pd_Act[0] = 1;
	// ps_red->pd_Act[1] = ps_datos->i_N;
	
	// ps_red->pd_Opi = (double*) malloc((ps_datos->i_T*ps_datos->i_N+2)*sizeof(double));
	// for(register int i_i=0; i_i<2+ps_datos->i_N*ps_datos->i_T; i_i++) ps_red->pd_Opi[i_i] = 0;
	// ps_red->pd_Opi[0] = ps_datos->i_N;
	// ps_red->pd_Opi[1] = ps_datos->i_T;
	
	// // Asigno las actividades a mis agentes
	// Actividad(ps_red->pd_Act, ps_datos->d_epsilon, -ps_datos->d_gamma);
	
	// // Inicializo mi matriz de opiniones
	// GenerarOpi(ps_red, ps_datos);
	
	// // // Inicializo mi matriz de Adyacencia
	// Adyacencia_Actividad(ps_red, ps_datos);
	
	// // Veamos cómo era la matriz de Opiniones cosa de ver que tal eran las chances de conexiones
	// printf("La matriz de Opiniones era:\n");
	// Visualizar_d(ps_red->pd_Opi);
	
	// // Visualicemos cómo quedó la matriz de Adyacencia
	// printf("La matriz de Adyacencia era:\n");
	// Visualizar_i(ps_red->pi_Ady);
	
	
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// free(ps_red->pi_Ady);
	// free(ps_red->pd_Opi);
	// free(ps_red->pd_Act);
	// free(ps_red);
	// free(ps_datos);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }

// //###########################################################

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Programemos la función que debería devolver un valor de actividad
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia){
	// // Defino las variables que voy a necesitar que son la probabilidad, la probabilidad mínima, La F del Epsilon
	// double d_Feps;
	
	// // Defino acá la cantidad de filas y de columnas de mi vector de double donde voy a guardar las actividades
	// int i_F,i_C;
	// i_F = (int) *pd_vec;
	// i_C = (int) *(pd_vec+1);
	
	// // Hago unas cuentas para tener de antemano
	// d_Feps = pow(d_epsilon,d_potencia+1); // Esto lo hago para no rehacer esta cuenta mil veces en el camino
	
	// for(register int i_i=0; i_i<i_F*i_C;i_i++){
		// *(pd_vec+i_i+2) = pow((1-d_Feps)*Random()+d_Feps,1/(d_potencia+1));
	// }
	
	// return 0;
// }


// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%.6lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir a la matriz de adyacencia y la va a armar según la actividad de los agentes
// // Creo que voy a cambiar esto en la función haciendo que directamente reciba los punteros de struct de Red
// // y de Parametros
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos){
	// // Primero armo las variables que voy a necesitar, como el tamaño de mis vectores
	// int i_F,i_C;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	
	// // Desarmo todos los enlaces de la red
	// for(register int i_i=0; i_i<i_F*i_C; i_i++) ps_red->pi_Ady[i_i+2] = 0;
	
	// // Ahora reviso todos los agentes, los intento activar y si se activa lo conecto con m agentes.
	// for(ps_red->i_agente=0; ps_red->i_agente<i_F; ps_red->i_agente++) if(Random()<ps_red->pd_Act[ps_red->i_agente+2]) Conectar_agentes(ps_red, ps_datos);
	
	// return 0;
// }


// // Esta función recibe la matriz de Adyacencia y el agente, y lo conecta con m agentes.
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos){
	// // Defino las variables del tamaño de la matriz y otras que voy a usar
	// int i_F,i_C, i_indice;
	// i_indice = 0;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	// double d_phomofilia = 0; // Esto va a ser la probabilidad de conectarse de dos agentes según su homofilia
	// double d_distancia = 0; // Este es el valor escalar de distancia entre dos opiniones
	
	
	// // Armo un vector para ver la diferencia entre opiniones. Este vector lo voy a usar primero para calcular el
	// // factor de normalización de la probabilidad de homofilia. Luego lo uso para calcular la probabilidad específica
	// // de cada par de agentes.
	// double* pd_DifOpi;
	// pd_DifOpi = (double*) malloc((ps_datos->i_T+2)*sizeof(double));
	// *pd_DifOpi = 1;
	// *(pd_DifOpi+1) = ps_datos->i_T;
	// for(register int i_i=0; i_i<ps_datos->i_T; i_i++) *(pd_DifOpi+i_i+2) = 0;
	
	// // Calculo el factor de normalización de la probabilidad
	// double d_normalizacionP = 0;
	// for(register int i_agente=0; i_agente < ps_datos->i_N; i_agente++){
		// for(register int i_i=0; i_i<ps_datos->i_T; i_i++) *(pd_DifOpi+i_i+2) = ps_red->pd_Opi[ps_red->i_agente*ps_datos->i_T+i_i+2]-ps_red->pd_Opi[i_agente*ps_datos->i_T+i_i+2];
		// d_distancia = Norma_d(pd_DifOpi);
		// if(d_distancia != 0) d_normalizacionP += pow(d_distancia,-ps_datos->d_beta);
	// }
	// printf("El factor de normalizacion total es %lf\n",d_normalizacionP);
	
	
	// // Armo un vector que voy a usar para samplear elementos y lo inicializo
	// int* pi_agentes_libres;
	// pi_agentes_libres = (int*) malloc((i_C+2)*sizeof(int));
	// *pi_agentes_libres = 1;
	// *(pi_agentes_libres+1) = i_C;
	// for(register int i_i=0; i_i<i_C; i_i++) *(pi_agentes_libres+i_i+2) = 0;
	
	// // Reviso la fila de la matriz de adyacencia del agente i_agente, y me guardo los números de los agentes
	// // que no tienen conexión con i_agente. Lo separo en dos fors, porque la idea es saltear el caso
	// // de i_i = ps_red->i_agente
	// for(register int i_i=0; i_i<ps_red->i_agente; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_indice+2) = i_i;
			// i_indice++;
		// }
	// }
	
	// for(register int i_i=ps_red->i_agente+1; i_i<i_C; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_indice+2) = i_i;
			// i_indice++;
		// }
	// }
	
	// // Ahora sampleo m agentes de esta lista. ¿Podría pasar que un agente no tenga 10 sujetos con los
	// // cuales conectarse? Eso requeriría que tenga un grado de 990 para arrancar. Muy poco probable.
	// // Igual después armo una solución para ese caso.
	// if(i_indice>ps_datos->i_m){
		// int i_contador = 0;
		// while(i_contador<ps_datos->i_m){
			// ps_red->i_agente2 = rand()%i_indice;
			// for(register int i_i=0; i_i<ps_datos->i_T; i_i++) *(pd_DifOpi+i_i+2) = ps_red->pd_Opi[ps_red->i_agente*ps_datos->i_T+i_i+2]-ps_red->pd_Opi[*(pi_agentes_libres+ps_red->i_agente2+2)*ps_datos->i_T+i_i+2];
			// d_distancia = Norma_d(pd_DifOpi);
			// printf("La distancia entre los agentes %d y %d es %lf\n",ps_red->i_agente,*(pi_agentes_libres+ps_red->i_agente2+2),d_distancia);
			// if(d_distancia != 0) d_phomofilia = pow(d_distancia,-ps_datos->d_beta)/d_normalizacionP;
			// else d_phomofilia = 1;
			// printf("Quise conectar a los agentes %d y %d. Su probabilidad era %lf\n",ps_red->i_agente,*(pi_agentes_libres+ps_red->i_agente2+2),d_phomofilia);
			// if(Random()<d_phomofilia){
				// printf("Conecte a los agentes %d y %d\n",ps_red->i_agente,*(pi_agentes_libres+ps_red->i_agente2+2));
				// ps_red->pi_Ady[ps_red->i_agente*i_F+*(pi_agentes_libres+ps_red->i_agente2+2)+2] = 1;
				// ps_red->pi_Ady[*(pi_agentes_libres+ps_red->i_agente2+2)*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
				// for(register int i_j=ps_red->i_agente2; i_j<*(pi_agentes_libres+1)-1; i_j++) *(pi_agentes_libres+i_j+2) =  *(pi_agentes_libres+i_j+1+2);
				// i_indice--;
				// i_contador++;
			// }
		// }
	// }
	// else for(register int i_i=0; i_i<i_indice; i_i++){
		// ps_red->pi_Ady[ps_red->i_agente*i_F+*(pi_agentes_libres+i_i+2)+2] = 1;
		// ps_red->pi_Ady[*(pi_agentes_libres+i_i+2)*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
	// }
		
	
	// free(pi_agentes_libres);
	// free(pd_DifOpi);
	// return 0;
// }

// // Esta función me calcula la norma de un vector
// double Norma_d(double *pd_x){
	// // Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	// double d_norm,d_sum = 0;
	// int i_C,i_F;
	// i_F = *pd_x;
	// i_C = *(pd_x+1);
	
	// // Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) d_sum += *(pd_x+i_i+2)*(*(pd_x+i_i+2));
	// d_norm = sqrt(d_sum);
	// return d_norm;
// }

// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(ps_Red ps_var, ps_Param ps_par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_F,i_C;
	// i_F = (int) ps_var->pd_Opi[0];
	// i_C = (int) ps_var->pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(register int i_i=0;i_i<i_F*i_C;i_i++) ps_var->pd_Opi[i_i+2] = floor(ps_par->i_Mopi*(Random()-0.5)*2);
	// return 0;
// }


//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA


Este programa lo terminé el 25/06/2021. La idea es optimizar el código de Conectar_agentes ya
que cuando lo usé en la función principal, la implementación de la homofilia hacía que el programa
se quedara encerrado en el loop de selección de agentes. Lo que hice acá fue lograr que en vez de tener
que seleccionar un agente, tirar un número Random y ahí ver si los conecto, acá distribuyo a los
agentes en el intervalo de probabilidades entre 0 y 1, tiro el Random una sola vez y según
donde caiga es el agente con el cual me conecto. Esto permite que una sola tirada de Random
me conecte automáticamente, evitando loops en los cuales el problema es que el código no
encuentra con quien conectarse.

Lo copado de esto es que no tengo que hacer una cantidad excesiva de cuentas complejas,
porque la cosa de elevar la distancia entre agentes a la -Beta lo hago una sóla vez aleatoria
principio y luego simplemente tomo esos números, que son doubles y los voy sumando para obtener
el factor de Normalización de la probabilidad. Luego, calculo probabilidades de agente usando
eso y listo.

Respecto a la selección de agentes, es importante destacar que cada vez que un agente es seleccionado
hay que recalcular todas las probabilidades porque ese agente se lleva todo el intervalo que ocupaba.
Pero recalcular las probabilidades es una boludez porque al final del día es sumar y dividir unos doubles,
nada complejo. Y además son N cuentas, nada complejo para el programa. Por otro lado, los agentes cuya
distancia con el agente activado es cero tienen probabilidad 1 de conectarse, así que esos tendrán
la chance de conectarse en caso de que el sistema llegue a ellos. Lo importante es que no aportan
nada al factor de normalización y cuando otros agentes se conectan no generan problemas.

Todo el cambio entre este código y el anterior se encuentra enmarcado en la función Conectar_agentes.



// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>


// // Acá vienen los structs.
// // El struct Red tiene los datos que definen mi red, y la info de la red
// typedef struct Red{
	// double *pd_Act; // Este vector contiene los valores de actividad de cada uno de los agentes.
	// double *pd_Opi; // Este es el vector de opiniones del sistema
	// int *pi_Ady; // Matriz de adyacencia que define mis conexiones. Tiene tamaño N*N
	// int i_agente; // Entero que representa el agente que estoy mirando. Es un valor que va entre 0 y N-1
	// int i_agente2; // Este es el segundo agente con el cual se pone en contacto el primero.
// }s_Red;

// typedef s_Red *ps_Red;

// // El struct de Parametros tiene todos los datos sobre los parámetros del modelo, valga la redundancia
// typedef struct Parametros{
	// int i_N; // Número de agentes en la red
	// int i_T; // Número de tópicos del sistema
	// int i_m; // Esto sería el número de conexiones que haría para cada agente que se activa.
	// int i_Mopi; // Este es el máximo valor de las opiniones iniciales
	// double d_beta; // Este parámetro regula la homofilia del sistema.
	// double d_epsilon; // Actividad mínima de los agentes
	// double d_gamma; // Potencia de la distribución de probabilidades de la actividad. Es el valor positivo.
// }s_Param;

// typedef s_Param *ps_Param;




// double Random();
// int Visualizar_i(int *pi_vec);
// int Visualizar_d(double *pd_vec);
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia);
// int Escribir_d(double *pd_vec, FILE *pa_archivo);
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos);
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos);
// int GenerarOpi(ps_Red ps_var, ps_Param ps_par);
// double Norma_d(double *pd_x);


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Defino mis punteros a structs
	// ps_Red ps_red;
	// ps_red = malloc(sizeof(s_Red));
	
	// ps_Param ps_datos;
	// ps_datos = malloc(sizeof(s_Param));
	
	// // Inicializo mis datos
	// ps_datos->i_N = 20;
	// ps_datos->i_T = 2;
	// ps_datos->i_m = 3;
	// ps_datos->d_epsilon = 0.01;
	// ps_datos->d_gamma = 2.1;
	// ps_datos->d_beta = 3;
	// ps_datos->i_Mopi = 20;
	
	// // Inicializo mis punteros
	// ps_red->pi_Ady =(int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int));
	// for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N+2; i_i++) ps_red->pi_Ady[i_i] = 0;
	// ps_red->pi_Ady[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	// ps_red->pi_Ady[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// ps_red->pd_Act = (double*) malloc((2+ps_datos->i_N)*sizeof(double));
	// for(register int i_i=0; i_i<2+ps_datos->i_N; i_i++) ps_red->pd_Act[i_i] = 0;
	// ps_red->pd_Act[0] = 1;
	// ps_red->pd_Act[1] = ps_datos->i_N;
	
	// ps_red->pd_Opi = (double*) malloc((ps_datos->i_T*ps_datos->i_N+2)*sizeof(double));
	// for(register int i_i=0; i_i<2+ps_datos->i_N*ps_datos->i_T; i_i++) ps_red->pd_Opi[i_i] = 0;
	// ps_red->pd_Opi[0] = ps_datos->i_N;
	// ps_red->pd_Opi[1] = ps_datos->i_T;
	
	// // Asigno las actividades a mis agentes
	// Actividad(ps_red->pd_Act, ps_datos->d_epsilon, -ps_datos->d_gamma);
	
	// // Inicializo mi matriz de opiniones
	// GenerarOpi(ps_red, ps_datos);
	
	// // Fuerzo a dos agentes a tener la misma opinión
	// ps_red->pd_Opi[5*ps_datos->i_T+0+2] = 1;
	// ps_red->pd_Opi[5*ps_datos->i_T+1+2] = 1;
	// ps_red->pd_Opi[3*ps_datos->i_T+0+2] = 1;
	// ps_red->pd_Opi[3*ps_datos->i_T+1+2] = 1;
	
	// // Veamos cómo era la matriz de Opiniones cosa de ver que tal eran las chances de conexiones
	// printf("La matriz de Opiniones era:\n");
	// Visualizar_d(ps_red->pd_Opi);
	
	// // // Inicializo mi matriz de Adyacencia
	// Adyacencia_Actividad(ps_red, ps_datos);
	
	// // Visualicemos cómo quedó la matriz de Adyacencia
	// printf("La matriz de Adyacencia era:\n");
	// Visualizar_i(ps_red->pi_Ady);
	
	
	
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// time(&tt_fin);
	// free(ps_red->pi_Ady);
	// free(ps_red->pd_Opi);
	// free(ps_red->pd_Act);
	// free(ps_red);
	// free(ps_datos);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }

// //###########################################################

// // Esta función me genera un número random entre 0 y 1
// double Random(){
	// return ((double) rand()/(double) RAND_MAX);
// }


// // Esta función es para observar los vectores int
// int Visualizar_i(int *pi_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Esta función es para observar los vectores double
// int Visualizar_d(double *pd_vec){
	// // Defino las variables que voy a necesitar.
	// int i_F,i_C;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Printeo mi vector
	// for(register int i_i=0; i_i<i_F; i_i++){
		// for(register int i_j=0; i_j<i_C; i_j++) printf("%lf\t",*(pd_vec+i_i*i_C+i_j+2));
		// printf("\n");
	// }
	// printf("\n");
	
	// return 0;
// }


// // Programemos la función que debería devolver un valor de actividad
// int Actividad(double* pd_vec, double d_epsilon, double d_potencia){
	// // Defino las variables que voy a necesitar que son la probabilidad, la probabilidad mínima, La F del Epsilon
	// double d_Feps;
	
	// // Defino acá la cantidad de filas y de columnas de mi vector de double donde voy a guardar las actividades
	// int i_F,i_C;
	// i_F = (int) *pd_vec;
	// i_C = (int) *(pd_vec+1);
	
	// // Hago unas cuentas para tener de antemano
	// d_Feps = pow(d_epsilon,d_potencia+1); // Esto lo hago para no rehacer esta cuenta mil veces en el camino
	
	// for(register int i_i=0; i_i<i_F*i_C;i_i++){
		// *(pd_vec+i_i+2) = pow((1-d_Feps)*Random()+d_Feps,1/(d_potencia+1));
	// }
	
	// return 0;
// }


// // Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
// int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pi_vec;
	// i_C = *(pi_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%d",*(pi_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
// int Escribir_d(double *pd_vec, FILE *pa_archivo){
	// // Defino las variables del tamao de mi vector
	// int i_C,i_F;
	// i_F = *pd_vec;
	// i_C = *(pd_vec+1);
	
	// // Ahora printeo todo el vector en mi archivo
	// for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"\t%.6lf",*(pd_vec+i_i+2));
	// fprintf(pa_archivo,"\n");
	
	// return 0;
// }


// // Esta función va a recibir a la matriz de adyacencia y la va a armar según la actividad de los agentes
// // Creo que voy a cambiar esto en la función haciendo que directamente reciba los punteros de struct de Red
// // y de Parametros
// int Adyacencia_Actividad(ps_Red ps_red, ps_Param ps_datos){
	// // Primero armo las variables que voy a necesitar, como el tamaño de mis vectores
	// int i_F,i_C;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	
	// // Desarmo todos los enlaces de la red
	// for(register int i_i=0; i_i<i_F*i_C; i_i++) ps_red->pi_Ady[i_i+2] = 0;
	
	// // Ahora reviso todos los agentes, los intento activar y si se activa lo conecto con m agentes.
	// // for(ps_red->i_agente=0; ps_red->i_agente<i_F; ps_red->i_agente++) if(Random()<ps_red->pd_Act[ps_red->i_agente+2]) Conectar_agentes(ps_red, ps_datos);
	
	// // Voy a probar conectar algunos agentes a la fuerza
	// ps_red->i_agente = 5;
	// Conectar_agentes(ps_red,ps_datos);
	
	// return 0;
// }


// // Esta función recibe la matriz de Adyacencia y el agente, y lo conecta con m agentes.
// int Conectar_agentes(ps_Red ps_red, ps_Param ps_datos){
	// // Defino las variables del tamaño de la matriz y otras que voy a usar
	// int i_F,i_C, i_indice, i_libres;
	// i_indice = 0;
	// i_libres = 0;
	// i_F = ps_red->pi_Ady[0];
	// i_C = ps_red->pi_Ady[1];
	// double d_distancia = 0; // Este es el valor escalar de distancia entre dos opiniones
	// double d_probacumulada = 0; // Esta es la probabilidad acumulada de los agentes
	// double d_proba = 0; // Esta es la probabilidad que lanzo una sola vez
	
	
	// // Armo un vector que voy a usar para samplear elementos y lo inicializo
	// int* pi_agentes_libres;
	// pi_agentes_libres = (int*) malloc((i_C+2)*sizeof(int));
	// *pi_agentes_libres = 1;
	// *(pi_agentes_libres+1) = i_C;
	// for(register int i_i=0; i_i<i_C; i_i++) *(pi_agentes_libres+i_i+2) = 0;
	
	// // Reviso la fila de la matriz de adyacencia del agente i_agente, y me guardo los números de los agentes
	// // que no tienen conexión con i_agente. Lo separo en dos fors, porque la idea es saltear el caso
	// // de i_i = ps_red->i_agente
	// for(register int i_i=0; i_i<ps_red->i_agente; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_libres+2) = i_i;
			// i_libres++;
		// }
	// }
	
	// for(register int i_i=ps_red->i_agente+1; i_i<i_C; i_i++){
		// if(ps_red->pi_Ady[ps_red->i_agente*i_F+i_i+2]==0){
			// *(pi_agentes_libres+i_libres+2) = i_i;
			// i_libres++;
		// }
	// }
	
	// printf("La lista de agentes libres es: \n");
	// Visualizar_i(pi_agentes_libres);
	
	// // Armo un vector para ver la diferencia entre opiniones. Este vector lo voy a usar primero para calcular la
	// // distancia entre agentes y con eso hallar el factor de normalización y sus probabilidades de conexión
	// double* pd_DifOpi;
	// pd_DifOpi = (double*) malloc((ps_datos->i_T+2)*sizeof(double));
	// *pd_DifOpi = 1;
	// *(pd_DifOpi+1) = ps_datos->i_T;
	// for(register int i_i=0; i_i<ps_datos->i_T; i_i++) *(pd_DifOpi+i_i+2) = 0;
	
	
	// // Voy a armar dos vectores más acá el primero es el de Distbeta, el cual me guarda
	// // la distancia de cada agente con el agente ps_red->i_agente elevada a la menos beta.
	// // El segundo ya toma el valor de normalización y a cada una de estas distancias las normaliza
	// // para obtener la probabilidad de interacción por homofilia.
	// double* pd_distbeta;
	// pd_distbeta = (double*) malloc((ps_datos->i_N+2)*sizeof(double));
	// *pd_distbeta = 1;
	// *(pd_distbeta+1) = ps_datos->i_N;
	// for(register int i_i=0; i_i<ps_datos->i_N; i_i++) *(pd_distbeta+i_i+2) = 0;
	
	// double* pd_probabilidades;
	// pd_probabilidades = (double*) malloc((ps_datos->i_N+2)*sizeof(double));
	// *pd_probabilidades = 1;
	// *(pd_probabilidades+1) = ps_datos->i_N;
	// for(register int i_i=0; i_i<ps_datos->i_N; i_i++) *(pd_probabilidades+i_i+2) = 0;
	
	
	// // Calculo todas las distancias y el factor de normalizacion
	// double d_normalizacionP = 0;
	// for(register int i_i=0; i_i<i_libres; i_i++){
		// i_indice = *(pi_agentes_libres+i_i+2);
		// for(register int i_j=0; i_j<ps_datos->i_T; i_j++) *(pd_DifOpi+i_j+2) = ps_red->pd_Opi[ps_red->i_agente*ps_datos->i_T+i_j+2]-ps_red->pd_Opi[i_indice*ps_datos->i_T+i_j+2];
		// d_distancia = Norma_d(pd_DifOpi);
		// if(d_distancia>0) *(pd_distbeta+i_indice+2) = pow(d_distancia,-ps_datos->d_beta);
		// else *(pd_distbeta+i_indice+2) = 0;
	// }
	// for(register int i_i=0; i_i<ps_datos->i_N; i_i++) d_normalizacionP += *(pd_distbeta+i_i+2); // Esta línea me calcula el factor de normalización
	
	// // // Ahora calculo las probabilidades
	// for(register int i_i=0; i_i<i_libres; i_i++){
		// i_indice = *(pi_agentes_libres+i_i+2);
		// for(register int i_j=0; i_j<ps_datos->i_T; i_j++) *(pd_DifOpi+i_j+2) = ps_red->pd_Opi[ps_red->i_agente*ps_datos->i_T+i_j+2]-ps_red->pd_Opi[i_indice*ps_datos->i_T+i_j+2];
		// d_distancia = Norma_d(pd_DifOpi);
		// if(d_distancia>0) *(pd_probabilidades+i_indice+2) = *(pd_distbeta+i_indice+2)/d_normalizacionP;
		// else *(pd_probabilidades+i_indice+2) = 1;
	// }
	
	
	// // Ahora sampleo m agentes de esta lista
	// if(i_libres>ps_datos->i_m){
		// int i_contador = 0;
		// while(i_contador<ps_datos->i_m){
			// printf("El factor de normalizacion es %lf \n", d_normalizacionP);
			// i_indice = 0;
			// d_proba = Random();
			// d_probacumulada = *(pd_probabilidades+2);
			// printf("Las probabilidades son: \n");
			// Visualizar_d(pd_probabilidades);
			// // Defino cuál es el agente que se conecta
			// while(d_proba>d_probacumulada){
				// i_indice++;
				// d_probacumulada += *(pd_probabilidades+i_indice+2);
			// }
			// // Hago la conexión
			// ps_red->pi_Ady[ps_red->i_agente*i_F+i_indice+2] = 1;
			// ps_red->pi_Ady[i_indice*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
			// // Saco al agente del grupo de agentes posibles para conectarse al
			// // poner su distancia como cero y su probabilidad como cero.
			// printf("Conecte al agente %d con el agente %d \n",ps_red->i_agente,i_indice);
			// printf("La probabilidad de conectarlos era %lf \n",*(pd_probabilidades+i_indice+2));
			// printf("El numero random es %lf y la proba acumulada es %lf \n",d_proba,d_probacumulada);
			// *(pd_distbeta+i_indice+2) = 0;
			// *(pd_probabilidades+i_indice+2) = 0;
			// // Calculo el nuevo factor de normalización y las nuevas probabilidades,
			// // ignorando aquellos sujetos cuya distancia es cero. Esto ignora tanto a los
			// // ya conectados como a los que tienen distancia cero.
			// d_normalizacionP = 0;
			// for(register int i_i=0; i_i<ps_datos->i_N; i_i++) d_normalizacionP += *(pd_distbeta+i_i+2);
			// for(register int i_i=0; i_i<ps_datos->i_N; i_i++) if(*(pd_distbeta+i_i+2)!=0) *(pd_probabilidades+i_i+2) = *(pd_distbeta+i_i+2)/d_normalizacionP;
			// // Avanzo el contador para marcar que hice una conexión
			// i_contador++;
		// }
	// }
	// else for(register int i_i=0; i_i<i_libres; i_i++){
		// ps_red->pi_Ady[ps_red->i_agente*i_F+*(pi_agentes_libres+i_i+2)+2] = 1;
		// ps_red->pi_Ady[*(pi_agentes_libres+i_i+2)*i_F+ps_red->i_agente+2] = 1; // Esto se encarga de marcar al sujeto simétrico
	// }
	
	
	// free(pi_agentes_libres);
	// free(pd_DifOpi);
	// free(pd_distbeta);
	// free(pd_probabilidades);
	// return 0;
// }

// // Esta función me calcula la norma de un vector
// double Norma_d(double *pd_x){
	// // Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	// double d_norm,d_sum = 0;
	// int i_C,i_F;
	// i_F = *pd_x;
	// i_C = *(pd_x+1);
	
	// // Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	// for(register int i_i=0; i_i<i_C*i_F; ++i_i) d_sum += *(pd_x+i_i+2)*(*(pd_x+i_i+2));
	// d_norm = sqrt(d_sum);
	// return d_norm;
// }

// // Esta función me genera los vectores opinión iniciales del sistema. Esto es una matriz de tamaño N*T
// int GenerarOpi(ps_Red ps_var, ps_Param ps_par){
	// // Obtengo las dimensiones de la "matriz" de opiniones de mis agentes.
	// int i_F,i_C;
	// i_F = (int) ps_var->pd_Opi[0];
	// i_C = (int) ps_var->pd_Opi[1];
	
	// // Inicializo la "matriz" de opiniones de mis agentes.
	// for(register int i_i=0;i_i<i_F*i_C;i_i++) ps_var->pd_Opi[i_i+2] = floor(ps_par->i_Mopi*(Random()-0.5)*2);
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA

Al parecer no tengo un código al respecto de esto así que mejor tener esto anotado. La idea es
ver cómo funcionan los arrays de punteros. Cuestión que al armar un array de punteros lo que tengo
que hacer es definir primero el tipo de datos que voy a guardar en los punteros, ya sea
(int*) o (double*) o lo que sea. Luego pongo el nombre del array y entre corchetes el número de punteros
que voy a contener. Si quiero, puedo asignar un puntero existente al array a partir de igualar el elemento del
array con el puntero. ap_Array[i_indice] = p_puntero. En cambio, si no quiero asignar punteros existentes
y quiero armar punteros nuevos, puedo malloquear el elemento del array como si fuera un puntero e inicializarlo
tal cual. (Eso es lo que está hecho abajo).
Finalmente, si quiero tomar un elemento del puntero es importante recordar que para seleccionar el puntero particular
uso corchetes y para elegir el elemento del puntero tengo que usar el * para desreferenciar el elemento. (Como se muestra abajo básicamente).

No lo mencioné, pero esto lo terminé el 12/07/2021.




// #include<stdio.h>
// #include<stdlib.h>
// #include<math.h>
// #include<time.h>
// #include<string.h>


// int main(int argc, char *argv[]){
	// // Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	// time_t tt_prin,tt_fin;
	// time(&tt_prin);
	// srand(time(NULL));
	// int i_tardanza;
	
	// // Armo un array a punteros y lo examino
	// int* ap_Array[3];
	
	// for(register int i_i=0; i_i<3; i_i++){
		// ap_Array[i_i] = (int*) malloc((2+5)*sizeof(int));
		// *ap_Array[i_i] = 1;
		// *(ap_Array[i_i]+1) = 4;
		// for(register int i_j=0; i_j<4; i_j++) *(ap_Array[i_i]+i_j+2) = i_i*4+i_j;
	// }
	
	// for(register int i_i=0; i_i<3; i_i++) for(register int i_j=0; i_j<4; i_j++) printf("El número de agente es %d\n",*(ap_Array[i_i]+i_j+2));
	
	// // Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// for(register int i_i=0; i_i<3; i_i++) free(ap_Array[i_i]);
	// time(&tt_fin);
	// i_tardanza = tt_fin-tt_prin;
	// printf("Tarde %d segundos en terminar",i_tardanza);
		
	// return 0;
// }



//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------


NUEVO_PROGRAMA