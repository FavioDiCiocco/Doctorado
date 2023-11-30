// Voy a armar el código con el término de la ecuación logística

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include <unistd.h> // Esto lo uso para poner un sleep
#include"general.h"
#include"inicializar.h"
#include"avanzar.h"



int main(int argc, char *argv[]){
	// Empecemos con la base. Defino variables de tiempo para medir cuanto tardo y cosas básicas
	time_t tt_prin,tt_fin,semilla;
	time(&tt_prin);
	semilla = time(NULL);
	srand(semilla); // Voy a definir la semilla a partir de time(NULL);
	float f_tardanza; // Este es el float que le paso al printf para saber cuanto tardé
	
	//#############################################################################################
	
	// Creo mis punteros a structs y los malloqueo.
	ps_Param ps_datos; // No te vayas a confundir, que ps_Param es el tipo de dato definido por usuario como un puntero al struct Parametros. En cambio, ps_datos es un puntero
	ps_datos = malloc(sizeof(s_Param));
	
	ps_Red ps_red; // No te vayas a confundir, que ps_Red es el tipo de dato definido por usuario como un puntero al struct Red. En cambio, ps_red es un puntero
	ps_red = malloc(sizeof(s_Red)); 
	
	//#############################################################################################
		
	// Defino los parámetros de mi modelo. Esto va desde número de agentes hasta el paso temporal de integración.
	// Primero defino los parámetros que requieren un input.
	ps_datos->i_N = strtol(argv[1],NULL,10); // Cantidad de agentes en el modelo
	ps_datos->d_kappa = strtof(argv[2],NULL); // Esta amplitud regula la relación entre el término lineal y el término con tanh
	ps_datos->d_beta = strtof(argv[3],NULL); // Esta es la potencia que determina el grado de homofilia.
	ps_datos->d_Cosangulo = strtof(argv[4],NULL); // Este es el coseno de Delta que define la relación entre tópicos.
	int i_iteracion = strtol(argv[5],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	ps_datos->i_T = 2;  //strtol(argv[1],NULL,10); Antes de hacer esto, arranquemos con número fijo   // Cantidad de temas sobre los que opinar
	ps_datos->d_dt = 0.1; // Paso temporal de iteración del sistema
	ps_datos->d_alfa = 1; // Controversialidad de los tópicos
	ps_datos->d_delta = 0.002*ps_datos->d_kappa; // Es un término que se suma en la homofilia y ayuda a que los pesos no diverjan.
	ps_datos->i_testigos = fmin(ps_datos->i_N,200); // Esta es la cantidad de agentes de cada distancia que voy registrar
		
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int i_pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int i_pasos_maximos = 1000000; // Esta es la cantidad de pasos máximos a simular
	// Estos valores de varianza me sirven para identificar si el sistema está polarizado o no.
	double d_varianza_1 = 0;
	double d_varianza_2 = 0;
	double d_corte_polarizacion = 0.1; // Este es el valor de corte para determinar si el sistema está polarizado o no
	// Lo definí este valor mirando valores que obtuve para mis simulaciones sintéticas.
	
	//#############################################################################################
	
	// Defino mis matrices y las inicializo
	
	// Matrices de mi sistema. Estas son la de Adyacencia, la de Superposición de Tópicos y la de vectores de opinión de los agentes.
	ps_red->pi_Adyacencia = (int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int)); // Matriz de adyacencia de la red. Determina quienes están conectados con quienes
	ps_red->pd_Angulos = (double*) malloc((2+ps_datos->i_T*ps_datos->i_T)*sizeof(double)); // Matriz simétrica de superposición entre tópicos.
	ps_red->pd_Opiniones = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	ps_red->pd_Ti = (double*) malloc((2+ps_datos->i_N)*sizeof(double)); // Opinión de todos los agentes sobre el tópico i
	
	// Inicializo mis cuatro "matrices".
	// Matriz de Adyacencia. Es de tamaño N*N
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N+2; i_i++) ps_red->pi_Adyacencia[i_i] = 0; // Inicializo la matriz
	ps_red->pi_Adyacencia[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pi_Adyacencia[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Superposición de Tópicos. Es de tamaño T*T
	for(register int i_i=0; i_i<ps_datos->i_T*ps_datos->i_T+2; i_i++) ps_red->pd_Angulos[i_i] = 0;
	ps_red->pd_Angulos[0] = ps_datos->i_T; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Angulos[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de vectores de opinión. Es de tamaño N*T
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_T+2; i_i++) ps_red->pd_Opiniones[i_i] = 0; // Inicializo la matriz
	ps_red->pd_Opiniones[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Opiniones[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de opinión del tópico i. Es de tamaño N
	for(register int i_i=0; i_i<ps_datos->i_N+2; i_i++) ps_red->pd_Ti[i_i] = 0; // Inicializo la matriz
	ps_red->pd_Ti[0] = 1; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Ti[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char s_Opiniones[355];
	sprintf(s_Opiniones,"../Programas Python/Prueba_tiempos/Datos/Opiniones_N=%d_kappa=%.1f_beta=%.2f_cosd=%.2f_Iter=%d.file"
		,ps_datos->i_N,ps_datos->d_kappa,ps_datos->d_beta,ps_datos->d_Cosangulo,i_iteracion);
	FILE *pa_Opiniones=fopen(s_Opiniones,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char s_matriz_adyacencia[355];
	sprintf(s_matriz_adyacencia,"MARE/Erdos-Renyi/ErdosRenyi_N=%d_ID=%d.file"
		,ps_datos->i_N,(int) i_iteracion%100); // El número es la cantidad de redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *pa_matriz_adyacencia=fopen(s_matriz_adyacencia,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*pf_Dinamica_Interaccion)(ps_Red ps_variables, ps_Param ps_parametros) = &Dinamica_opiniones;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarAng(ps_red, ps_datos); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia(ps_red->pi_Adyacencia, pa_matriz_adyacencia); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(pa_matriz_adyacencia); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	//################################################################################################################################
	
	fprintf(pa_Opiniones,"Evolucion Opiniones \n");
	// printf("Voy a crear los datos\n");
	do{
		GenerarOpi(ps_red, ps_datos->d_kappa); // Esto me inicializa mi matriz de opiniones 
		
		// Hago las primeras 1000 simulaciones. Si el sistema está polarizado al final de esto, continuo con el resto.
		// Sino, vuelvo a redistribuir las opiniones iniciales
		
		for(i_pasos_simulados=0; i_pasos_simulados<1000; i_pasos_simulados++) RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero las opiniones
		
		// Calculo las varianzas
		for(register int i_j=0; i_j<ps_datos->i_N;i_j++) *(ps_red->pd_Ti+i_j+2) = *(ps_red->pd_Opiniones+i_j*ps_datos->i_T+2);
		d_varianza_1 = Varianza(ps_red->pd_Ti,ps_datos->d_kappa);
		for(register int i_j=0; i_j<ps_datos->i_N;i_j++) *(ps_red->pd_Ti+i_j+2) = *(ps_red->pd_Opiniones+i_j*ps_datos->i_T+2+1);
		d_varianza_2 = Varianza(ps_red->pd_Ti,ps_datos->d_kappa);
		
	}
	while((d_varianza_1+d_varianza_2)/2 < d_corte_polarizacion);
	
	// printf("Ya generé un estado polarizado, ahora voy a evolucionar el sistema\n");
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que deje de estar polarizado o hasta un tiempo 100*1000
	
	while((d_varianza_1+d_varianza_2)/2 > d_corte_polarizacion && i_pasos_simulados < i_pasos_maximos){		
		
		RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero las opiniones
		
		i_pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
		
		if(i_pasos_simulados%1000 == 0){
			// Calculo las varianzas
			for(register int i_j=0; i_j<ps_datos->i_N;i_j++) *(ps_red->pd_Ti+i_j+2) = *(ps_red->pd_Opiniones+i_j*ps_datos->i_T+2);
			d_varianza_1 = Varianza(ps_red->pd_Ti,ps_datos->d_kappa);
			for(register int i_j=0; i_j<ps_datos->i_N;i_j++) *(ps_red->pd_Ti+i_j+2) = *(ps_red->pd_Opiniones+i_j*ps_datos->i_T+2+1);
			d_varianza_2 = Varianza(ps_red->pd_Ti,ps_datos->d_kappa);
		}
		
		if(i_pasos_simulados%10000==0){
			Escribir_d(ps_red->pd_Opiniones,pa_Opiniones); // Anoto las opiniones luego de 10000 pasos de evolución, que es un tiempo 1000
			// printf("Guardé datos del sistema\n");
		}
		
	}
	
	
	
	//################################################################################################################################
	
	// Guardo las últimas cosas, libero las memorias malloqueadas y luego termino
	
	// Guardo la semilla en el primer archivo.
	fprintf(pa_Opiniones,"Semilla\n");
	fprintf(pa_Opiniones,"%ld\n",semilla);
	
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	free(ps_red->pd_Angulos);
	free(ps_red->pi_Adyacencia);
	free(ps_red->pd_Opiniones);
	free(ps_red->pd_Ti);
	free(ps_red);
	free(ps_datos);
	fclose(pa_Opiniones);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tt_fin);
	f_tardanza = tt_fin-tt_prin;
	printf("Tarde %.1f segundos \n",f_tardanza);
	
	return 0;
 }





//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
// Esto lo comento porque considero que no voy a necesitar esto para el programa en el caso de la transcrítica logarítmica

// Defino unas variables double que voy a necesitar
// double d_tope = 0; // Este es el máximo valor que puede tomar la norma del vector de opiniones
// for(register int i_i = 0; i_i < ps_datos->i_N; i_i++) for(register int i_j=0; i_j < ps_datos->i_T; i_j++) d_tope += 100*100; // Considero como máximo que las opiniones valgan 100
// d_tope = sqrt(d_tope); // Le tomo la raíz cuadrada para que sea la norma.
// double d_normav = 0; // Este es el valor que tiene la norma del vector de opiniones
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
