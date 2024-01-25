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
	time_t tprin, tfin, semilla;
	time(&tprin);
	semilla = time(NULL);
	srand(semilla); // Voy a definir la semilla a partir de time(NULL);
	float Tiempo; // Este es el float que le paso al printf para saber cuanto tardé
	
	//#############################################################################################
	
	// Creo mis punteros a structs y los malloqueo.
	puntero_Parametros param;
	param = malloc(sizeof( struct_Parametros ));
	
	puntero_Matrices red;
	red = malloc(sizeof( struct_Matrices )); 
	
	//#############################################################################################
		
	// Defino los parámetros de mi modelo. Esto va desde número de agentes hasta el paso temporal de integración.
	// Primero defino los parámetros que requieren un input.
	
	param->N = strtol(argv[1],NULL,10); // Cantidad de agentes en el modelo
	param->kappa = strtof(argv[2],NULL); // Esta amplitud regula la relación entre el término lineal y el término logístico
	param->epsilon = strtof(argv[3],NULL); // Este es el umbral que determina si el interés del vecino puede generarme más interés.
	param->Cosangulo = strtof(argv[4],NULL);// Este es el coseno de Delta que define la relación entre tópicos.
	int iteracion = strtol(argv[5],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	param->T = 2;  // Cantidad de temas sobre los que opinar
	param->Iteraciones_extras = 100; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	param->pasosprevios = 20; // Elegimos 20 de manera arbitraria con Pablo y Sebas. Sería la cantidad de pasos hacia atrás que miro para comparar cuanto varió el sistema
	param->alfa = 4; // Ex-Controversialidad de los tópicos
	param->dt = 0.01; // Paso temporal de iteración del sistema
	param->NormDif = sqrt(param->N*param->T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	param->CritCorte = pow(10,-4); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	param->testigos = fmin(param->N,6); // Esta es la cantidad de agentes de cada distancia que voy registrar
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	
	// Voy a armar mi array de punteros, el cual voy a usar para guardar los datos de pasos previos del sistema
	double* OpiPrevias[param->pasosprevios];
	
	for(int i=0; i<param->pasosprevios; i++){
		OpiPrevias[i] = (double*) calloc(( 2+param->T*param->N ), sizeof(double)); // Malloqueo los punteros de mis pasos previos
		// Defino su número de filas y columnas como N*T
		*OpiPrevias[i] = param->N;
		*( OpiPrevias[i]+1 ) = param->T;
	}
	
	//#############################################################################################
	
	// Defino mis matrices y las inicializo
	
	// Matrices de mi sistema. Estas son la de Adyacencia, la de Superposición de Tópicos y la de vectores de opinión de los agentes.
	red->Ady = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Lista de vecinos de la red. Determina quienes están conectados con quienes
	red->Ady_vecinos = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Complemento de lista de vecinos de la red. Determina quienes están conectados con quienes
	
	red->Ang = (double*) calloc( ( 2+param->T * param->T ), sizeof(double) ); // Matriz simétrica de superposición entre tópicos.
	red->Opi = (double*) calloc( ( 2+param->T * param->N ), sizeof(double) ); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// También hay un vector para guardar la diferencia entre el paso previo y el actual, un vector con los valores de saturación,
	red->Dif = (double*) calloc( ( 2+param->T * param->N ), sizeof(double) ); // Vector que guarda la diferencia entre dos pasos del sistema
	
	
	// Inicializo mis cinco "matrices".
	// Lista de vecinos. Tiene N filas y cada fila tiene tamaño variable
	red->pi_Adyacencia[0] = (int*) malloc(sizeof(int));
	red->pi_Adyacencia[0][0] = param->i_N; // Pongo el número de filas en la primer coordenada
	red->pi_Adyacencia[1] = (int*) malloc(sizeof(int));
	red->pi_Adyacencia[1][0] = 1; // Pongo el número de columnas en la segunda coordenada
	
	// Complemento de lista de vecinos. Tiene N filas y cada fila tiene tamaño variable
	red->pi_Adyacencia_vecinos[0] = (int*) malloc(sizeof(int));
	red->pi_Adyacencia_vecinos[0][0] = param->i_N; // Pongo el número de filas en la primer coordenada
	red->pi_Adyacencia_vecinos[1] = (int*) malloc(sizeof(int));
	red->pi_Adyacencia_vecinos[1][0] = 1; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Superposición de Tópicos. Es de tamaño T*T
	red->pd_Angulos[0] = param->i_T; // Pongo el número de filas en la primer coordenada
	red->pd_Angulos[1] = param->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de vectores de opinión. Es de tamaño N*T
	red->pd_Opiniones[0] = param->i_N; // Pongo el número de filas en la primer coordenada
	red->pd_Opiniones[1] = param->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
	red->pd_Diferencia[0] = param->i_N; // Pongo el número de filas en la primer coordenada
	red->pd_Diferencia[1] = param->i_T; // Pongo el número de columnas en la segunda coordenada
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// En el cuarto me anoto la dirección del archivo de texto con la matriz de adyacencia.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	// char s_Opiniones[355];
	// sprintf(s_Opiniones,"../Programas Python/CI_variables/Datos/Opiniones_N=%d_Cosd=%.2f_kappa=%.2f_epsilon=%.2f_Iter=%d.file"
		// ,param->i_N,param->d_Cosangulo,param->d_kappa,param->d_epsilon,i_iteracion);
	// FILE *pa_Opiniones=fopen(s_Opiniones,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	// char s_Testigos[355];
	// sprintf(s_Testigos,"../Programas Python/CI_variables/Datos/Testigos_N=%d_Cosd=%.2f_kappa=%.2f_epsilon=%.2f_Iter=%d.file"
		// ,param->i_N,param->d_Cosangulo,param->d_kappa,param->d_epsilon,i_iteracion);
	// FILE *pa_Testigos=fopen(s_Testigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char s_matriz_adyacencia[355];
	sprintf(s_matriz_adyacencia,"MARE/Erdos-Renyi/gm=8/ErdosRenyi_N=%d_ID=%d.file"
		,param->i_N,(int) i_iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *pa_matriz_adyacencia=fopen(s_matriz_adyacencia,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	// double (*pf_Din_Sat)(red var, ps_Param par) = &Din_saturacion;
	// double (*pf_Dinamica_Interaccion)(red ps_variables, ps_Param ps_parametros) = &Dinamica_interes;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(red, 0, param->d_kappa); // Esto me inicializa mi matriz de opiniones 
	GenerarAng(red, param); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia_Ejes(red, pa_matriz_adyacencia); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(pa_matriz_adyacencia); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	
	//################################################################################################################################

	/*
	// Acá voy a hacer las simulaciones de pasos previos del sistema
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(pa_Opiniones,"Opiniones Iniciales\n");
	Escribir_d(red->pd_Opiniones,pa_Opiniones);
	
	// fprintf(pa_Testigos,"Opiniones Testigos\n");
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar
	for(register int i_i=0; i_i<param->i_pasosprevios; i_i++){
		
		// Evolución
		RK4(red->pd_Opiniones, pf_Dinamica_Interaccion, red, param);
		
		// Escritura
		// for(register int i_j=0; i_j<param->i_testigos; i_j++) for(register int i_k=0; i_k<param->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",red->pd_Opiniones[i_j*param->i_T+i_k+2]);
		// fprintf(pa_Testigos,"\n");
		
		// Registro el estado actual en el array de OpinionesPrevias.
		for(register int i_j=0; i_j<param->i_N*param->i_T; i_j++) *(ap_OpinionesPrevias[i_i]+i_j+2) = red->pd_Opiniones[i_j+2];
	}
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	fprintf(pa_Opiniones,"Variación promedio \n");
	
	int i_IndiceOpiPasado = 0; 
	
	while(i_contador < param->i_Iteraciones_extras){
		
		i_contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			// Evolución
			RK4(red->pd_Opiniones, pf_Dinamica_Interaccion, red, param); // Itero los intereses
			
			// Cálculos derivados
			Delta_Vec_d(red->pd_Opiniones,ap_OpinionesPrevias[i_IndiceOpiPasado%param->i_pasosprevios],red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<param->i_N*param->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%param->i_pasosprevios]+i_p+2) = red->pd_Opiniones[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			red->d_Variacion_promedio = Norma_d(red->pd_Diferencia)/param->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			
			// Escritura
			fprintf(pa_Opiniones, "%lf\t",red->d_Variacion_promedio); // Guardo el valor de variación promedio
			// for(register int i_j=0; i_j<param->i_testigos; i_j++) for(register int i_k=0; i_k<param->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",red->pd_Opiniones[i_j*param->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
			// fprintf(pa_Testigos,"\n");
			
			// Actualización de índices
			i_IndiceOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
		}
		while( red->d_Variacion_promedio > param->d_CritCorte);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while(i_contador < param->i_Iteraciones_extras && red->d_Variacion_promedio <= param->d_CritCorte ){
			// Evolución
			RK4(red->pd_Opiniones, pf_Dinamica_Interaccion, red, param); // Itero los intereses
			
			// Cálculos derivados
			Delta_Vec_d(red->pd_Opiniones,ap_OpinionesPrevias[i_IndiceOpiPasado%param->i_pasosprevios],red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<param->i_N*param->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%param->i_pasosprevios]+i_p+2) = red->pd_Opiniones[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			red->d_Variacion_promedio = Norma_d(red->pd_Diferencia)/param->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			
			// Escritura
			fprintf(pa_Opiniones, "%lf\t",red->d_Variacion_promedio); // Guardo el valor de variación promedio 
			// for(register int i_j=0; i_j<param->i_testigos; i_j++) for(register int i_k=0; i_k<param->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",red->pd_Opiniones[i_j*param->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
			// fprintf(pa_Testigos,"\n");
			
			// Actualización de índices
			i_IndiceOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
			i_contador +=1; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
		}
		
		// Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	}
	
	//################################################################################################################################
	
	// Guardo las últimas cosas, libero las memorias malloqueadas y luego termino
	
	// Guardo las opiniones finales, la matriz de adyacencia y la semilla en el primer archivo.
	fprintf(pa_Opiniones,"\n");
	fprintf(pa_Opiniones,"Opiniones finales\n");
	Escribir_d(red->pd_Opiniones,pa_Opiniones);
	fprintf(pa_Opiniones,"Matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	Escribir_i(red->pi_Adyacencia,pa_Opiniones);
	fprintf(pa_Opiniones,"Semilla\n");
	fprintf(pa_Opiniones,"%ld\n",semilla);
	*/
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	for(register int i_i=0; i_i<param->i_pasosprevios; i_i++) free(ap_OpinionesPrevias[i_i]);
	free(red->pd_Angulos);
	for(int i_i=0; i_i<param->i_N+2; i_i++) free(red->pi_Adyacencia[i_i]);
	free(red->pi_Adyacencia);
	for(register int i_i=0; i_i<param->i_N+2; i_i++) free(red->pi_Adyacencia_vecinos[i_i]);
	free(red->pi_Adyacencia_vecinos);
	free(red->pd_Opiniones);
	free(red->pd_Diferencia);
	free(red);
	free(param);
	// fclose(pa_Opiniones);
	// fclose(pa_Testigos);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tt_fin);
	f_tardanza = tt_fin-tt_prin;
	// sleep(1);
	printf("Tarde %.1f segundos \n",f_tardanza);
	
	return 0;
 }





//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
// Esto lo comento porque considero que no voy a necesitar esto para el programa en el caso de la transcrítica logarítmica

// Defino unas variables double que voy a necesitar
// double d_tope = 0; // Este es el máximo valor que puede tomar la norma del vector de opiniones
// for(register int i_i = 0; i_i < param->i_N; i_i++) for(register int i_j=0; i_j < param->i_T; i_j++) d_tope += 100*100; // Considero como máximo que las opiniones valgan 100
// d_tope = sqrt(d_tope); // Le tomo la raíz cuadrada para que sea la norma.
// double d_normav = 0; // Este es el valor que tiene la norma del vector de opiniones
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~