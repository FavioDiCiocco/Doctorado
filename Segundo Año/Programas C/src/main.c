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
	ps_datos->d_beta = strtof(argv[2],NULL); // Esta es la potencia que determina el grado de homofilia.
	ps_datos->d_Cosangulo = strtof(argv[3],NULL); // Este es el coseno de Delta que define la relación entre tópicos.
	int i_iteracion = strtol(argv[4],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	ps_datos->i_T = 2;  //strtol(argv[1],NULL,10); Antes de hacer esto, arranquemos con número fijo   // Cantidad de temas sobre los que opinar
	ps_datos->i_Iteraciones_extras = 40; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	ps_datos->i_pasosprevios = 20; // Elegimos 20 de manera arbitraria con Pablo y Sebas. Sería la cantidad de pasos hacia atrás que miro para comparar cuanto varió el sistema
	ps_datos->d_dt = 0.01; // Paso temporal de iteración del sistema
	ps_datos->d_alfa = 1; // Controversialidad de los tópicos
	ps_datos->d_kappa = 10; // Esta amplitud regula la relación entre el término lineal y el término logístico
	ps_datos->d_delta = 0.002*ps_datos->d_kappa; // Es un término que se suma en la homofilia y ayuda a que los pesos no diverjan.
	ps_datos->d_NormDif = sqrt(ps_datos->i_N*ps_datos->i_T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	ps_datos->d_CritCorte = pow(10,-4); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	ps_datos->i_testigos = fmin(ps_datos->i_N,50); // Esta es la cantidad de agentes de cada distancia que voy registrar
		
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int i_contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int i_pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int i_pasos_maximos = 20000; // Esta es la cantidad de pasos máximos a simular
	
	// Voy a armar mi array de punteros, el cual voy a usar para guardar los datos de pasos previos del sistema
	double* ap_OpinionesPrevias[ps_datos->i_pasosprevios];
		
	for(register int i_i=0; i_i<ps_datos->i_pasosprevios; i_i++){
		ap_OpinionesPrevias[i_i] = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Malloqueo los punteros de mis pasos previos
		// Defino su número de filas y columnas como N*T
		*ap_OpinionesPrevias[i_i] = ps_datos->i_N;
		*(ap_OpinionesPrevias[i_i]+1) = ps_datos->i_T;
		for(register int i_j=0; i_j<ps_datos->i_T*ps_datos->i_N;i_j++) *(ap_OpinionesPrevias[i_i]+i_j+2) = 0; // Inicializo los punteros
	}
	
	//#############################################################################################
	
	// Defino mis matrices y las inicializo
	
	// Matrices de mi sistema. Estas son la de Adyacencia, la de Superposición de Tópicos y la de vectores de opinión de los agentes.
	ps_red->pi_Adyacencia = (int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int)); // Matriz de adyacencia de la red. Determina quienes están conectados con quienes
	ps_red->pd_Angulos = (double*) malloc((2+ps_datos->i_T*ps_datos->i_T)*sizeof(double)); // Matriz simétrica de superposición entre tópicos.
	ps_red->pd_Opiniones = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// También hay un vector para guardar la diferencia entre el paso previo y el actual, un vector con los valores de saturación,
	ps_red->pd_Diferencia = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Vector que guarda la diferencia entre dos pasos del sistema
	
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
	
	// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_T+2; i_i++) ps_red->pd_Diferencia[i_i] = 0; // Inicializo la matriz
	ps_red->pd_Diferencia[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Diferencia[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char s_Opiniones[355];
	sprintf(s_Opiniones,"../Programas Python/Homofilia_estatica/%dD/Opiniones_N=%d_beta=%.2f_cosd=%.2f_Iter=%d.file"
		,ps_datos->i_T,ps_datos->i_N,ps_datos->d_beta,ps_datos->d_Cosangulo,i_iteracion);
	FILE *pa_Opiniones=fopen(s_Opiniones,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	char s_Testigos[355];
	sprintf(s_Testigos,"../Programas Python/Homofilia_estatica/%dD/Testigos_N=%d_beta=%.2f_cosd=%.2f_Iter=%d.file"
		,ps_datos->i_T,ps_datos->i_N,ps_datos->d_beta,ps_datos->d_Cosangulo,i_iteracion);
	FILE *pa_Testigos=fopen(s_Testigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char s_matriz_adyacencia[355];
	sprintf(s_matriz_adyacencia,"MARE/Erdos-Renyi/ErdosRenyi_N=%d_ID=%d.file"
		,ps_datos->i_N,(int) i_iteracion%30); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *pa_matriz_adyacencia=fopen(s_matriz_adyacencia,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*pf_Dinamica_Interaccion)(ps_Red ps_variables, ps_Param ps_parametros) = &Dinamica_opiniones;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(ps_red, ps_datos->d_kappa); // Esto me inicializa mi matriz de opiniones 
	GenerarAng(ps_red, ps_datos); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia(ps_red->pi_Adyacencia, pa_matriz_adyacencia); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(pa_matriz_adyacencia); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	printf("Armé las matrices y las inicialicé\n");
	
	//################################################################################################################################

	// Acá voy a hacer las simulaciones de pasos previos del sistema
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(pa_Opiniones,"Opiniones Iniciales\n");
	Escribir_d(ps_red->pd_Opiniones,pa_Opiniones);
	
	if(i_iteracion < 2) fprintf(pa_Testigos,"Opiniones Testigos\n");
	
	printf("Guardé las condiciones iniciales \n");
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar
	for(register int i_i=0; i_i<ps_datos->i_pasosprevios; i_i++){
		RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
		// Me guardo los valores de opinión de mis agentes testigo
		if(i_iteracion < 2){
			for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]);
			fprintf(pa_Testigos,"\n");
		}
		// Registro el estado actual en el array de OpinionesPrevias.
		for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) *(ap_OpinionesPrevias[i_i]+i_j+2) = ps_red->pd_Opiniones[i_j+2];
	}
	
	printf("Termalicé el sistema \n");
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	fprintf(pa_Opiniones,"Variación promedio \n");
	
	int i_IndiceOpiPasado = 0; 
	
	while(i_contador < ps_datos->i_Iteraciones_extras && i_pasos_simulados < i_pasos_maximos){
		
		i_contador = 0; // Inicializo el contador
		
		printf("Estoy simulando por encima del criterio de corte \n");
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			// Evolución
			RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
			// Cálculos derivados
			Delta_Vec_d(ps_red->pd_Opiniones,ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios],ps_red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<ps_datos->i_N*ps_datos->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios]+i_p+2) = ps_red->pd_Opiniones[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			ps_red->d_Variacion_promedio = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			// Escritura
			fprintf(pa_Opiniones, "%lf\t",ps_red->d_Variacion_promedio); // Guardo el valor de variación promedio
			if(i_iteracion < 2){
				for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
				fprintf(pa_Testigos,"\n");
			}
			// Actualización de índices
			i_IndiceOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
			i_pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
		}
		while( ps_red->d_Variacion_promedio > ps_datos->d_CritCorte && i_pasos_simulados < i_pasos_maximos);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		printf("Estoy simulando por debajo del criterio de corte \n");
		
		while(i_contador < ps_datos->i_Iteraciones_extras && ps_red->d_Variacion_promedio <= ps_datos->d_CritCorte && i_pasos_simulados < i_pasos_maximos){
			// Evolución
			RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
			// Cálculos derivados
			Delta_Vec_d(ps_red->pd_Opiniones,ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios],ps_red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<ps_datos->i_N*ps_datos->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios]+i_p+2) = ps_red->pd_Opiniones[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			ps_red->d_Variacion_promedio = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			// Escritura
			fprintf(pa_Opiniones, "%lf\t",ps_red->d_Variacion_promedio); // Guardo el valor de variación promedio 
			if(i_iteracion < 2){
				for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
				fprintf(pa_Testigos,"\n");
			}
			
			// Actualización de índices
			i_IndiceOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
			i_contador++; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
			i_pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
		}
		
		// Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	}
	
	printf("Terminé de simular, guardo todo \n");
	
	//################################################################################################################################
	
	// Guardo las últimas cosas, libero las memorias malloqueadas y luego termino
	
	// Guardo las opiniones finales, la matriz de adyacencia y la semilla en el primer archivo.
	fprintf(pa_Opiniones,"\n");
	fprintf(pa_Opiniones,"Opiniones finales\n");
	Escribir_d(ps_red->pd_Opiniones,pa_Opiniones);
	fprintf(pa_Opiniones,"Matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	Escribir_i(ps_red->pi_Adyacencia,pa_Opiniones);
	fprintf(pa_Opiniones,"Semilla\n");
	fprintf(pa_Opiniones,"%ld\n",semilla);
	
	printf("Ya guardé todo \n");
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	for(register int i_i=0; i_i<ps_datos->i_pasosprevios; i_i++) free(ap_OpinionesPrevias[i_i]);
	free(ps_red->pd_Angulos);
	free(ps_red->pi_Adyacencia);
	free(ps_red->pd_Opiniones);
	free(ps_red->pd_Diferencia);
	free(ps_red);
	free(ps_datos);
	fclose(pa_Opiniones);
	fclose(pa_Testigos);
	
	// Borro el archivo de Testigos que no necesito
	if(i_iteracion >= 2) remove(s_Testigos); // Esta función recibe el path al archivo de Testigos
	
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
// for(register int i_i = 0; i_i < ps_datos->i_N; i_i++) for(register int i_j=0; i_j < ps_datos->i_T; i_j++) d_tope += 100*100; // Considero como máximo que las opiniones valgan 100
// d_tope = sqrt(d_tope); // Le tomo la raíz cuadrada para que sea la norma.
// double d_normav = 0; // Este es el valor que tiene la norma del vector de opiniones
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~