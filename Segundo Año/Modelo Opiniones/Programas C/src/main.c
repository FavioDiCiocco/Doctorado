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
	ps_datos->i_Iteraciones_extras = 500; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	ps_datos->d_dt = 0.1; // Paso temporal de iteración del sistema
	ps_datos->d_alfa = 1; // Controversialidad de los tópicos
	ps_datos->d_delta = 0.002*ps_datos->d_kappa; // Es un término que se suma en la homofilia y ayuda a que los pesos no diverjan.
	ps_datos->d_NormDif = sqrt(ps_datos->i_N*ps_datos->i_T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	ps_datos->d_CritCorte = pow(10,-3); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	ps_datos->i_testigos = ps_datos->i_N; // Esta es la cantidad de agentes de cada distancia que voy registrar
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int i_contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int i_pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int i_pasos_maximos = 200000; // Esta es la cantidad de pasos máximos a simular
	int i_ancho_ventana = 100; // Este es el ancho temporal que voy a tomar para promediar las opiniones de mis agentes.
		
	//#############################################################################################
	
	// Defino mis matrices y las inicializo
	
	// Matrices de mi sistema. Estas son la de Adyacencia, la de Superposición de Tópicos, la de vectores de opinión de los agentes y la de Separación
	ps_red->pi_Adyacencia = (int*) calloc(2+ps_datos->i_N*ps_datos->i_N,sizeof(int)); // Matriz de adyacencia de la red. Determina quienes están conectados con quienes
	ps_red->pd_Angulos = (double*) calloc(2+ps_datos->i_T*ps_datos->i_T,sizeof(double)); // Matriz simétrica de superposición entre tópicos.
	ps_red->pd_Opiniones = (double*) calloc(2+ps_datos->i_T*ps_datos->i_N,sizeof(double)); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// También hay un vector para guardar la diferencia entre el paso previo y el actual, un vector con los valores de saturación,
	ps_red->pd_Diferencia = (double*) calloc(2+ps_datos->i_T*ps_datos->i_N,sizeof(double)); // Vector que guarda la diferencia entre dos pasos del sistema
	
	// También hay un vector para la inversa a la beta de la distancia no ortogonal entre agentes
	ps_red->pd_Separacion = (double*) calloc(2+ps_datos->i_N*ps_datos->i_N,sizeof(double)); // Matriz de Separacion. Determina las dsitancias entre agentes.
	
	// También hay un vector para guardar el promedio temporal de las opiniones de los agentes en todos los tópicos
	ps_red->pd_Prom_Opi = (double*) calloc(2+ps_datos->i_T*ps_datos->i_N*2,sizeof(double)); // Vector que guarda la diferencia entre dos pasos del sistema
	
	// Inicializo mis cinco "matrices".
	// Matriz de Adyacencia. Es de tamaño N*N
	ps_red->pi_Adyacencia[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pi_Adyacencia[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Superposición de Tópicos. Es de tamaño T*T
	ps_red->pd_Angulos[0] = ps_datos->i_T; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Angulos[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de vectores de opinión. Es de tamaño N*T
	ps_red->pd_Opiniones[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Opiniones[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
	ps_red->pd_Diferencia[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Diferencia[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Separacion. Es de tamaño N*N
	ps_red->pd_Separacion[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Separacion[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Promedio de opiniones de cada agente y cada tópico. Es de tamaño 2*(N*T)
	ps_red->pd_Prom_Opi[0] = 2; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Prom_Opi[1] = ps_datos->i_N*ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char s_Opiniones[355];
	sprintf(s_Opiniones,"../Programas Python/Evolucion_temporal/1D_Hilos/Opiniones_N=%d_kappa=%.1f_beta=%.2f_cosd=%.2f_Iter=%d.file"
		,ps_datos->i_N,ps_datos->d_kappa,ps_datos->d_beta,ps_datos->d_Cosangulo,i_iteracion);
	FILE *pa_Opiniones=fopen(s_Opiniones,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// // Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	// char s_Testigos[355];
	// sprintf(s_Testigos,"../Programas Python/Evolucion_temporal/2D_dtchico/Testigos_N=%d_kappa=%.1f_beta=%.2f_cosd=%.2f_Iter=%d.file"
		// ,ps_datos->i_N,ps_datos->d_kappa,ps_datos->d_beta,ps_datos->d_Cosangulo,i_iteracion);
	// FILE *pa_Testigos=fopen(s_Testigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char s_matriz_adyacencia[355];
	sprintf(s_matriz_adyacencia,"MARE/Erdos-Renyi/ErdosRenyi_N=%d_ID=%d.file"
		,ps_datos->i_N,(int) i_iteracion%100); // El número es la cantidad de redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *pa_matriz_adyacencia=fopen(s_matriz_adyacencia,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*pf_Dinamica_Interaccion)(ps_Red ps_variables, ps_Param ps_parametros) = &Dinamica_opiniones;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(ps_red, ps_datos->d_kappa); // Esto me inicializa mi matriz de opiniones 
	GenerarAng(ps_red, ps_datos); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia(ps_red->pi_Adyacencia, pa_matriz_adyacencia); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(pa_matriz_adyacencia); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	
	//################################################################################################################################

	// Acá voy a hacer las simulaciones de pasos previos del sistema
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(pa_Opiniones,"Opiniones Iniciales\n");
	Escribir_d(ps_red->pd_Opiniones,pa_Opiniones);
	
	// // Me guardo los valores de opinión de mis agentes testigos
	// fprintf(pa_Testigos,"Opiniones Testigos\n");
	// for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]);
	// fprintf(pa_Testigos,"\n");
	
	// Sumo el estado inicial de las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
	for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+2] += ps_red->pd_Opiniones[i_j+2];
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar
	for(register int i_i=0; i_i<i_ancho_ventana-1; i_i++){
		RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
		
		// Voy sumando las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
		for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+2] += ps_red->pd_Opiniones[i_j+2];
		
	}
	
	// Promedio el valor de Prom_Opi al dividir por el tamaño de la ventana
	for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+2] = ps_red->pd_Prom_Opi[i_j+2]/i_ancho_ventana;
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	// for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]);
	// fprintf(pa_Testigos,"\n");
	
	fprintf(pa_Opiniones,"Variación promedio \n");
	
	while(i_contador < ps_datos->i_Iteraciones_extras && i_pasos_simulados < i_pasos_maximos){
		
		i_contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			
			// Evolución
			RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] += ps_red->pd_Opiniones[i_j+2];
			
			// Actualización de índices
			i_pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			// Cálculos derivados
			if(i_pasos_simulados%100==0){
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2]/i_ancho_ventana;
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Diferencia[i_j+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] - ps_red->pd_Prom_Opi[i_j+2];
				ps_red->d_Variacion_promedio = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Reinicio los promedios
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2];
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] = 0;
				
				// Escritura
				fprintf(pa_Opiniones, "%lf\t",ps_red->d_Variacion_promedio); // Guardo el valor de variación promedio
				// for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]);
				// fprintf(pa_Testigos,"\n");
			}
		}
		while( ps_red->d_Variacion_promedio > ps_datos->d_CritCorte && i_pasos_simulados < i_pasos_maximos);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while(i_contador < ps_datos->i_Iteraciones_extras && ps_red->d_Variacion_promedio <= ps_datos->d_CritCorte && i_pasos_simulados < i_pasos_maximos){
			
			// Evolución
			RK4(ps_red->pd_Opiniones, pf_Dinamica_Interaccion, ps_red, ps_datos); // Itero los intereses
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] += ps_red->pd_Opiniones[i_j+2];
			
			// Actualización de índices
			i_contador++; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
			i_pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			// Cálculos derivados
			if(i_pasos_simulados%100==0){
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2]/i_ancho_ventana;
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Diferencia[i_j+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] - ps_red->pd_Prom_Opi[i_j+2];
				ps_red->d_Variacion_promedio = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Reinicio los promedios
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+2] = ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2];
				for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) ps_red->pd_Prom_Opi[i_j+ps_datos->i_N*ps_datos->i_T+2] = 0;
				
				// Escritura
				fprintf(pa_Opiniones, "%lf\t",ps_red->d_Variacion_promedio); // Guardo el valor de variación promedio
				// for(register int i_j=0; i_j<ps_datos->i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_Testigos,"%lf\t",ps_red->pd_Opiniones[i_j*ps_datos->i_T+i_k+2]);
				// fprintf(pa_Testigos,"\n");
			}
			
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
	Escribir_d(ps_red->pd_Opiniones,pa_Opiniones);
	fprintf(pa_Opiniones,"Matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	Escribir_i(ps_red->pi_Adyacencia,pa_Opiniones);
	fprintf(pa_Opiniones,"Pasos Simulados\n");
	fprintf(pa_Opiniones,"%d\n",i_pasos_simulados);
	fprintf(pa_Opiniones,"Semilla\n");
	fprintf(pa_Opiniones,"%ld\n",semilla);
	
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	free(ps_red->pd_Angulos);
	free(ps_red->pi_Adyacencia);
	free(ps_red->pd_Opiniones);
	free(ps_red->pd_Separacion);
	free(ps_red->pd_Prom_Opi);
	free(ps_red->pd_Diferencia);
	free(ps_red);
	free(ps_datos);
	fclose(pa_Opiniones);
	// fclose(pa_Testigos);
	
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
