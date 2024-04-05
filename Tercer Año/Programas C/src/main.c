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
	param->kappa = strtof(argv[2],NULL); // Esta amplitud regula la relación entre el término lineal y el término con tanh
	param->beta = strtof(argv[3],NULL); // Esta es la potencia que determina el grado de homofilia.
	param->Cosd = strtof(argv[4],NULL); // Este es el coseno de Delta que define la relación entre tópicos.
	int iteracion = strtol(argv[5],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	param->T = 2;  //strtol(argv[1],NULL,10); Antes de hacer esto, arranquemos con número fijo   // Cantidad de temas sobre los que opinar
	param->Iteraciones_extras = 5000; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	param->dt = 0.1; // Paso temporal de iteración del sistema
	param->alfa = 1; // Controversialidad de los tópicos
	param->delta = 0.002*param->kappa; // Es un término que se suma en la homofilia y ayuda a que los pesos no diverjan.
	param->NormDif = sqrt(param->N*param->T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	param->CritCorte = pow(10,-3); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	param->testigos = fmin(param->N,50); // Esta es la cantidad de agentes de cada distancia que voy registrar
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int pasos_maximos = 200000; // Esta es la cantidad de pasos máximos a simular
	int ancho_ventana = 1000; // Este es el ancho temporal que voy a tomar para promediar las opiniones de mis agentes.
		
	//#############################################################################################
	
	// Defino mis matrices y las inicializo
	
	// Matrices de mi sistema. Estas son la de Adyacencia, Adyacencia_vecinos, la de Superposición de Tópicos y la de vectores de opinión de los agentes.
	red->Ady = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Lista de vecinos de la red. Determina quienes están conectados con quienes
	red->Ady_vecinos = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Complemento de lista de vecinos de la red. Determina quienes están conectados con quienes
	
	red->Ang = (double*) calloc( 2+param->T*param->T, sizeof(double) ); // Matriz simétrica de superposición entre tópicos.
	red->Opi = (double*) calloc( 2+param->T*param->N, sizeof(double) ); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// Vector que guarda la diferencia entre el paso previo y el actual, un vector con los valores de saturación,
	red->Dif = (double*) calloc( 2+param->T*param->N, sizeof(double) ); // Vector que guarda la diferencia entre dos pasos del sistema
	
	// Vector para la inversa a la beta de la distancia no ortogonal entre agentes
	red->Sep = (double*) calloc( 2+param->N*param->N, sizeof(double) ); // Matriz de Separacion. Determina las dsitancias entre agentes.
	
	// Vector para los valores de la tangente hiperbólica aplicada a las opiniones de los agentes
	red->Exp = (double*) calloc( 2+param->T * param->N , sizeof(double) ); // Vector que guarda los cálculos de las exponenciales de cada agente.
	
	// Vector para guardar el promedio temporal de las opiniones de los agentes en todos los tópicos
	red->Prom_Opi = (double*) calloc( 2+param->T*param->N*2, sizeof(double) ); // Vector que guarda la diferencia entre dos pasos del sistema
	
	// Inicializo mis cinco "matrices".
	// Lista de vecinos. Tiene N filas y cada fila tiene tamaño variable
	red->Ady[0] = (int*) malloc( sizeof(int) );
	red->Ady[0][0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Ady[1] = (int*) malloc( sizeof(int) );
	red->Ady[1][0] = 1; // Pongo el número de columnas en la segunda coordenada
	
	// Complemento de lista de vecinos. Tiene N filas y cada fila tiene tamaño variable
	red->Ady_vecinos[0] = (int*) malloc(sizeof(int));
	red->Ady_vecinos[0][0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Ady_vecinos[1] = (int*) malloc(sizeof(int));
	red->Ady_vecinos[1][0] = 1; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Superposición de Tópicos. Es de tamaño T*T
	red->Ang[0] = param->T; // Pongo el número de filas en la primer coordenada
	red->Ang[1] = param->T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de vectores de opinión. Es de tamaño N*T
	red->Opi[0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Opi[1] = param->T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
	red->Dif[0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Dif[1] = param->T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Separacion. Es de tamaño N*N
	red->Sep[0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Sep[1] = param->N; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Promedio de opiniones de cada agente y cada tópico. Es de tamaño 2*(N*T)
	red->Prom_Opi[0] = 2; // Pongo el número de filas en la primer coordenada
	red->Prom_Opi[1] = param->N*param->T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de cálculo de exponencial para cada agetne. Es de tamaño N*T
	red->Exp[0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Exp[1] = param->T; // Pongo el número de columnas en la segunda coordenada
	
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char TextOpi[355];
	sprintf(TextOpi,"../Programas Python/Opinion_actualizada/Zoom_Beta-Cosd/Opiniones_N=%d_kappa=%.1f_beta=%.2f_cosd=%.2f_Iter=%d.file"
		,param->N, param->kappa, param->beta, param->Cosd, iteracion);
	FILE *FileOpi = fopen(TextOpi,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// // Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	// char TextTestigos[355];
	// sprintf(TextTestigos,"../Programas Python/Opinion_actualizada/Datos/Testigos_N=%d_kappa=%.1f_beta=%.2f_cosd=%.2f_Iter=%d.file"
		// ,param->N, param->kappa, param->beta, param->Cosd, iteracion);
	// FILE *FileTestigos = fopen(TextTestigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char TextMatriz[355];
	sprintf(TextMatriz, "MARE/Erdos-Renyi/gm=10/ErdosRenyi_N=%d_ID=%d.file", param->N, (int) iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *FileMatriz = fopen(TextMatriz,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*func_dinamica)(puntero_Matrices red, puntero_Parametros param) = &Dinamica_opiniones;
	double (*func_activacion)(puntero_Matrices red, puntero_Parametros param) = &Dinamica_sumatoria;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(red, param->kappa); // Esto me inicializa mi matriz de opiniones 
	GenerarAng(red, param); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia_Ejes(red, FileMatriz); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(FileMatriz); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	
	//################################################################################################################################

	// Acá voy a hacer las simulaciones de pasos previos del sistema
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(FileOpi,"Opiniones Iniciales\n");
	Escribir_d(red->Opi,FileOpi);
	
	// // Me guardo los valores de opinión de mis agentes testigos
	// fprintf(FileTestigos,"Opiniones Testigos\n");
	// for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos, "%lf\t", red->Opi[ j*param->T +k+2 ] );
	// fprintf(FileTestigos,"\n");
	
	// Sumo el estado inicial de las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
	for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[j+2] += red->Opi[j+2];
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar
	for(int i=0; i<ancho_ventana-1; i++){
		RK4(red->Opi, func_dinamica, func_activacion, red, param); // Itero los intereses
		
		// Voy sumando las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
		for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[j+2] += red->Opi[j+2];
	}
	
	// Promedio el valor de Prom_Opi al dividir por el tamaño de la ventana
	for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[j+2] = red->Prom_Opi[j+2] / ancho_ventana;
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	// for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos, "%lf\t", red->Opi[ j*param->T +k+2 ] );
	// fprintf(FileTestigos,"\n");
	
	fprintf(FileOpi,"Variación promedio \n");
	
	
	while(contador < param->Iteraciones_extras && pasos_simulados < pasos_maximos){
		
		contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			
			// Evolución
			RK4(red->Opi, func_dinamica, func_activacion, red, param); // Itero los intereses
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] += red->Opi[j+2];
			
			// Actualización de índices
			pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			// Cálculos derivados
			if( pasos_simulados%100==0 ){
				// Escritura
				// for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos, "%lf\t", red->Opi[ j*param->T +k+2 ] );
				// fprintf(FileTestigos,"\n");
			}
			
			if( pasos_simulados%ancho_ventana==0 ){
				// Mido la variación de promedios
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] = red->Prom_Opi[ j+param->N*param->T+2 ] / ancho_ventana;
				for(int j=0; j<param->N*param->T; j++) red->Dif[j+2] = red->Prom_Opi[ j+param->N*param->T+2 ] - red->Prom_Opi[j+2];
				red->Variacion_promedio = Norma_d(red->Dif) / param->NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Escritura
				fprintf(FileOpi, "%lf\t", red->Variacion_promedio); // Guardo el valor de variación promedio
				
				// Reinicio los promedios
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[j+2] = red->Prom_Opi[ j+param->N*param->T+2 ];
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] = 0;
			}

		}
		while( red->Variacion_promedio > param->CritCorte && pasos_simulados < pasos_maximos);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while( contador < param->Iteraciones_extras && red->Variacion_promedio <= param->CritCorte && pasos_simulados < pasos_maximos){
			
			// Evolución
			RK4(red->Opi, func_dinamica, func_activacion, red, param); // Itero los intereses
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] += red->Opi[j+2];
			
			// Actualización de índices
			contador++; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
			pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			// Cálculos derivados
			if( pasos_simulados%100==0 ){
				// Escritura
				// for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos, "%lf\t", red->Opi[ j*param->T +k+2 ] );
				// fprintf(FileTestigos,"\n");
			}
			
			if( pasos_simulados%ancho_ventana==0 ){
				// Mido la variación de promedios
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] = red->Prom_Opi[ j+param->N*param->T+2 ] / ancho_ventana;
				for(int j=0; j<param->N*param->T; j++) red->Dif[j+2] = red->Prom_Opi[ j+param->N*param->T+2 ] - red->Prom_Opi[j+2];
				red->Variacion_promedio = Norma_d(red->Dif) / param->NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Escritura
				fprintf(FileOpi, "%lf\t", red->Variacion_promedio); // Guardo el valor de variación promedio
				
				// Reinicio los promedios
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[j+2] = red->Prom_Opi[ j+param->N*param->T+2 ];
				for(int j=0; j<param->N*param->T; j++) red->Prom_Opi[ j+param->N*param->T+2 ] = 0;
			}
			
		}
		
		// Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	}
	
	//################################################################################################################################
	
	// Guardo las últimas cosas, libero las memorias malloqueadas y luego termino
	
	// Guardo las opiniones finales, la matriz de adyacencia y la semilla en el primer archivo.
	fprintf(FileOpi, "\n");
	fprintf(FileOpi, "Opiniones finales\n");
	Escribir_d(red->Opi, FileOpi);
	fprintf(FileOpi, "Pasos Simulados\n");
	fprintf(FileOpi, "%d\n", pasos_simulados);
	fprintf(FileOpi, "Semilla\n");
	fprintf(FileOpi, "%ld\n", semilla);
	fprintf(FileOpi, "Matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	for(int i=0; i<param->N; i++) Escribir_i(red->Ady[i+2], FileOpi);
	
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	for(int i=0; i<param->N+2; i++) free( red->Ady[i] );
	free( red->Ady );
	for(int i=0; i<param->N+2; i++) free( red->Ady_vecinos[i] );
	free( red->Ady_vecinos );
	free( red->Ang );
	free( red->Opi );
	free( red->Sep );
	free( red->Prom_Opi );
	free( red->Dif );
	free( red->Exp );
	free( red );
	free( param );
	fclose( FileOpi );
	// fclose( FileTestigos );
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tfin);
	Tiempo = tfin-tprin;
	printf("Tarde %.1f segundos \n", Tiempo);
	
	return 0;
 }


