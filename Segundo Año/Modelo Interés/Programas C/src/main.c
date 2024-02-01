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
	param->alfa = strtof(argv[4],NULL); // Ex-Controversialidad de los tópicos
	param->Gradomedio = strtol(argv[5],NULL,10); // Este es el grado medio de la red utilizada
	int iteracion = strtol(argv[6],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	param->T = 1;  // Cantidad de temas sobre los que opinar
	param->Iteraciones_extras = 100; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	param->pasosprevios = 20; // Elegimos 20 de manera arbitraria con Pablo y Sebas. Sería la cantidad de pasos hacia atrás que miro para comparar cuanto varió el sistema
	param->Cosd = 0; // Este es el coseno de Delta que define la relación entre tópicos.
	param->dt = 0.01; // Paso temporal de iteración del sistema
	param->NormDif = sqrt(param->N*param->T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	param->CritCorte = pow(10,-4); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	param->testigos = fmin(param->N,500); // Esta es la cantidad de agentes de cada distancia que voy registrar
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int pasos = 0; // Esta es la cantidad de pasos de simulación que realiza el código
	
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
	
	// Matrices de mi sistema. Estas son la de Adyacencia, Adyacencia_vecinos, la de Superposición de Tópicos y la de vectores de opinión de los agentes.
	red->Ady = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Lista de vecinos de la red. Determina quienes están conectados con quienes
	red->Ady_vecinos = (int**) malloc( ( 2+param->N ) * sizeof(int*) ); // Complemento de lista de vecinos de la red. Determina quienes están conectados con quienes
	
	red->Ang = (double*) calloc( 2+param->T * param->T , sizeof(double) ); // Matriz simétrica de superposición entre tópicos.
	red->Opi = (double*) calloc( 2+param->T * param->N , sizeof(double) ); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// También hay un vector para guardar la diferencia entre el paso previo y el actual y un vector con los valores de la exponencial aplicada a los agentes,
	red->Dif = (double*) calloc( 2+param->T * param->N , sizeof(double) ); // Vector que guarda la diferencia entre dos pasos del sistema
	red->Exp = (double*) calloc( 2+param->T * param->N , sizeof(double) ); // Vector que guarda los cálculos de las exponenciales de cada agente.
	
	
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
	
	// Matriz de cálculo de exponencial para cada agetne. Es de tamaño N*T
	red->Exp[0] = param->N; // Pongo el número de filas en la primer coordenada
	red->Exp[1] = param->T; // Pongo el número de columnas en la segunda coordenada
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir dos archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	
	// En el cuarto me anoto la dirección del archivo de texto con la matriz de adyacencia.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char TextOpi[355];
	sprintf(TextOpi,"../Programas Python/Interes_actualizado/Datos/gm=%d/Opiniones_N=%d_Cosd=%.2f_kappa=%.2f_epsilon=%.2f_Iter=%d.file"
		, param->Gradomedio, param->N, param->Cosd, param->kappa, param->epsilon, iteracion);
	FILE *FileOpi = fopen(TextOpi,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	char TextTestigos[355];
	sprintf(TextTestigos,"../Programas Python/Interes_actualizado/Datos/gm=%d/Testigos_N=%d_Cosd=%.2f_kappa=%.2f_epsilon=%.2f_Iter=%d.file"
		, param->Gradomedio, param->N, param->Cosd, param->kappa, param->epsilon, iteracion);
	FILE *FileTestigos = fopen(TextTestigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char TextMatriz[355];
	sprintf(TextMatriz,"MARE/Erdos-Renyi/gm=%d/ErdosRenyi_N=%d_ID=%d.file"
		, param->Gradomedio, param->N, (int) iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *FileMatriz = fopen(TextMatriz,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*func_dinamica)(puntero_Matrices red, puntero_Parametros param) = &Dinamica_interes;
	double (*func_activacion)(puntero_Matrices red, puntero_Parametros param) = &Dinamica_sumatoria;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(red, 0, param->kappa); // Esto me inicializa mi matriz de opiniones 
	GenerarAng(red, param); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	Lectura_Adyacencia_Ejes(red, FileMatriz); // Leo el archivo de la red estática y lo traslado a la matriz de adyacencia
	fclose(FileMatriz); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	
	//################################################################################################################################

	
	// Acá voy a hacer las simulaciones de pasos previos del sistema
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(FileOpi,"Opiniones Iniciales\n");
	Escribir_d(red->Opi,FileOpi);
	
	fprintf(FileTestigos,"Opiniones Testigos\n");
	for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos,"%lf\t", red->Opi[ j*param->T+k+2 ] ); // Me guardo los valores de opinión de mis agentes testigo
	fprintf(FileTestigos,"\n");
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar
	for(int i=0; i<param->pasosprevios; i++){
		
		// Evolución
		RK4(red->Opi, func_dinamica, func_activacion, red, param);
		
		// Registro el estado actual en el array de OpinionesPrevias.
		for(int j=0; j<param->N*param->T; j++) *(OpiPrevias[i]+j+2) = red->Opi[j+2];
		
		// Actualización índices
		pasos++;
	}
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	fprintf(FileOpi,"Variación promedio \n");
	
	int iOpiPasado = 0; 
	
	while(contador < param->Iteraciones_extras){
		
		contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			// Evolución
			RK4(red->Opi, func_dinamica, func_activacion, red, param); // Itero los intereses
			
			// Cálculos derivados
			Delta_Vec_d(red->Opi, OpiPrevias[iOpiPasado % param->pasosprevios], red->Dif); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(int p=0; p<param->N*param->T; p++) *(OpiPrevias[iOpiPasado % param->pasosprevios] +p+2) = red->Opi[p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			red->Variacion_promedio = Norma_d(red->Dif) / param->NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			
			// Escritura
			if(pasos % 50 == 0){
				fprintf(FileOpi, "%lf\t", red->Variacion_promedio); // Guardo el valor de variación promedio
				for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos,"%lf\t", red->Opi[ j*param->T+k+2 ] ); // Me guardo los valores de opinión de mis agentes testigo
				fprintf(FileTestigos,"\n");
			}
			
			// Actualización de índices
			iOpiPasado++; // Avanzo el valor de iOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
			pasos++;
			
		}
		while( red->Variacion_promedio > param->CritCorte);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while(contador < param->Iteraciones_extras && red->Variacion_promedio <= param->CritCorte ){
			
			// Evolución
			RK4(red->Opi, func_dinamica, func_activacion, red, param); // Itero los intereses
			
			// Cálculos derivados
			Delta_Vec_d(red->Opi, OpiPrevias[ iOpiPasado % param->pasosprevios], red->Dif); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(int p=0; p<param->N*param->T; p++) *(OpiPrevias[ iOpiPasado % param->pasosprevios ] +p+2) = red->Opi[p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			red->Variacion_promedio = Norma_d(red->Dif) / param->NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			
			// Escritura
			if(pasos % 50 == 0){
				fprintf(FileOpi, "%lf\t", red->Variacion_promedio); // Guardo el valor de variación promedio 
				for(int j=0; j<param->testigos; j++) for(int k=0; k<param->T; k++) fprintf(FileTestigos, "%lf\t", red->Opi[ j*param->T +k+2 ] ); // Me guardo los valores de opinión de mis agentes testigo
				fprintf(FileTestigos,"\n");
			}
			
			// Actualización de índices
			iOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
			contador++; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
			pasos++;
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
	fprintf(FileOpi, "%d\n", pasos);
	fprintf(FileOpi, "Semilla\n");
	fprintf(FileOpi, "%ld\n", semilla);
	fprintf(FileOpi, "Matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	for(int i=0; i<param->N; i++) Escribir_i(red->Ady[i+2], FileOpi);
	
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	for(int i=0; i<param->pasosprevios; i++) free( OpiPrevias[i] );
	free( red->Ang );
	for(int i=0; i<param->N+2; i++) free( red->Ady[i] );
	free( red->Ady );
	for(int i=0; i<param->N+2; i++) free( red->Ady_vecinos[i] );
	free( red->Ady_vecinos );
	free( red->Opi );
	free( red->Dif );
	free( red->Exp );
	free( red );
	free( param );
	fclose( FileOpi );
	fclose( FileTestigos );
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tfin);
	Tiempo = tfin-tprin;
	// sleep(1);
	printf("Tarde %.1f segundos \n", Tiempo);
	
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