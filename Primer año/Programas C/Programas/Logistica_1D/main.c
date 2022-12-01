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
	ps_datos->d_alfa = strtof(argv[2],NULL); // Controversialidad de los tópicos
	ps_datos->d_umbral = strtof(argv[3],NULL); // Este es el umbral que determina si el interés del vecino puede generarme más interés.
	int i_iteracion = strtol(argv[4],NULL,10); // Número de instancia de la simulación.
	
	// Los siguientes son los parámetros que están dados en los structs
	ps_datos->i_T = 1;  //strtol(argv[1],NULL,10); Antes de hacer esto, arranquemos con número fijo   // Cantidad de temas sobre los que opinar
	ps_datos->i_Itextra = 40; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	ps_datos->i_pasosprevios = 20; // Elegimos 20 de manera arbitraria con Pablo y Sebas. Sería la cantidad de pasos hacia atrás que miro para comparar cuanto varió el sistema
	ps_datos->d_Cosangulo = 0; // Este es el coseno de Delta que define la relación entre tópicos.
	ps_datos->d_dt = 0.01; // Paso temporal de iteración del sistema
	ps_datos->d_NormDif = sqrt(ps_datos->i_N*ps_datos->i_T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	ps_datos->d_CritCorte = pow(10,-4); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	ps_datos->d_amp = 1; // Este es el amplificador de la presión social. Este parámetro lo voy a sacar posiblemente
		
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int i_contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int i_testigos = 6; // Este es el número de testigos que registraré. Voy a registrar los testigos de 0 a 9
		
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
	ps_red->pi_Ady = (int*) malloc((2+ps_datos->i_N*ps_datos->i_N)*sizeof(int)); // Matriz de adyacencia de la red. Determina quienes están conectados con quienes
	ps_red->pd_Ang = (double*) malloc((2+ps_datos->i_T*ps_datos->i_T)*sizeof(double)); // Matriz simétrica de superposición entre tópicos.
	ps_red->pd_Opi = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
	
	// También hay una matriz de paso posterior del sistema, un vector para guardar la diferencia entre 
	// el paso previo y el actual y un vector con la actividad de cada uno d elos agentes
	ps_red->pd_OpiPosterior = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Paso previo del sistema antes de iterar.
	ps_red->pd_Diferencia = (double*) malloc((2+ps_datos->i_T*ps_datos->i_N)*sizeof(double)); // Vector que guarda la diferencia entre dos pasos del sistema
	
	// Inicializo mis cinco "matrices".
	// Matriz de Adyacencia. Es de tamaño N*N
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_N+2; i_i++) ps_red->pi_Ady[i_i] = 0; // Inicializo la matriz
	ps_red->pi_Ady[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pi_Ady[1] = ps_datos->i_N; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de Superposición de Tópicos. Es de tamaño T*T
	for(register int i_i=0; i_i<ps_datos->i_T*ps_datos->i_T+2; i_i++) ps_red->pd_Ang[i_i] = 0;
	ps_red->pd_Ang[0] = ps_datos->i_T; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Ang[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
			
	// Matriz de vectores de opinión. Es de tamaño N*T
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_T+2; i_i++) ps_red->pd_Opi[i_i] = 0; // Inicializo la matriz
	ps_red->pd_Opi[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Opi[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de vectores de opinión en el paso temporal Posterior. Es de tamaño N*T
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_T+2; i_i++) ps_red->pd_OpiPosterior[i_i] = 0; // Inicializo la matriz
	ps_red->pd_OpiPosterior[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_OpiPosterior[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
	for(register int i_i=0; i_i<ps_datos->i_N*ps_datos->i_T+2; i_i++) ps_red->pd_Diferencia[i_i] = 0; // Inicializo la matriz
	ps_red->pd_Diferencia[0] = ps_datos->i_N; // Pongo el número de filas en la primer coordenada
	ps_red->pd_Diferencia[1] = ps_datos->i_T; // Pongo el número de columnas en la segunda coordenada
	
	//################################################################################################################################
	
	// Abro los archivos en los que guardo datos y defino mi puntero a función.
	
	// Voy a abrir tres archivos. En el primero guardo la opinión inicial, la Varprom, la opinión final y la semilla
	// En el segundo me anoto la evolución de las opiniones de los testigos.
	// En el tercero me anoto la dirección del archivo de texto con la matriz de adyacencia.
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char s_archivo1[355];
	sprintf(s_archivo1,"../Programas Python/Logistica_1D/Opiniones_alfa=%.1f_N=%d_umbral=%.1f_Iter=%d.file"
		,ps_datos->d_alfa,ps_datos->i_N,ps_datos->d_umbral,i_iteracion);
	FILE *pa_archivo1=fopen(s_archivo1,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	char s_archivo2[355];
	sprintf(s_archivo2,"../Programas Python/Logistica_1D/Testigos_alfa=%.1f_N=%d_umbral=%.1f_Iter=%d.file"
		,ps_datos->d_alfa,ps_datos->i_N,ps_datos->d_umbral,i_iteracion);
	FILE *pa_archivo2=fopen(s_archivo2,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que levanta los datos de la matriz de Adyacencia de las redes generadas con Python
	char s_mady[355];
	sprintf(s_mady,"MARE/Erdos-Renyi/ErdosRenyi_N=%d_ID=%d.file"
		,ps_datos->i_N,(int) i_iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
	FILE *pa_mady=fopen(s_mady,"r");
	
	// Puntero a la función que define mi ecuación diferencial
	double (*pf_EcDin)(ps_Red var, ps_Param par) = &Din2;
	
	//################################################################################################################################
	
	// Genero los datos de las matrices de mi sistema
	
	GenerarOpi(ps_red, ps_datos); // Esto me inicializa mis vectores de opinión, asignándole a cada agente una opinión en cada tópico
	GenerarAng(ps_red, ps_datos); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	Lectura_Adyacencia(ps_red->pi_Ady, pa_mady);
	
	fclose(pa_mady); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	
	//################################################################################################################################

	// Acá voy a hacer las simulaciones de pasos previos del sistema y también voy a evolucionar el sistema hasta tener una red conexa.
	// Primero que nada arranco guardando las opiniones iniciales.
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(pa_archivo1,"Opiniones Iniciales\n");
	Escribir_d(ps_red->pd_Opi,pa_archivo1);
	
	fprintf(pa_archivo2,"Opiniones Testigos\n");
	
	// Hago los primeros pasos del sistema para tener estados previos con los que comparar	
	for(register int i_i=0; i_i<ps_datos->i_pasosprevios; i_i++){
		Iteracion(ps_red->pd_Opi,ps_red,ps_datos,pf_EcDin); // Itero mi sistema
		// Me guardo los valores de opinión de mis agentes testigo
		for(register int i_j=0; i_j<i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_archivo2,"%lf\t",ps_red->pd_Opi[i_j*ps_datos->i_T+i_k+2]);
		fprintf(pa_archivo2,"\n");
		// Registro el estado actual en el array de OpinionesPrevias.
		for(register int i_j=0; i_j<ps_datos->i_N*ps_datos->i_T; i_j++) *(ap_OpinionesPrevias[i_i]+i_j+2) = ps_red->pd_Opi[i_j+2];
	}
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	fprintf(pa_archivo1,"Variación promedio \n");
	
	int i_IndiceOpiPasado = 0; 
	
	while(i_contador < ps_datos->i_Itextra){
		
		i_contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			// Evolución
			Iteracion(ps_red->pd_Opi,ps_red,ps_datos,pf_EcDin); // Itero mi sistema
			// Cálculos derivados
			Delta_Vec_d(ps_red->pd_Opi,ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios],ps_red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<ps_datos->i_N*ps_datos->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios]+i_p+2) = ps_red->pd_Opi[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			ps_red->d_Varprom = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			// Escritura
			fprintf(pa_archivo1, "%lf\t",ps_red->d_Varprom); // Guardo el valor de variación promedio
			for(register int i_j=0; i_j<i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_archivo2,"%lf\t",ps_red->pd_Opi[i_j*ps_datos->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
			fprintf(pa_archivo2,"\n");
			// Actualización de índices
			i_IndiceOpiPasado++; // Avanzo el valor de IndiceOpiPasado para que las comparaciones entre pasos se mantengan a distancia $i_pasosprevios
		}
		while( ps_red->d_Varprom > ps_datos->d_CritCorte);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while(i_contador < ps_datos->i_Itextra && ps_red->d_Varprom <= ps_datos->d_CritCorte ){
			// Evolución
			Iteracion(ps_red->pd_Opi,ps_red,ps_datos,pf_EcDin); // Itero mi sistema
			// Cálculos derivados
			Delta_Vec_d(ps_red->pd_Opi,ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios],ps_red->pd_Diferencia); // Veo la diferencia entre $i_pasosprevios pasos anteriores y el actual en las opiniones
			for(register int i_p=0; i_p<ps_datos->i_N*ps_datos->i_T; i_p++) *(ap_OpinionesPrevias[i_IndiceOpiPasado%ps_datos->i_pasosprevios]+i_p+2) = ps_red->pd_Opi[i_p+2]; // Me guardo el estado actual en la posición correspondiente de ap_OpinionesPrevias
			ps_red->d_Varprom = Norma_d(ps_red->pd_Diferencia)/ps_datos->d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			// Escritura
			fprintf(pa_archivo1, "%lf\t",ps_red->d_Varprom); // Guardo el valor de variación promedio 
			for(register int i_j=0; i_j<i_testigos; i_j++) for(register int i_k=0; i_k<ps_datos->i_T; i_k++) fprintf(pa_archivo2,"%lf\t",ps_red->pd_Opi[i_j*ps_datos->i_T+i_k+2]); // Me guardo los valores de opinión de mis agentes testigo
			fprintf(pa_archivo2,"\n");
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
	fprintf(pa_archivo1,"\n");
	fprintf(pa_archivo1,"Opiniones finales\n");
	Escribir_d(ps_red->pd_Opi,pa_archivo1);
	fprintf(pa_archivo1,"matriz de Adyacencia\n"); // Guardo esto para poder comprobar que la red sea conexa.
	Escribir_i(ps_red->pi_Ady,pa_archivo1);
	fprintf(pa_archivo1,"Semilla\n");
	fprintf(pa_archivo1,"%ld\n",semilla);
	
	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	for(register int i_i=0; i_i<ps_datos->i_pasosprevios; i_i++) free(ap_OpinionesPrevias[i_i]);
	free(ps_red->pd_Act);
	free(ps_red->pd_Ang);
	free(ps_red->pi_Ady);
	free(ps_red->pd_Opi);
	free(ps_red->pd_OpiPosterior);
	free(ps_red->pd_Diferencia);
	free(ps_red);
	free(ps_datos);
	fclose(pa_archivo1);
	fclose(pa_archivo2);
	
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