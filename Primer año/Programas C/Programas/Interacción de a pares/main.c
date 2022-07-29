// Acá viene el TP de Tesis. La idea es empezar a armar la red que voy a evaluar

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include"general.h"
#include"inicializar.h"
#include"avanzar.h"



int main(int argc, char *argv[]){
	// Empecemos con la base. Defino variables de tiempo para medir cuanto tardo y cosas básicas
	time_t tt_prin,tt_fin;
	time(&tt_prin);
	srand(time(NULL));
	float f_tardanza; // Este es el float que le paso al printf para saber cuanto tardé
	
	// Acá Defino los parámetros, asigno memoria a mis vectores y los inicializo.
		s_Param datos;
		s_Red red;
		
		// Parámetros de mi modelo. Esto va desde número de agentes hasta el paso temporal de integración.
		datos.i_N = strtol(argv[1],NULL,10); // Cantidad de agentes en el modelo
		datos.i_T = 2;  //strtol(argv[1],NULL,10); Antes de hacer esto, arranquemos con número fijo   // Cantidad de temas sobre los que opinar
		datos.f_Beta = 0.5; // Exponente que regula homofilia. Arranquemos con homofilia intermedia.
		datos.f_Pint = 0.6; // Probabilidad de que se forme un enlace entre dos agentes aleatorios.
		datos.f_K = strtof(argv[2],NULL); // Influencia social
		datos.f_dt = 0.001; // Paso temporal de iteración del sistema
		datos.f_alfa = 1; // Controversialidad de los tópicos. Arranquemos con controversialidad intermedia.
		datos.i_Mopi = 3; // Este es el valor de máxima opinión inicial del sistema
		datos.f_Tint = 20; // Este es el valor de tiempo total en el que voy a integrar mi sistema
		datos.d_NormDif = sqrt(datos.i_N*datos.i_T); // Este es el valor de Normalización de la variación del sistema, que me da la varaiación promedio de las opiniones.
		// datos.d_CritCorte = pow(10,-6); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
		// datos.i_Itextra = 1000; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
		// int i_contador = 0; // Esta variable se encarga de llevar la cuenta de las iteraciones extra que realiza mi sistema.
		
		// Matrices de mi sistema. Estas son la de Adyacencia, la de Superposición de Tópicos y la de vectores de opinión de los agentes
		red.pd_Ang = (double*) malloc((2+datos.i_T*datos.i_T)*sizeof(double)); // Matriz simétrica de superposición entre tópicos.
		red.pi_Ady = (int*) malloc((2+datos.i_N*datos.i_N)*sizeof(int)); // Matriz de adyacencia de la red. Determina quienes están conectados con quienes
		red.pd_Opi = (double*) malloc((2+datos.i_T*datos.i_N)*sizeof(double)); // Lista de vectores de opinión de la red, Tengo T elementos para cada agente.
		red.pd_PreOpi = (double*) malloc((2+datos.i_T*datos.i_N)*sizeof(double)); // Paso previo del sistema antes de iterar.
		red.pd_Diferencia = (double*) malloc((2+datos.i_T*datos.i_N)*sizeof(double)); // Paso previo del sistema antes de iterar.
		
		// Voy a abrir dos archivos ahora. Uno para la evolución de opiniones, otro para la evolución del error.
		char s_archivo1[255];
		sprintf(s_archivo1,"Datos_Evolucion_Opinion_N=%d_T=%d_K=%.3f",datos.i_N,datos.i_T,datos.f_K);
		FILE *pa_archivo1=fopen(s_archivo1,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
		
		char s_archivo2[255];
		sprintf(s_archivo2,"Datos_Variacion_Promedio_N=%d_T=%d_K=%.3f",datos.i_N,datos.i_T,datos.f_K);
		FILE *pa_archivo2=fopen(s_archivo2,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
		
		// Inicializo mis cinco "matrices".
		// Matriz de Adyacencia. Es de tamaño N*N
		for(register int i_i=0; i_i<datos.i_N*datos.i_N+2; i_i++) red.pi_Ady[i_i] = 0;
		red.pi_Ady[0] = datos.i_N; // Pongo el número de filas en la primer coordenada
		red.pi_Ady[1] = datos.i_N; // Pongo el número de columnas en la segunda coordenada
				
		// Matriz de Superposición de Tópicos. Es de tamaño T*T
		for(register int i_i=0; i_i<datos.i_T*datos.i_T+2; i_i++) red.pd_Ang[i_i] = 0;
		red.pd_Ang[0] = datos.i_T; // Pongo el número de filas en la primer coordenada
		red.pd_Ang[1] = datos.i_T; // Pongo el número de Columnas en la segunda coordenada
				
		// Matriz de vectores de opinión. Es de tamaño N*T
		for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Opi[i_i] = 0;
		red.pd_Opi[0] = datos.i_N; // Pongo el número de filas en la primer coordenada
		red.pd_Opi[1] = datos.i_T; // Pongo el número de columnas en la segunda coordenada
		
		// Matriz de vectores de opinión en el paso temporal Previo. Es de tamaño N*T
		for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_PreOpi[i_i] = 0;
		red.pd_PreOpi[0] = datos.i_N; // Pongo el número de filas en la primer coordenada
		red.pd_PreOpi[1] = datos.i_T; // Pongo el número de columnas en la segunda coordenada
		
		// Matriz de diferencia entre los vectores Opi y PreOpi. Es de tamaño N*T
		for(register int i_i=0; i_i<datos.i_N*datos.i_T+2; i_i++) red.pd_Diferencia[i_i] = 0;
		red.pd_Diferencia[0] = datos.i_N; // Pongo el número de filas en la primer coordenada
		red.pd_Diferencia[1] = datos.i_T; // Pongo el número de columnas en la segunda coordenada
				
		// Inicializo el Agente y Tópico a mirar. Esto no significa mucho porque después lo voy a cambiar.
		red.i_agente = 0;
		red.i_topico = 0;
		
		// Puntero a la función que define mi ecuación diferencial
		double (*pf_EcDin)(s_Red var, s_Param par) = &Din2;
		
	
	// Genero las redes de mi sistema
	GenerarOpi(red,datos); // Esto me inicializa mis vectores de opinión, asignándole a cada agente una opinión en cada tópico
	GenerarAdy(red,datos); // Esto me inicializa mi matriz de adyacencia, creando las conexiones de mi red
	GenerarAng(red); // Esto me inicializa mi matriz de superposición, definiendo el solapamiento entre tópicos.
	
	// Anoto los valores de mis tres matrices en mi archivo de evolución de opiniones
	fprintf(pa_archivo1, "\tMatriz de Adyacencia\n");
	Escribir_i(red.pi_Ady,pa_archivo1); // Matriz de Adyacencia
	fprintf(pa_archivo1, "\tMatriz de Superposicion\n");
	Escribir_d(red.pd_Ang,pa_archivo1); // Matriz de Superposición
	fprintf(pa_archivo1, "\tMatriz de Opiniones\n");
	Escribir_d(red.pd_Opi,pa_archivo1); // Matriz de Opinión
	
	// printf("Este es mi sistema antes de evolucionarlo\n");
	// Visualizar_d(red.pd_Opi);
	
	// Evolucionemos el sistema utilizando un mecanismo de corte
	while(i_contador<datos.i_Itextra){
		// Inicializo el contador
		i_contador = 0;
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			for(register int i_j=0; i_j<datos.i_N*datos.i_T; i_j++) red.pd_PreOpi[i_j+2] = red.pd_Opi[i_j+2];
			Iteracion(red,datos,pf_EcDin);
			Escribir_d(red.pd_Opi,pa_archivo1); // Matriz de Opinión
			Delta_Vec_d(red.pd_Opi,red.pd_PreOpi,red.pd_Diferencia); // Veo la diferencia entre el paso previo y el actual en las opiniones
			red.d_Varprom = Norma_d(red.pd_Diferencia)/datos.d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			fprintf(pa_archivo2,"\t%.12lf",red.d_Varprom);
		}
		while(red.d_Varprom > datos.d_CritCorte);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		while(i_contador < datos.i_Itextra && red.d_Varprom <= datos.d_CritCorte ){
			for(register int i_j=0; i_j<datos.i_N*datos.i_T; i_j++) red.pd_PreOpi[i_j+2] = red.pd_Opi[i_j+2];
			Iteracion(red,datos,pf_EcDin);
			Escribir_d(red.pd_Opi,pa_archivo1); // Matriz de Opinión
			Delta_Vec_d(red.pd_Opi,red.pd_PreOpi,red.pd_Diferencia); // Veo la diferencia entre el paso previo y el actual en las opiniones
			red.d_Varprom = Norma_d(red.pd_Diferencia)/datos.d_NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
			fprintf(pa_archivo2,"\t%.12lf",red.d_Varprom);
			i_contador +=1;
		}
		// Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	}
	
	// printf("Este es mi sistema final\n");
	// Visualizar_d(red.pd_Opi);

	// Libero los espacios dedicados a mis vectores y cierro mis archivos
	free(red.pd_Ang);
	free(red.pi_Ady);
	free(red.pd_Opi);
	free(red.pd_PreOpi);
	free(red.pd_Diferencia);
	fclose(pa_archivo1);
	fclose(pa_archivo2);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tt_fin);
	f_tardanza = tt_fin-tt_prin;
	printf("Tarde %.1f segundos \n",f_tardanza);
	
	
	return 0;
 }

