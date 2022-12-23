//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include<stdbool.h>


int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo);
int Distancia_agentes(int *pi_ady, int *pi_sep);
int Cantidad_agentes_conjuntos(int *pi_conj, int *pi_cant);
int Lista_testigos(int *pi_separacion,int *pi_cantidad,int *pi_testigos, int i_distmax, int i_testigos);
int Escribir_i(int *pi_vec, FILE *pa_archivo);
int Visualizar_i(int *pi_vec);

// Mi objetivo es construir una función que a partir de la matriz de adyacencia recorra la red e identifique a cada agente con
// la distancia a la que se encuentra del agente central.


int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tt_prin,tt_fin;
	time(&tt_prin);
	srand(time(NULL));
	int i_tardanza;
	
	//################################################################################################################################
	
	// Preparo los datos que son necesarios para la función pero que en realidad estarían definidos
	// por fuera en el código original.
	
	// Defino algunas variables iniciales
	int i_N = 1000; // Cantidad de agentes
	int i_distmax = 0; // Máxima distancia al primer nodo
	int i_testigos = 3; // Esta es la cantidad máxima de agentes a cada distancia que voy a tomar.
	int i_total_testigos = 0; // Esta es la cantidad de testigos de los que voy a guardar datos
	
	
	// Defino el puntero que tendrá la matriz de adyacencia.
	int *pi_adyacencia;
	pi_adyacencia = (int*) malloc((2+i_N*i_N)*sizeof(int));
	
	// Lo inicializo
	for(register int i_i = 0; i_i<2+i_N*i_N; i_i++) *(pi_adyacencia+i_i) = 0;
	*pi_adyacencia = i_N; // Fijo las filas
	*(pi_adyacencia+1) = i_N; // Fijo las columnas
	
	// Defino el puntero que tendrá las etiquetas de la distancia a la que se encuentra cada agente
	int *pi_separacion;
	pi_separacion = (int*) malloc((2+i_N)*sizeof(int));
	
	// Lo inicializo
	for(register int i_i = 0; i_i<2+i_N; i_i++) *(pi_separacion+i_i) = 0;
	*pi_separacion = 1; // Fijo las filas
	*(pi_separacion+1) = i_N; // Fijo las columnas
	
	// Preparo el puntero para levantar los datos de la matriz de adyacencia
	char s_archivo1[350]; // Defino el string
	sprintf(s_archivo1, "MARE/Random_Regulars/Random-regular_N=1000_ID=1.file");  // Asigno la dirección del archivo al string
	FILE *pa_archivo1 = fopen(s_archivo1,"r");  // Abro el archivo para lectura
	
	Lectura_Adyacencia(pi_adyacencia, pa_archivo1); // Levanto los datos de la matriz del archivo de texto
	fclose(pa_archivo1); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	//################################################################################################################################
	
	// Uso la función para etiquetar a los agentes según su distancia al primer agente.
	i_distmax = Distancia_agentes(pi_adyacencia, pi_separacion);
	
	// Inicializo el vector con la cantidad de agentes a cada distancia.
	int *pi_cantidad;
	pi_cantidad = (int*) malloc((2+(i_distmax+1))*sizeof(int)); // Le pongo tamaño i_distmax+1 porque los agentes están separados en distancias que van desde 0 a i_distmax.
	// Vector con la cantidad de agentes a cada distancia del primer agente
	for(register int i_i=0; i_i<(i_distmax+1)+2; i_i++) pi_cantidad[i_i] = 0; // Inicializo la matriz
	pi_cantidad[0] = 1; // Pongo el número de filas en la primer coordenada
	pi_cantidad[1] = i_distmax+1; // Pongo el número de columnas en la segunda coordenada
	// Cuento cuántos agentes tengo a cada distancia del primero
	for(register int i_i=0; i_i<i_N; i_i++) pi_cantidad[pi_separacion[i_i+2]+2]++;
	
	// Identifico la cantidad de agentes testigos que voy a necesitar
	for(int register i_distancia=0; i_distancia<i_distmax+1; i_distancia++) i_total_testigos += (int) fmin(i_testigos, pi_cantidad[i_distancia+2]);
	
	// Con la cantidad total de agentes testigos ahora puedo inicializar mi vector de testigos
	int *pi_testigos;
	pi_testigos = (int*) malloc((2+i_total_testigos)*sizeof(int));
	for(register int i_i=0; i_i<i_total_testigos; i_i++) pi_testigos[i_i+2] = 0; // Inicializo la matriz
	*pi_testigos = 1; // Pongo el número de filas en la primer coordenada
	*(pi_testigos+1) = i_total_testigos;// Pongo el número de columnas en la segunda coordenada
	
	Lista_testigos(pi_separacion, pi_cantidad, pi_testigos, i_distmax, i_testigos);
	
	printf("Los agentes testigos son:\n");
	Visualizar_i(pi_testigos);
	printf("Y sus distancias al nodo original son:\n");
	for(register int i_i=0; i_i<i_total_testigos; i_i++) printf("%d\t",*(pi_separacion+*(pi_testigos+i_i+2)+2));
	printf("\n");
	
	// // Para ver que esto funque, voy a guardar los datos en un archivo y listo. Primero abro el archivo
	// char s_archivo2[350]; // Defino el string
	// sprintf(s_archivo2, "../Programas Python/Ola_interes/categorizacion_prueba.file");  // Asigno la dirección del archivo al string
	// FILE *pa_archivo2 = fopen(s_archivo2,"w");  // Abro el archivo para escritura
	
	// // Con esto me guardo los datos en un archivo
	// fprintf(pa_archivo2,"Categorias de los agentes\n");
	// for(register int i_i = 0; i_i<i_N; i_i++) fprintf(pa_archivo2, "%d\t", i_i);
	// fprintf(pa_archivo2, "\n");
	// Escribir_i(pi_separacion, pa_archivo2);
	
	// Quiero ver cuál es la distancia máxima al agente inicial
	printf("La distancia máxima al agente inicial es %d\n", i_distmax);
	
	// Ejecuto los comandos finales para medir el tiempo y liberar memoria
	// fclose(pa_archivo2);
	free(pi_adyacencia);
	free(pi_separacion);
	free(pi_cantidad);
	free(pi_testigos);
	
	time(&tt_fin);
	i_tardanza = tt_fin-tt_prin;
	printf("Tarde %d segundos en terminar\n",i_tardanza);
	
	return 0;
}


//########################################################################################
//########################################################################################


// // Esta función es la que lee un archivo y me arma la matriz de Adyacencia
int Lectura_Adyacencia(int *pi_vec, FILE *pa_archivo){
	// Defino los enteros que voy a usar para leer el archivo y escribir sobre el vector.	
	int i_indice = 2;
	int i_salida = 0;
	
	// Leo la matriz de Adyacencia del archivo y la guardo en el vector de Adyacencia.
	while(fscanf(pa_archivo,"%d",&i_salida) != EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		*(pi_vec+i_indice) = i_salida;
		i_indice++;
	}
	
	// Aviso si hubo algún problema.
	if(fscanf(pa_archivo,"%d",&i_salida) != EOF){
		printf("La matriz del archivo es mas grande que tu vector\n");
		return 1;
	}
	if(fscanf(pa_archivo,"%d",&i_salida) == EOF && i_indice < *pi_vec * *(pi_vec+1)+2){
		printf("La matriz del archivo es mas chica que el vector\n");
		return 1;
	}
	
	return 0;
}


// Esta función recibe la matriz de adyacencia y a partir de eso coloca en el vector pi_sep la distancia de cada agente al agente cero.
int Distancia_agentes(int *pi_ady, int *pi_sep){
	// Defino las variables necesarias para mi función
	int i_distmax,i_F,i_restantes, i_distancia;
	i_F = *pi_ady; // Número de filas de la matriz de Adyacencia.
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
		*(pi_Marcados+i_i+2) = *(pi_ady+i_i+2);
		*(pi_sep+i_i+2) = *(pi_ady+i_i+2);
	}
	
	// Marco al primer agente con un número, sólo para que no sea visitado al pedo
	*(pi_sep+2) = -1;
	
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
					if(*(pi_ady+i_vecino+i_agente*i_F+2) == 1){
						if(*(pi_sep+i_vecino+2) == 0){ 
							*(pi_Marcados+i_vecino+2) = 1; // Esta línea me agrega el sujeto a visitar sólo si no estaba en el grupo
							*(pi_sep+i_vecino+2) = i_distancia; // Esta línea me marca al sujeto en el grupo, porque al final si ya había un uno ahí, simplemente lo vuelve a escribir.
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
	*(pi_sep+2) = 0; // La distancia del primer nodo a sí mismo es cero.
	i_distmax = i_distancia-1; // El while termina en distancia una unidad extra de la distancia recorrida.
	
	return i_distmax;
}


// Esta función recibe el vector separación de los agentes al nodo inicial, el de cantidad de agentes
// a cada distancia y el de testigos, y me agarra los primeros tres agentes que se encuentran a esa
// distancia. Si no hay tres, agarra los que haya.
int Lista_testigos(int *pi_separacion,int *pi_cantidad,int *pi_testigos, int i_distmax, int i_testigos){
	// Preparo las variables con las que inicio mi código
	int i_agente_guardar=0; // Esta variable representa a los agentes que voy a anotar para guardar sus datos
	int i_posicion_testigo=0; // Esta variable es la posición en el vector de Testigos a medida que voy completando el vector.
	
	// Hago todo el proceso de anotar agentes de cada una de las distancias. Me anoto i_testigos o pi_cantidad de agentes, lo que sea menor.
	for(register int i_distancia=0; i_distancia<i_distmax+1; i_distancia++){
		i_agente_guardar = 0;
		for(register int i_iteracion_testigo=0; i_iteracion_testigo<fmin(i_testigos, pi_cantidad[i_distancia+2]); i_iteracion_testigo++){
			while(pi_separacion[i_agente_guardar+2] != i_distancia) i_agente_guardar++;
			pi_testigos[i_posicion_testigo+2] = i_agente_guardar;
			i_agente_guardar++;
			i_posicion_testigo++;
		}
	}
	
	return 0;
}


// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
int Escribir_i(int *pi_vec, FILE *pa_archivo){
	// Defino las variables del tamao de mi vector
	int i_C,i_F;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"%d\t",*(pi_vec+i_i+2));
	fprintf(pa_archivo,"\n");
	
	return 0;
}


// Esta función es para observar los vectores int
int Visualizar_i(int *pi_vec){
	// Defino las variables que voy a necesitar.
	int i_F,i_C;
	i_F = *pi_vec;
	i_C = *(pi_vec+1);
	
	// Printeo mi vector
	for(register int i_i=0; i_i<i_F; i_i++){
		for(register int i_j=0; i_j<i_C; i_j++) printf("%d\t",*(pi_vec+i_i*i_C+i_j+2));
		printf("\n");
	}
	printf("\n");
	
	return 0;
}

