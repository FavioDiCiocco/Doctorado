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
int Escribir_i(int *pi_vec, FILE *pa_archivo);

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
	
	// Defino el puntero que tendrá las etiquetas de la distancia a la que se encuentra cada agente
	int *pi_;
	pi_separacion = (int*) malloc((2+i_N)*sizeof(int));
	
	// Lo inicializo
	for(register int i_i = 0; i_i<2+i_N; i_i++) *(pi_separacion+i_i) = 0;
	*pi_separacion = 1; // Fijo las filas
	*(pi_separacion+1) = i_N; // Fijo las columnas
	
	// Preparo el puntero para levantar los datos de la matriz de adyacencia
	char s_archivo1[350]; // Defino el string
	sprintf(s_archivo1, "./MARE/Random_Regulars/Random-regular_N=1000_ID=1.file");  // Asigno la dirección del archivo al string
	FILE *pa_archivo1 = fopen(s_archivo1,"r");  // Abro el archivo para lectura
	
	Lectura_Adyacencia(pi_adyacencia, pa_archivo1); // Levanto los datos de la matriz del archivo de texto
	fclose(pa_archivo1); // Aprovecho y cierro el puntero al archivo de la matriz de adyacencia
	
	//################################################################################################################################
	
	// Uso la función para etiquetar a los agentes según su distancia al primer agente.
	i_distmax = Distancia_agentes(pi_adyacencia, pi_separacion);
	
	// Para ver que esto funque, voy a guardar los datos en un archivo y listo. Primero abro el archivo
	char s_archivo2[350]; // Defino el string
	sprintf(s_archivo2, "../Programas Python/Ola_interes/categorizacion_prueba.file");  // Asigno la dirección del archivo al string
	FILE *pa_archivo2 = fopen(s_archivo2,"w");  // Abro el archivo para escritura
	
	// Con esto me guardo los datos en un archivo
	fprintf(pa_archivo2,"Categorias de los agentes\n");
	for(register int i_i = 0; i_i<i_N; i_i++) fprintf(pa_archivo2, "%d\t", i_i);
	fprintf(pa_archivo2, "\n");
	Escribir_i(pi_separacion, pa_archivo2);
	
	// Quiero ver cuál es la distancia máxima al agente inicial
	printf("La distancia máxima al agente inicial es %d\n", i_distmax);
	
	// Ejecuto los comandos finales para medir el tiempo y liberar memoria
	fclose(pa_archivo2);
	free(pi_adyacencia);
	free(pi_separacion);
	
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
	i_distancia = 2; // Este valor lo uso para asignar la distancia de los agentes al nodo principal.
	
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
		i_distancia++; // Paso a revisar a los vecinos que se encuentran a un paso más de distancia
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

