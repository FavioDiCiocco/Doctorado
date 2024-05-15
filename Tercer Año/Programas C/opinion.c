#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

int **A; // matriz de adyacencia
int **A_neigh;
double **weights; // Pesos de los enlaces
int *v_degree; // vector con el grado de cada nodo
int N, L; // n� de nodos y enlaces
double mean_degree;

#define N_steps 5000
#define dt 0.1

//#define RRN
//#define LAT
//#define BA
//#define FC
#define ER
//#define WS
//#define prob 1.0

//#define EXP_ATT


//Declaro las variables del generador aleatorio.
unsigned int irr[256];
unsigned int ir1;
unsigned char ind_ran,ig1,ig2,ig3;
#define NormRANu 2.3283063671E-10F //necesario para el generador aleatorio

void ini_ran(int SEMILLA)
{
    int INI, FACTOR, SUM;
    int i;
    printf("Semilla: %d\n", SEMILLA);
    srand(SEMILLA);

    INI=SEMILLA;
    FACTOR=67397;
    SUM=7364893;

    for(i=0; i<256; i++){
        INI=INI*FACTOR+SUM;
        irr[i]=INI;
    }
    ind_ran=ig1=ig2=ig3=0;
}

double ran(void)
{
    double r_random;

    ig1=ind_ran-25;
    ig2=ind_ran-55;
    ig3=ind_ran-61;

    irr[ind_ran]=irr[ig1]+irr[ig2];
    ir1=(irr[ind_ran]^irr[ig3]);
    ind_ran++;
    r_random=ir1*NormRANu; // por esto, r est� entre 0 y 1

    return r_random;
}


void create_header(FILE *f, double mean_opinion, double desv_opinion, double beta, double K){
    time_t current_time = time(NULL);
    char *fecha;

    fecha = ctime(&current_time);
    printf("Creando cabecera...\t");
    fprintf(f, "# MONTECARLO OPINION FORMATION WITH POT WEIGHTS ");
    fprintf(f, "(Mopinion.c) \n");
    fprintf(f, "#\t fecha: %s", fecha);
    fprintf(f, "#\n");
    #ifdef RRN
    fprintf(f, "# RRN, nodos: %d\t links: %d\t \n", N, L);
    #endif // RRN
    #ifdef BA
    fprintf(f, "# BA, nodos: %d\t links: %d\t \n", N, L);
    #endif // BA
    #ifdef LAT
    fprintf(f, "# LAT, nodos: %d\t links: %d\t \n", N, L);
    #endif // LAT
    #ifdef ER
    fprintf(f, "# ER, nodos: %d\t links: %d\t \n", N, L);
    #endif // ER
    #ifdef WS
    fprintf(f, "# Watts-Strogatz, nodos: %d\t links: %d\t prob: %lf \n", N, L, prob);
    #endif // WS
    fprintf(f, "#\n");
    fprintf(f, "# Opinion inicial: %lf +- %lf", mean_opinion, desv_opinion);
    #ifdef OUTERMOST
    fprintf(f, ", Vecinos exteriores");
    #endif // OUTERMOST
    fprintf(f, "\n#\n");
    fprintf(f, "# Constantes:\n");
    fprintf(f, "#\t K: %.3lf \t homophily (beta): %.3lf\n", K, beta);
    fprintf(f, "#\n");
    fprintf(f, "# Parametros simulacion:\n");
    fprintf(f, "# \t Runge-Kutta, dt: %.2lf \t N_steps: %d\n",  dt, N_steps);
    fprintf(f, "#\n");
    fprintf(f, "#==================================================\n");
    fprintf(f, "#\n");
    printf("Cabecera creada.\n");

}



void readNetwork(int iteracion){

    int i, n, l, *v_degree_aux;
    FILE *fin;
	char TextMatriz[355];
	sprintf(TextMatriz, "MARE/Erdos-Renyi/gm=10/ErdosRenyi_N=10000_ID=%d.file", (int) iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
    printf("Leemos la red...");
    #ifdef RRN
    //fin=fopen("networks/RR10000k=4.txt", "r");
    fin=fopen("networks/RR10000k=10.txt", "r");
    //fin=fopen("networks/RR2000k=10.txt", "r");
    //fin=fopen("networks/RR10000k=20.txt", "r");
    //fin=fopen("networks/RR10000k=50.txt", "r");
    //fin=fopen("networks/RR1000k=100.txt", "r");
    #endif // RRN
    #ifdef BA
    fin=fopen("networks/BA10000k=10.txt", "r");
    #endif // BA
    #ifdef ER
    fin=fopen(TextMatriz, "r");
    #endif // ER
    #ifdef LAT
    fin=fopen("networks/lat10000k=4.txt", "r");
    #endif // LAT
    #ifdef FC
    fin=fopen("networks/FC100.txt", "r");
    #endif // FC
    #ifdef WS
    char filename[100];
    sprintf(filename, "networks/WS1000k=10_p=%.1lf.txt", prob);
    fin=fopen(filename, "r");
    #endif // WS

    if(fin == NULL){
        printf("Error: no se ha abierto correctamente el archivo\n");
        exit(0);
    }

    fscanf(fin, "%d %d", &N, &L);
    //printf("%d %d\n", N, L);

    A=(int**)malloc(N*sizeof(int*)); // Asigno memoria a los punteros que apunten a cada fila de la matriz
    weights = (double**)malloc(N*sizeof(double*));
    v_degree=(int*)calloc(N, sizeof(int)); // Una componente para cada nodo
    v_degree_aux=(int*)calloc(N, sizeof(int));

    for(i=0; i<L; i++){ // leo tantas l�neas como enlaces haya en la red
        fscanf(fin, "%d %d", &n, &l);
        v_degree[n] += 1;
        v_degree[l] += 1;
    }

    for(i=0; i<N; i++){
        A[i]=(int*)calloc(v_degree[i], sizeof(int)); // En cada componente de A se declaran tantas componentes como enlaces tenga, no m�s
        weights[i] = (double*)calloc(v_degree[i], sizeof(double));
    }

    rewind(fin);
    fscanf(fin, "%d %d", &N, &L);

    for(i=0; i<L; i++){
        fscanf(fin, "%d %d", &n, &l);
        A[n][v_degree_aux[n]]=l;
        A[l][v_degree_aux[l]]=n;
        weights[n][v_degree_aux[n]] = 1.0; // Esto quiere decir que la componente ij de weights es el peso con el vecino j-�simo, NO con el agente j.
        weights[l][v_degree_aux[l]] = 1.0;
        v_degree_aux[n] += 1;
        v_degree_aux[l] += 1;
    }

    fclose(fin);
    free(v_degree_aux);
    printf("Red leida.\n");


    int j, neigh;
    A_neigh=(int**)malloc(N*sizeof(int*));

    for(i=0; i<N; i++){
        A_neigh[i]=(int*)calloc(v_degree[i], sizeof(int));

        for(j=0; j<v_degree[i]; j++){
            neigh = A[i][j];
            for(l=0; l<v_degree[neigh]; l++)
                if(A[neigh][l] == i) break;
            A_neigh[i][j] = l;
        }
    }
}


void free_all(double *network, double *new_network){
    int i;
    free(v_degree);
    for(i=0; i<N-1; i++){
        free(A[i]);
        free(A_neigh[i]);
        free(weights[i]);
    }
    free(A);
    free(A_neigh);
    free(weights);
    free(network);
    free(new_network);
}

// Esta función me genera un número random entre 0 y 1
double Random(){
	return ((double) rand()/(double) RAND_MAX);
}

// Esta función me genera la matriz de Superposicion del sistema. Esto es una matriz de T*T
void GenerarAng(double *Ang, int T, double cosd){
	
	// Obtengo las dimensiones de la matriz de Superposicion.
	int F,C;
	F = T;
	C = T;
	
	// Inicializo la matriz de Superposicion de mi sistema.
	for(int i=0; i<F; i++) for(int j=0; j<i; j++) *(Ang + i*C+j+2) = cosd; //
	for(int i=0; i<F; i++) *(Ang + i*C+i+2) = 1; // Esto me pone 1 en toda la diagonal
	for(int i=0; i<F; i++) for(int j=i+1; j<C; j++) *(Ang + i*C+j+2) = *(Ang + j*C+i+2); // Esta sola línea simetriza la matriz
}

// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
void Escribir_i(int *vec,int F, int C, FILE *archivo){
	
	// Ahora printeo todo el vector en mi archivo
	for(int i=0; i<C*F; i++) fprintf(archivo,"%d\t",*( vec+i ));
	fprintf(archivo,"\n");
}

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
int Escribir_d(double *pd_vector, int i_F, int i_C, FILE *pa_archivo){
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"%.6lf\t",*(pd_vector+i_i));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

// Esta función sirve para armar el histograma final de las opiniones
void Clasificacion(double* distribucion, double* network, int N,int bines){
	
	double ancho = (double) 2/bines;
	int indice = 0;
	
	
	for(int i=0; i<N; i++){
		indice = floor(network[i]/ancho);
		indice = fmin(indice, bines-1);
		distribucion[indice] += 1;
	}
	
	int sumatoria = 0;
	
	for(int i=0; i<bines; i++) sumatoria += distribucion[i];
	for(int i=0; i<bines; i++) distribucion[i] = distribucion[i]/sumatoria;
}

void initialize_network(double *network, double *new_network, double K){
    int i;
    double alea, max;
    int *order;

    order = (int*)malloc(N*T*sizeof(int));

    for(i=0; i<N*T; i++) order[i] = i;
    if(K<1) max = 1.0;
    else max = K;

    for(i=0; i<N*T; i++){
        alea = 2*max*Random();
        network[order[i]] = new_network[order[i]] = alea-max;

    }
    free(order);
}


void compute_derivative(double *network, double *slope, double K, double beta, double delta){

    int i, j;
    double neighbors_opinions;

    /// Favio: No hagas mucho caso de la frikada de los mallocs, no ahorra nada de tiempo y no merece la pena
    static double *norm_factors = NULL;
    static double *tanhs = NULL; // OJO: De esta manera no podemos hacer un free
    if(!tanhs) tanhs = (double *)malloc(N*sizeof(double));
    if(!norm_factors) norm_factors = (double*)malloc(N*sizeof(double));


    for(i=0; i<N; i++){
        norm_factors[i] = 0;
    }

    for(i=0; i<N; i++)
    {
        tanhs[i] = tanh(network[i]); // Ahorra algo de tiempo calculando N tanhs en lugar de N*k tanhs

        for(j=0; j<v_degree[i]; j++)
        {
            if(i<A[i][j]){ // Solo calculo el pow una vez por link, y lo guardo en dos lugares de la matriz weights[][].
                            //OJO: No son los pesos propiamente dichos, porque falta normalizarlos
                            // La normalización se hace con el vector 'norm_factors', que contiene al final los denominadores para todos los agentes
                weights[i][j] = weights[A[i][j]][A_neigh[i][j]] = pow(fabs(network[i]-network[A[i][j]]) + delta, -beta);

                norm_factors[i] += weights[i][j];
                norm_factors[A[i][j]] += weights[A[i][j]][A_neigh[i][j]];
            }
        }
    }


    for(i=0; i<N; i++)
	{
        neighbors_opinions = 0;
        for(j=0; j<v_degree[i]; j++){
            neighbors_opinions += weights[i][j]*tanhs[A[i][j]]/norm_factors[i];
        }
        neighbors_opinions *= K;
        slope[i] = -network[i] + neighbors_opinions;
    }
}


void actualize_network(double *network, double *new_network, double K, double beta, double delta){

    int i;

    static double *slope, *new_position;
    if(!slope) slope = (double *)malloc(N*sizeof(double));
    if(!new_position) new_position = (double*)malloc(N*sizeof(double));

    /// Step 1
    compute_derivative(network, slope, K, beta, delta);
    for(i=0; i<N; i++){
        new_network[i] = network[i] + slope[i]*dt/6.0;
        new_position[i] = network[i] + slope[i]*dt/2.0;
    }
    /// Step 2
    compute_derivative(new_position, slope, K, beta, delta);
    for(i=0; i<N; i++){
        new_network[i] += slope[i]*dt/3.0;
        new_position[i] = network[i] + slope[i]*dt/2.0;
    }
    /// Step 3
    compute_derivative(new_position, slope, K, beta, delta);
    for(i=0; i<N; i++){
        new_network[i] += slope[i]*dt/3.0;
        new_position[i] = network[i] + slope[i]*dt;
    }
    /// Step 4
    compute_derivative(new_position, slope, K, beta, delta);
    for(i=0; i<N; i++){
        new_network[i] += slope[i]*dt/6.0;
        //printf("\t %d cambio: %lf -> %lf\n", i, network[i], new_network[i]);
    }

    for(i=0; i<N; i++)
        network[i] = new_network[i];

}


void calculate_mean_var_lf(double *data, int N_comp, double *mean, double *desv){
    int i, conv_check;
    double sum, sum_squared;

    sum = 0;
    sum_squared = 0;
    *mean = 0; // Por si acaso
    conv_check = 0;

    for (i=0; i<N_comp; i++){
        sum += data[i];
        sum_squared += data[i]*data[i];
        if(data[i] != 0) conv_check += 1;
    }

    if(conv_check == 0){ // Entonces no podemos calcular correctamente la media
        *mean = 0;
        *desv = 0;
        return;
    }

    *mean = sum/(double)N_comp;
    //*desv = sqrt(((sum_squared/N_comp)-(*mean)*(*mean))/(double)N_comp);
    *desv = (sum_squared/N_comp)-(*mean)*(*mean);
    if(*desv<0) *desv = 0;
    else *desv = sqrt(*desv);
    //printf("mean: %lf desv: %lf (%lf)\n", *mean, *desv, (sum_squared/N_comp)-(*mean)*(*mean));
}


int belong(int *array, int N_comp, int number){
    int i;

    for(i=0; i<N_comp; i++){
        if(array[i] == number) return 1;
    }
    return 0;
}


double compute_bounds(double K){
    double opinion, old_opinion;

    opinion = 1;

    do{
        old_opinion = opinion;
        opinion += (-old_opinion) + K*tanh(old_opinion);
        //printf("Opinion: %lf\n", opinion);
    }while(fabs(opinion-old_opinion) > 0.000001);

    //printf("limite: %lf\n", opinion);
    return opinion*1.00001;
}


void compute_opinions(double *data, int N_comp, double *mean, double *desv){
    int i, conv_check;
    double sum, sum_squared;

    sum = 0;
    sum_squared = 0;
    *mean = 0; // Por si acaso
    conv_check = 0;

    for (i=0; i<N_comp; i++){
        sum += data[i];
        sum_squared += data[i]*data[i];
        if(data[i] != 0) conv_check += 1;
    }

    if(conv_check == 0){ // Entonces no podemos calcular correctamente la media
        *mean = 0;
        *desv = 0;
        return;
    }
    else if(conv_check == 1){
        *mean = sum/(double)N_comp;
        *desv = 0;
        return;
    }

    *mean = sum/(double)N_comp;
    *desv = (sum_squared/N_comp)-(*mean)*(*mean);
    if(*desv<0) *desv = 0;
    else *desv = sqrt(*desv);
    //printf("%d comp, mean: %lf desv: %lf (%lf)\n", conv_check, *mean, *desv, (sum_squared/N_comp)-(*mean)*(*mean));
}


#ifdef EXP_ATT
void compute_attentions(double *network, double *attentions){
    int i, j;
    double attention, norm_factor;

    for(i=0; i<N; i++){
        attention = 0;
        norm_factor = 0;
        for(j=0; j<v_degree[i]; j++){
            norm_factor += weights[i][j];
            if(network[i]*network[A[i][j]]<0) attention += weights[i][j];
        }
        attentions[i] = attention/norm_factor;
    }
}


void compute_exposures(double *network, double *exposures){
    int i, j;
    double exposure;

    for(i=0; i<N; i++){
        exposure = 0;
        for(j=0; j<v_degree[i]; j++){
            if(network[i]*network[A[i][j]]<0) exposure += 1;
        }
        exposures[i] = exposure *1.0/v_degree[i];;
    }
}
#endif // EXP_ATT


int main(int argc, char *argv[])
{
    //ini_ran(time(NULL)^getpid());
	
	// Empecemos con la base. Defino variables de tiempo para medir cuanto tardo y cosas básicas
	time_t tprin,tfin,semilla;
	time(&tprin);
	semilla = time(NULL);
	srand(semilla); // Voy a definir la semilla a partir de time(NULL);
	float Tiempo; // Este es el float que le paso al printf para saber cuanto tardé
	
	//#############################################################################################
	
	// Defino los parámetros de mi modelo. Esto va desde número de agentes hasta el paso temporal de integración.
	// Primero defino los parámetros que requieren un input.
	
	double K, beta, delta, cosd;
    int step, T;
	
	K = 10.; // Esta amplitud regula la relación entre el término lineal y el término con tanh
	if(K != 0) delta = 2*K*0.001; // Una milesima parte de la diferencia maxima de opiniones
    else delta = 1E-20;
	
	beta = strtof(argv[1],NULL); // Esta es la potencia que determina el grado de homofilia.
	cosd = strtof(argv[2],NULL); // Este es el coseno de Delta que define la relación entre tópicos.
	int iteracion = strtol(argv[3],NULL,10); // Número de instancia de la simulación.
	
	// Leo la matriz de Adyacencia. Esto me fija N y L
	readNetwork(iteracion);
    mean_degree = L*2.0/N;
	
	printf("Leí la red de adyacencia\n");
	
	// Variables para cortar rápido
	NormDif = sqrt(N*T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	int contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int pasos_maximos = 200000; // Esta es la cantidad de pasos máximos a simular
	int ancho_ventana = 500; // Este es el ancho temporal que voy a tomar para promediar las opiniones de mis agentes.
	int Iteraciones_extras = 1500; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	
	//#############################################################################################
	
	// Armo las matrices que necesito para el sistema
	
    double *network, *new_network, *Ang, *Dif, *Prom_Opi;
    network = (double*) calloc(N*T, sizeof(double));
    new_network = (double*) calloc(N*T, sizeof(double));
	Ang = (double*) calloc(T*T, sizeof(double));
	Dif = (double*) calloc(N*T, sizeof(double));
	Prom_Opi = (double*) calloc(2*N*T, sizeof(double));
	
	// Inicializo la red de opiniones y de superposición
	
	Generar_Ang(Ang, T, cosd);
    initialize_network(network, new_network, K);
    
	//################################################################################################################################
	
	// Este archivo es el que guarda las opiniones del sistema mientras evoluciona
	char TextOpi[355];
	sprintf(TextOpi,"../Programas Python/Comparacion_datos/Beta-Cosd/Opiniones_N=%d_kappa=10_beta=%.2f_cosd=%.2f_Iter=%d.file", N,beta, cosd, iteracion);
	FILE *FileOpi=fopen(TextOpi,"w"); // Con esto abro mi archivo y dirijo el puntero a él.

	//################################################################################################################################
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(FileOpi,"Opiniones Iniciales\n");
	Escribir_d(network,1,N*T,FileOpi);
	
	// Sumo el estado inicial de las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
	for(int j=0; j<N*T; j++) *(Prom_Opi +j+2) += *(network+j+2);
	
	for(step=0; step<ancho_ventana-1; step++){
		// Evolución
		actualize_network(network, new_network, K, beta, delta); // Se actualizan las opiniones
		for(int j=0; j<N*T; j++) *(Prom_Opi +j+2) += *(network+j+2);
	}
	
	// Promedio el valor de Prom_Opi al dividir por el tamaño de la ventana
	for(int j=0; j<N*T; j++) *(Prom_Opi +j+2) = *(Prom_Opi +j+2) / ancho_ventana;
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	step = 0;
	
	fprintf(FileOpi,"Variación promedio \n");
	
	while(contador < Iteraciones_extras && pasos_simulados < pasos_maximos){
		
		contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			
			// Evolución
			actualize_network(network, new_network, K, beta, delta); // Se actualizan las opiniones
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T+2) += *(network+j+2);
			
			// Actualización de índices
			pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
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
	
    for(step=0; step<N_steps; step++)
    {
        // Evolución
		actualize_network(network, new_network, K, beta, delta); // Se actualizan las opiniones
		
		if( step%ancho_ventana==0 ){
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
	
	/*
	// Tomo el array de opiniones, lo normalizo, armo el histograma de agentes a lo largo de la recta de opiniones
	// y después lo escribo en mi archivo de opiniones.
	int bines = 42;
	double *distribucion;
	distribucion = (double*) calloc(bines, sizeof(double));
	*/
	// Antes de clasificar mis agentes, normalizo los valores de mi red y los desplazo para que sean todos positivos.
	fprintf(FileOpi, "Opiniones finales\n");
	Escribir_d(network,1,N*T,FileOpi);
	
	for(int i = 0; i<N; i++) network[i] = network[i]/K + 1;
	
	Clasificacion(distribucion, network, N, bines);
	
	// Escritura final
	fprintf(FileOpi,"Distribución final\n");
	Escribir_d(distribucion,1,bines,FileOpi);
	fprintf(FileOpi,"Pasos Simulados\n");
	fprintf(FileOpi,"%d\n",N_steps);
	fprintf(FileOpi,"Semilla\n");
	fprintf(FileOpi,"%ld\n",semilla);
	fprintf(FileOpi, "Primeras filas de la Matriz de Adyacencia\n"); // Guardo esto para poder corroborar cuál es la Matriz de Adyacencia.
	for(int i=0; i<10; i++) Escribir_i(A[i],1,v_degree[i],FileOpi);

    #ifdef EXP_ATT
    double *exposures, *attentions, *v_trash;

    exposures = (double*)calloc(N,sizeof(double));
    attentions = (double*)calloc(N, sizeof(double));
    v_trash = (double*)calloc(N, sizeof(double));

    compute_derivative(network, v_trash, K, beta, delta); // Para calcular los pesos de la situacion final, a falta del factor de normalizacion
    compute_attentions(network, attentions);
    compute_exposures(network, exposures);

    FILE *f_inter;
    sprintf(filename_results, "results/%s_%d_inter_measures_K=%.2lf_b=%.2lf.txt", network_type, N, K, beta);
    f_inter = fopen(filename_results, "wt");
    create_header(f_inter, initial_opinion, initial_desv_opinion, beta, K);
    fprintf(f_inter, "# node opinion exposure attention\n");

    for(i=0; i<N; i++){
        fprintf(f_inter, "%d %lf %lf %lf\n", i, network[i], exposures[i], attentions[i]);
    }
    fclose(f_inter);

    free(v_trash);
    free(exposures);
    free(attentions);
    #endif // EXP_ATT


    free_all(network, new_network);
	free(distribucion);
	fclose(FileOpi);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tfin);
	Tiempo = tfin-tprin;
	printf("Tarde %.1f segundos \n",Tiempo);

    return 0;
}
