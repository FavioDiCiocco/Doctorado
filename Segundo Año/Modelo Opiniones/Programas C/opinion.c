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

#define N_steps 200000
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



void readNetwork(void){

    int i, n, l, *v_degree_aux;
    FILE *fin;
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
    fin=fopen("ER1000k=8.file", "r");
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

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
int Escribir_d(double *pd_vector, int i_F, int i_C, FILE *pa_archivo){
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i_i=0; i_i<i_C*i_F; i_i++) fprintf(pa_archivo,"%.6lf\t",*(pd_vector+i_i));
	fprintf(pa_archivo,"\n");
	
	return 0;
}

void initialize_network(double *network, double *new_network, double K){
    int i;
    double alea, max;
    int *order;

    order = (int*)malloc(N*sizeof(int));

    for(i=0; i<N; i++) order[i] = i;
    if(K<1) max = 1.0;
    else max = K;

    for(i=0; i<N; i++){
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


int main()
{
    //ini_ran(time(NULL)^getpid());
	
	// Empecemos con la base. Defino variables de tiempo para medir cuanto tardo y cosas básicas
	time_t tt_prin,tt_fin,semilla;
	time(&tt_prin);
	semilla = 1702971081;
	srand(semilla); // Voy a definir la semilla a partir de time(NULL);
	float f_tardanza; // Este es el float que le paso al printf para saber cuanto tardé

    readNetwork();

    mean_degree = L*2.0/N;
    printf("N: %d L: %d\n", N, L);

    double *network, *new_network;
    network = (double*)calloc(N, sizeof(double));
    new_network = (double*)calloc(N, sizeof(double));

    double K, beta, delta;
    int i,step;
    double mean_opinion, desv_opinion;
    double initial_opinion, initial_desv_opinion;

    beta = 0.9;
    K = 10.;

    if(K != 0) delta = 2*K*0.001; // Una milesima parte de la diferencia maxima de opiniones
    else delta = 1E-20;

    initialize_network(network, new_network, K);
    calculate_mean_var_lf(network, N, &mean_opinion, &desv_opinion);
    initial_opinion = mean_opinion;
    initial_desv_opinion = desv_opinion;

    printf("Opinion inicial: %lf +- %lf\n", initial_opinion, initial_desv_opinion);
	
	// Este archivo es el que guarda la Varprom del sistema mientras evoluciona
	char s_Opiniones[355];
	sprintf(s_Opiniones,"../Programas Python/Evolucion_temporal/1D_Hugo/Opiniones_N=1000_kappa=10_beta=0.9_cosd=0_Iter=51.file");
	FILE *pa_Opiniones=fopen(s_Opiniones,"w"); // Con esto abro mi archivo y dirijo el puntero a él.
	
	// Este archivo es el que guarda las opiniones de todos los agentes del sistema.
	char s_Testigos[355];
	sprintf(s_Testigos,"../Programas Python/Evolucion_temporal/1D_Hugo/Testigos_N=1000_kappa=10_beta=0.9_cosd=0_Iter=51.file");
	FILE *pa_Testigos=fopen(s_Testigos,"w"); // Con esto abro mi archivo y dirijo el puntero a él.

	//################################################################################################################################
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	fprintf(pa_Opiniones,"Opiniones Iniciales\n");
	Escribir_d(network,1,N,pa_Opiniones);
	
	// Me guardo los valores de opinión de mis agentes testigos (Arranco abajo por temas de numeración)
	fprintf(pa_Testigos,"Opiniones Testigos\n");
	
	
	/*
    FILE *f_opinions;
    char filename_results[1000], filename_hist[100], filename_hist_weights[100], network_type[10];


    #ifdef RRN
    sprintf(network_type, "RRN");
    #endif // RRN
    #ifdef BA
    sprintf(network_type, "BA");
    #endif // BA
    #ifdef LAT
    sprintf(network_type, "LAT");
    #endif // LAT
    #ifdef RGG
    sprintf(network_type, "RGG");
    #endif // RGG
    #ifdef FC
    sprintf(network_type, "FC");
    #endif // FC
    #ifdef ER
    sprintf(network_type, "ER");
    #endif // ER
    #ifdef WS
    sprintf(network_type, "WS");
    #endif // WS

    sprintf(filename_results, "results/%s_opinions_time.txt", network_type);
    f_opinions = fopen(filename_results, "wt");
    create_header(f_opinions, initial_opinion, initial_desv_opinion, beta, K);
    fprintf(f_opinions, "# step mean_opinion desv_opinion\n");
	*/
	
	fprintf(pa_Opiniones,"Variación promedio \n");
	fprintf(pa_Opiniones,"Vacío \n");
	
    for(step=0; step<N_steps; step++)
    {
        // Evolución
		actualize_network(network, new_network, K, beta, delta); // Se actualizan las opiniones
        compute_opinions(network, N, &mean_opinion, &desv_opinion); // Se calcula la opinion media y desv. estandar
	
		// Escritura
        if(step %100 == 0) Escribir_d(network,1,N,pa_Testigos);
		
    }
	
	// Escritura final
	fprintf(pa_Opiniones,"Opiniones finales\n");
	Escribir_d(network,1,N,pa_Opiniones);
	fprintf(pa_Opiniones,"Pasos Simulados\n");
	fprintf(pa_Opiniones,"%d\n",N_steps);
	fprintf(pa_Opiniones,"Semilla\n");
	fprintf(pa_Opiniones,"%ld\n",semilla);

	/*
    printf("Opinion final: %lf +- %lf\n", mean_opinion, desv_opinion);
    fclose(f_opinions);
	*/

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
	fclose(pa_Opiniones);
	fclose(pa_Testigos);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tt_fin);
	f_tardanza = tt_fin-tt_prin;
	printf("Tarde %.1f segundos \n",f_tardanza);

    return 0;
}
