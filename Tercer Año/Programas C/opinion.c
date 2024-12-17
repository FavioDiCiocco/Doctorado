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



void readNetwork(int iteracion){

    int i, n, l, *v_degree_aux;
    FILE *fin;
	char TextMatriz[355];
	sprintf(TextMatriz, "MARE/Erdos-Renyi/gm=10/ErdosRenyi_N=1000_ID=%d.file", (int) iteracion%100); // El 100 es porque tengo 100 redes creadas. Eso lo tengo que revisar si cambio el código
    // printf("Leemos la red...");
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
    // printf("Red leida.\n");


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


void free_all(double *network, double *new_network, double *Ang, double *Dif, double *Prom_Opi, double *distribucion){
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
	free(Ang);
	free(Dif);
	free(Prom_Opi);
    free(network);
	free(distribucion);
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
	for(int i=0; i<F; i++) for(int j=0; j<i; j++) *(Ang + i*C+j) = cosd; //
	for(int i=0; i<F; i++) *(Ang + i*C+i) = 1; // Esto me pone 1 en toda la diagonal
	for(int i=0; i<F; i++) for(int j=i+1; j<C; j++) *(Ang + i*C+j) = *(Ang + j*C+i); // Esta sola línea simetriza la matriz
}

// Esta función va a recibir un vector int y va a escribir ese vector en mi archivo.
void Escribir_i(int *vec,int F, int C, FILE *archivo){
	
	// Ahora printeo todo el vector en mi archivo
	for(int i=0; i<C*F; i++) fprintf(archivo,"%d\t",*( vec+i ));
	fprintf(archivo,"\n");
}

// Esta función va a recibir un vector double y va a escribir ese vector en mi archivo.
int Escribir_d(double *vec, int F, int C, FILE *archivo){
	
	// Ahora printeo todo el vector en mi archivo
	for(register int i=0; i<C*F; i++) fprintf( archivo,"%.6lf\t",*(vec+i) );
	fprintf(archivo,"\n");
	
	return 0;
}

// Esta función sirve para armar el histograma final de las opiniones
void Clasificacion(double* distribucion, double* network, int N, int T,int bines){
	
	double ancho = (double) 2/bines; // Este es el ancho de cada cajita en la que separo el espacio de opiniones.
	int fila,columna;
	
	for(int agente = 0; agente < N; agente++ ){
		columna = fmin(floor(network[agente*T+1]/ancho),bines-1);
		fila = fmin(floor(network[agente*T]/ancho),bines-1);
		distribucion[fila*bines+columna] += 1;
	}
	
	int sumatoria = 0;
	
	for(int i=0; i<bines*bines; i++) sumatoria += distribucion[i];
	for(int i=0; i<bines*bines; i++) distribucion[i] = distribucion[i]/sumatoria;
}

// Esta función me calcula la norma de un vector
double Norma_d(double *x, int F, int C){
	
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double norm, sum = 0;
	
	// Calculo la norma como la raíz cuadrada de la sumatoria de los cuadrados de cada coordenada.
	for(int i=0; i< C*F; ++i) sum += *(x+i) * ( *(x+i) );
	norm = sqrt(sum);
	return norm;
}

// Esta función me calcula la norma de un vector
double Norma_No_Ortogonal_d(double *vec, double *Sup, int F, int C){
	
	// Defino mis variables iniciales que son el resultado final, la suma de los cuadrados y el tamao de mi vector
	double norma = 0; // norma es la norma cuadrada del vector x
	double sumatoria = 0; // sumatoria es lo que iré sumando de los términos del denominador y después returneo
	
	// Yo voy a querer hacer el producto escalar en mi espacio no ortogonal. Para eso
	// uso mi matriz de Superposición, que contiene el ángulo entre todos los ejes
	// de mi espacio no ortogonal. Tengo que hacer el producto Vector*matriz*Vector.
	
	// Defino un puntero que guarde los valores del producto intermedio matriz*Vector.
	
	double *Inter;
	Inter = (double*) calloc(F, sizeof(double));
	
	// Armo el producto de matriz*Vector
	for(int fila=0; fila<F; fila++){
		sumatoria = 0; // La seteo a 0 para volver a iniciar la sumatoria
		
		for(int columna=0; columna<C; columna++) sumatoria += *( Sup +fila*C +columna ) * ( *(vec +columna ) );
		*( Inter+fila ) = sumatoria;
	}
	
	// Armo el producto Vector*Intermedios
	sumatoria = 0;
	for(int topico=0; topico<F; topico++) sumatoria += *( vec +topico ) * ( * ( Inter+topico ) );
	norma = sqrt(sumatoria);
	
	// Libero el puntero armado
	free(Inter);
	
	return norma;
}

void initialize_network(double *network, double *new_network, double K, int T){
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


void compute_derivative(double *network, double *slope, double *Ang, double K, double beta, double delta,int N, int T){

    int i, j;
    double neighbors_opinions;
	double distancia = 0, opiniones_superpuestas;
	
	// Armo un puntero a un vector en el cuál pondre la diferencia entre las opiniones del
	// agente i y el agente j.
	double *Vec_Dif;
	Vec_Dif = (double*) calloc(T, sizeof(double));

    /// Favio: No hagas mucho caso de la frikada de los mallocs, no ahorra nada de tiempo y no merece la pena
    static double *norm_factors = NULL;
    static double *tanhs = NULL; // OJO: De esta manera no podemos hacer un free
    if(!tanhs) tanhs = (double *)malloc(N*T*sizeof(double));
    if(!norm_factors) norm_factors = (double*)malloc(N*sizeof(double));


    for(i=0; i<N; i++){
        norm_factors[i] = 0;
    }

    for(i=0; i<N; i++)
    {
        for(int topico=0; topico<T; topico++)
		{
			opiniones_superpuestas = 0;
			for(int p=0; p<T; p++) opiniones_superpuestas += *(Ang +topico*T+p) * network[i*T+p];
			tanhs[i*T+topico] = tanh(opiniones_superpuestas); // Ahorra algo de tiempo calculando N tanhs en lugar de N*k tanhs
		}

        for(j=0; j<v_degree[i]; j++)
        {
            if(i<A[i][j]){ // Solo calculo el pow una vez por link, y lo guardo en dos lugares de la matriz weights[][].
                            //OJO: No son los pesos propiamente dichos, porque falta normalizarlos
                            // La normalización se hace con el vector 'norm_factors', que contiene al final los denominadores para todos los agentes
				
				// Armo el vector que apunta del agente i al agente vecino en el espacio de opiniones
				for(int topic=0; topic<T; topic++) *(Vec_Dif +topic) = network[i*T+topic] - network[A[i][j]*T+topic];
				distancia = Norma_No_Ortogonal_d(Vec_Dif,Ang,T,T);
				
				weights[i][j] = weights[A[i][j]][A_neigh[i][j]] = pow(distancia+ delta, -beta);

                norm_factors[i] += weights[i][j];
                norm_factors[A[i][j]] += weights[A[i][j]][A_neigh[i][j]];
            }
        }
    }


    for(i=0; i<N; i++)
	{
		for(int top=0; top<T; top++)
		{
			neighbors_opinions = 0;
			for(j=0; j<v_degree[i]; j++){
				neighbors_opinions += weights[i][j]*tanhs[A[i][j]*T+top]/norm_factors[i];
			}
			neighbors_opinions *= K;
			slope[i*T+top] = -network[i*T+top] + neighbors_opinions;
		}
    }
	
	free(Vec_Dif);
}


void actualize_network(double *network, double *new_network, double *Ang, double K, double beta, double delta, int N, int T){

    int i;

    static double *slope, *new_position;
    if(!slope) slope = (double *)malloc(N*T*sizeof(double));
    if(!new_position) new_position = (double*)malloc(N*T*sizeof(double));

    /// Step 1
    compute_derivative(network, slope, Ang, K, beta, delta, N, T);
    for(i=0; i<N*T; i++){
        new_network[i] = network[i] + slope[i]*dt/6.0;
        new_position[i] = network[i] + slope[i]*dt/2.0;
    }
    /// Step 2
    compute_derivative(new_position, slope, Ang, K, beta, delta, N, T);
    for(i=0; i<N*T; i++){
        new_network[i] += slope[i]*dt/3.0;
        new_position[i] = network[i] + slope[i]*dt/2.0;
    }
    /// Step 3
    compute_derivative(new_position, slope, Ang, K, beta, delta, N, T);
    for(i=0; i<N*T; i++){
        new_network[i] += slope[i]*dt/3.0;
        new_position[i] = network[i] + slope[i]*dt;
    }
    /// Step 4
    compute_derivative(new_position, slope, Ang, K, beta, delta, N, T);
    for(i=0; i<N*T; i++){
        new_network[i] += slope[i]*dt/6.0;
        //printf("\t %d cambio: %lf -> %lf\n", i, network[i], new_network[i]);
    }

    for(i=0; i<N*T; i++)
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
	
	T=2;
	K = 10.; // Esta amplitud regula la relación entre el término lineal y el término con tanh
	if(K != 0) delta = 2*K*0.001; // Una milesima parte de la diferencia maxima de opiniones
    else delta = 1E-20;
	
	beta = strtof(argv[1],NULL); // Esta es la potencia que determina el grado de homofilia.
	cosd = strtof(argv[2],NULL); // Este es el coseno de Delta que define la relación entre tópicos.
	int iteracion = strtol(argv[3],NULL,10); // Número de instancia de la simulación.
	
	// Leo la matriz de Adyacencia. Esto me fija N y L
	readNetwork(iteracion);
    mean_degree = L*2.0/N;
	
	// printf("Leí la red de adyacencia\n");
	
	// Variables para cortar rápido
	double NormDif = sqrt(N*T); // Este es el valor de Normalización de la variación del sistema, que me da la variación promedio de las opiniones.
	
	// Estos son unas variables que si bien podrían ir en el puntero red, son un poco ambiguas y no vale la pena pasarlas a un struct.
	double Variacion_promedio = 0.1; // Esto es la Variación promedio del sistema. Es cuanto cambia en promedio cada opinión
	double CritCorte = pow(10,-3); // Este valor es el criterio de corte. Con este criterio, toda variación más allá de la quinta cifra decimal es despreciable.
	int contador = 0; // Este es el contador que verifica que hayan transcurrido la cantidad de iteraciones extra
	int pasos_simulados = 0; // Esta variable me sirve para cortar si simulo demasiado tiempo.
	int ancho_ventana = 500; // Este es el ancho temporal que voy a tomar para promediar las opiniones de mis agentes.
	int Iteraciones_extras = 1500; // Este valor es la cantidad de iteraciones extra que el sistema tiene que hacer para cersiorarse que el estado alcanzado efectivamente es estable
	int bines = 42; // Esta es la cantidad de cajas en las que separar cada eje al construir la distribución de opiniones
	
	//#############################################################################################
	
	// Armo las matrices que necesito para el sistema
	
    double *network, *new_network, *Ang, *Dif, *Prom_Opi, *distribucion;
    network = (double*) calloc(N*T, sizeof(double));
    new_network = (double*) calloc(N*T, sizeof(double));
	Ang = (double*) calloc(T*T, sizeof(double));
	Dif = (double*) calloc(N*T, sizeof(double));
	Prom_Opi = (double*) calloc(2*N*T, sizeof(double));
	distribucion = (double*) calloc(bines*bines, sizeof(double));
	
	// Inicializo la red de opiniones y de superposición
	
	GenerarAng(Ang, T, cosd);
    initialize_network(network, new_network, K, T);
    
	//################################################################################################################################
	
	// Este archivo es el que guarda las opiniones del sistema mientras evoluciona
	char TextOpi[355];
	sprintf(TextOpi,"../Programas Python/Barrido_final/Extremo_polarizado/Opiniones_N=%d_kappa=10_beta=%.2f_cosd=%.2f_Iter=%d.file", N,beta, cosd, iteracion);
	FILE *FileOpi=fopen(TextOpi,"w"); // Con esto abro mi archivo y dirijo el puntero a él.

	//################################################################################################################################
	
	// Guardo la distribución inicial de las opiniones de mis agentes y preparo para guardar la Varprom.
	// fprintf(FileOpi,"Opiniones Iniciales\n");
	// Escribir_d(network,1,N*T,FileOpi);
	
	// Sumo el estado inicial de las opiniones de mis agentes en el vector de Prom_Opi. Guardo esto en la primer fila
	for(int j=0; j<N*T; j++) *(Prom_Opi +j) += *(network+j);
	
	for(step=0; step<ancho_ventana-1; step++){
		// Evolución
		actualize_network(network, new_network, Ang, K, beta, delta, N, T); // Se actualizan las opiniones
		for(int j=0; j<N*T; j++) *(Prom_Opi +j) += *(network+j);
	}
	
	// Promedio el valor de Prom_Opi al dividir por el tamaño de la ventana
	for(int j=0; j<N*T; j++) *(Prom_Opi +j) = *(Prom_Opi +j) / ancho_ventana;
	
	//################################################################################################################################
	
	// Realizo la simulación del modelo hasta que este alcance un estado estable
	// También preparo para guardar los valores de Varprom en mi archivo
	
	step = 0;
	
	// fprintf(FileOpi,"Variación promedio \n");
	
	while(contador < Iteraciones_extras && pasos_simulados < N_steps){
		
		contador = 0; // Inicializo el contador
		
		// Evoluciono el sistema hasta que se cumpla el criterio de corte
		do{
			
			// Evolución
			actualize_network(network, new_network, Ang, K, beta, delta, N, T); // Se actualizan las opiniones
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) += *(network+j);
			
			// Actualización de índices
			pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			if( pasos_simulados%ancho_ventana==0 ){
				// Mido la variación de promedios
				for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) = *(Prom_Opi +j+N*T) / ancho_ventana;
				for(int j=0; j<N*T; j++) *(Dif +j) = *(Prom_Opi +j+N*T) - *(Prom_Opi +j);
				Variacion_promedio = Norma_d(Dif,N,T) / NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Escritura
				// fprintf(FileOpi, "%lf\t", Variacion_promedio); // Guardo el valor de variación promedio
				
				// Reinicio los promedios
				for(int j=0; j<N*T; j++) *(Prom_Opi +j) = *(Prom_Opi +j+N*T);
				for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) = 0;
			}

		}
		while( Variacion_promedio > CritCorte && pasos_simulados < N_steps);
		
		// Ahora evoluciono el sistema una cantidad i_Itextra de veces. Le pongo como condición que si el sistema deja de cumplir la condición de corte, deje de evolucionar
		
		while( contador < Iteraciones_extras && Variacion_promedio <= CritCorte && pasos_simulados < N_steps){
			
			// Evolución
			actualize_network(network, new_network, Ang, K, beta, delta, N, T); // Se actualizan las opiniones
			// Sumo las opiniones en la segunda fila de Prom_Opi
			for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) += *(network+j);
			
			// Actualización de índices
			contador++; // Avanzo el contador para que el sistema haga una cantidad $i_Itextra de iteraciones extras
			pasos_simulados++; // Avanzo el contador de la cantidad de pasos simulados
			
			if( pasos_simulados%ancho_ventana==0 ){
				// Mido la variación de promedios
				for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) = *(Prom_Opi +j+N*T) / ancho_ventana;
				for(int j=0; j<N*T; j++) *(Dif +j) = *(Prom_Opi +j+N*T) - *(Prom_Opi +j);
				Variacion_promedio = Norma_d(Dif,N,T) / NormDif; // Calculo la suma de las diferencias al cuadrado y la normalizo.
				
				// Escritura
				// fprintf(FileOpi, "%lf\t", Variacion_promedio); // Guardo el valor de variación promedio
				
				// Reinicio los promedios
				for(int j=0; j<N*T; j++) *(Prom_Opi +j) = *(Prom_Opi +j+N*T);
				for(int j=0; j<N*T; j++) *(Prom_Opi +j+N*T) = 0;
			}
			
		}
		
		// Si el sistema evolucionó menos veces que la cantidad arbitraria, es porque rompió la condiciones de corte.
		// Por tanto lo vuelvo a hacer trabajar hasta que se vuelva a cumplir la condición de corte.
		// Si logra evolucionar la cantidad arbitraria de veces sin problemas, termino la evolución.
	}
	
	//################################################################################################################################
	
	// fprintf(FileOpi, "\n");
	// fprintf(FileOpi, "Opiniones finales\n");
	// Escribir_d(network,1,N*T,FileOpi);
	// fprintf(FileOpi,"Pasos Simulados\n");
	// fprintf(FileOpi,"%d\n",pasos_simulados);
	
	// Para armar la distribución de opiniones tengo que normalizar las opiniones y ubicarlas en la region [0,2]
	for(int i =0; i<N*T; i++) network[i] = network[i]/K + 1;
	Clasificacion(distribucion, network, N, T, bines);
	
	fprintf(FileOpi,"Distribución final\n");
	Escribir_d(distribucion,bines,bines,FileOpi);
	// Escritura final
	fprintf(FileOpi,"Semilla\n");
	fprintf(FileOpi,"%ld\n",semilla);
	fprintf(FileOpi, "Primeras filas de la Matriz de Adyacencia\n"); // Guardo esto para poder corroborar cuál es la Matriz de Adyacencia.
	for(int i=0; i<10; i++) Escribir_i(A[i],1,v_degree[i],FileOpi);

    free_all(network, new_network, Ang, Dif, Prom_Opi, distribucion);
	fclose(FileOpi);
	
	// Finalmente imprimo el tiempo que tarde en ejecutar todo el programa
	time(&tfin);
	Tiempo = tfin-tprin;
	printf("Tarde %.1f segundos \n",Tiempo);

    return 0;
}
