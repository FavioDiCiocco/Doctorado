//Este es el archivo para testeos

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>


double Random();
double Gaussiana(float f_mu, float f_sigma);


int main(int argc, char *argv[]){
	// Defino mis variables temporales para medir el tiempo que tarda el programa. También genero una nueva semilla
	time_t tt_prin,tt_fin;
	time(&tt_prin);
	srand(time(NULL));
	int i_tardanza;
	
	// Abro un archivo
	char s_nombre[255];
	sprintf(s_nombre,"Valores_Gaussiana");
	FILE* pa_archivo = fopen(s_nombre,"w");
	
	
	// Quiero probar si estoy usando bien el método que me pasó Sebas para generar números con una distribución normal
	double d_resultado;
	for(register int i_i=0; i_i<1000000; i_i++){
		d_resultado = Gaussiana(0,sqrt(2.5));
		fprintf(pa_archivo,"\t%lf",d_resultado);
	}
	
	
	// Ejecuto los comandos finales para medir el tiempo y liberar memoria
	fclose(pa_archivo);
	time(&tt_fin);
	i_tardanza = tt_fin-tt_prin;
	printf("Tarde %d segundos en terminar",i_tardanza);
		
	return 0;
}


//########################################################################################
//########################################################################################

// Esta función me genera un número random entre 0 y 1
double Random(){
	return ((double) rand()/(double) RAND_MAX);
}


// Esta función me da un valor tomado de una distribución gaussiana con valor medio mu y desviación sigma
double Gaussiana(float f_mu, float f_sigma){
	// Defino mis variables iniciales
	int i_n=100;
	double d_z=0;
	
	// Genero el número que voy a obtener de mi Gaussiana.
	// Para ser sincero, esto es un código legado del cual no comprendo la matemática involucrada.
	for(int i_i=0;i_i<i_n;i_i++) d_z += Random();
	d_z = sqrt(12*i_n) * (d_z/i_n-0.5);
	return d_z*f_sigma+f_mu;
}