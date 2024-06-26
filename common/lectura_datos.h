#ifndef LECTURA_DATOS_H
#define LECTURA_DATOS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
@brief lee en I el fichero .vox, que debe ser una volumen 3D en escala de grises (unsigned char) de tamaño MxMxM
       el buffer I no debe tener la memoria reservada porque se reserva en esta función
*/
void leer_volumen3D_vox(char* fichero_vox, unsigned char** I, int *M) {
	FILE* vox;
	int x,y,z;

	#ifdef _WIN32
	fopen_s(&vox, fichero_vox,"rb");
	#elif __linux__
	vox = fopen(fichero_vox,"rb");
	#endif

	if (vox == NULL) {
		printf("Error: no se puede abrir el fichero: %s", fichero_vox);
		exit(0);
	} 

	/* se lee la cabecera con el tamaño del volumen en las direcciones X,Y y Z */
	if (fread (&x,sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread (&y,sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread (&z,sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if ((x != y) || (x != z) || (y != z)) {
		printf("Error: la dimensión en X, y y Z debe ser igual");
		exit(0);
	}

	/* se lee la imagen */
	*M = x;
	if (!(*I = (unsigned char *) malloc(x * y * z * sizeof(unsigned char)))) {
		printf("Error: no se puede reservar la memoria para el volumen");
		exit(0);
	}
	
	if (fread (*I, sizeof(unsigned char), x * y * z, vox) != (x * y * z)) {
		printf("Error: no se puede leer el volumen");
		exit(0);
	}

	fclose(vox);
}


/**
@brief lee en I el fichero .vox, que debe ser una volumen 3D en escala de grises (unsigned char) de tamaño MxMxM
	   el buffer I no debe tener la memoria reservada porque se reserva en esta función
*/
void leer_volumen4D_vox(char* fichero_vox, unsigned char** I, int* M) {
	FILE* vox;
	int x, y, z, t;

#ifdef _WIN32
	fopen_s(&vox, fichero_vox, "rb");
#elif __linux__
	vox = fopen(fichero_vox, "rb");
#endif

	if (vox == NULL) {
		printf("Error: no se puede abrir el fichero: %s", fichero_vox);
		exit(0);
	}

	/* se lee la cabecera con el tamaño del volumen en las direcciones X,Y,Z,T */
	if (fread(&x, sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread(&y, sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread(&z, sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread(&t, sizeof(unsigned int), 1, vox) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if ((x != y) || (x != z) || (y != z) || (x != t)) {
		printf("Error: la dimensión en X, y y Z debe ser igual");
		exit(0);
	}

	/* se lee la imagen */
	*M = x;
	if (!(*I = (unsigned char*)malloc(x * y * z * t * sizeof(unsigned char)))) {
		printf("Error: no se puede reservar la memoria para el volumen");
		exit(0);
	}

	if (fread(*I, sizeof(unsigned char), x * y * z * t, vox) != (x * y * z * t)) {
		printf("Error: no se puede leer el volumen");
		exit(0);
	}

	fclose(vox);
}

/**
@brief lee en I el fichero ebmp, que debe ser una imagen en escala de grises (unsigned char ) de tamaño MxM
       el buffer I no debe tener la memoria reservada porque se reserva en esta función
*/
void leer_imagen_ebmp(char* fichero_ebmp, unsigned char** I, int *M) {
	FILE* ebmp;
	int f,c;

	#ifdef _WIN32
	fopen_s(&ebmp, fichero_ebmp,"rb");
	#elif __linux__
	ebmp = fopen(fichero_ebmp,"rb");
	#endif

	if (ebmp == NULL) {
		printf("Error: no se puede abrir el fichero: %s", fichero_ebmp);
		exit(0);
	} 

	/* se lee la cabecera con el tamaño de las filas y las columnas de la imagen */
	if (fread ((int *) &f,sizeof(int), 1, ebmp) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (fread (&c,sizeof(int), 1, ebmp) != 1) {
		printf("Error: no se puede leer la cabecera");
		exit(0);
	}
	if (f != c) {
		printf("Error: el número de filas no es igual al número de columnas");
		exit(0);
	}

	/* se lee la imagen */
	*M = f;
	if (!(*I = (unsigned char *) malloc(f * c * sizeof(unsigned char)))) {
		printf("Error: no se puede reservar la memoria para la imagen");
		exit(0);
	}
	
	if (fread (*I, sizeof(unsigned char), f*c, ebmp) != (f*c)) {
		printf("Error: no se puede leer la imagen");
		exit(0);
	}

	fclose(ebmp);
}


#endif /* LECTURA_DATOS_H */
