#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "../common/lectura_datos.h"  // para lectura de im�genes y vol�menes 3D
#include "regression_line.h"

#ifdef _WIN32
#include "../common/common.h"  // para toma de tiempos en CPU
#elif __linux__
#include <sys/time.h>
#endif


/**
@brief calcula el maximo y minimo de los valores a hasta h utilizando sentencias if
*/
void max_min_if(const unsigned char a, const unsigned char b, const unsigned char c, const unsigned char d,
                  const unsigned char e, const unsigned char f, const unsigned char g, const unsigned char h,
                  const unsigned char i, const unsigned char j, const unsigned char k, const unsigned char l,
                  const unsigned char m, const unsigned char n, const unsigned char o, const unsigned char p,
                  const unsigned char a2, const unsigned char b2, const unsigned char c2, const unsigned char d2,
                  const unsigned char e2, const unsigned char f2, const unsigned char g2, const unsigned char h2,
                  const unsigned char i2, const unsigned char j2, const unsigned char k2, const unsigned char l2,
                  const unsigned char m2, const unsigned char n2, const unsigned char o2, const unsigned char p2,
                  unsigned char& maxv, unsigned char& minv) {
    if ((a >= b) && (a >= c) && (a >= d) && (a >= e) && (a >= f) && (a >= g) && (a >= h) && 
        (a >= i) && (a >= j) && (a >= k) && (a >= l) && (a >= m) && (a >= n) && (a >= o) && (a >= p)) maxv = a;
    else if ((b >= a) && (b >= c) && (b >= d) && (b >= e) && (b >= f) && (b >= g) && (b >= h) &&
            (b >= i) && (b >= j) && (b >= k) && (b >= l) && (b >= m) && (b >= n) && (b >= o) && (b >= p)) maxv = b;
    else if ((c >= a) && (c >= b) && (c >= d) && (c >= e) && (c >= f) && (c >= g) && (c >= h) &&
            (c >= i) && (c >= j) && (c >= k) && (c >= l) && (c >= m) && (c >= n) && (c >= o) && (c >= p)) maxv = c;
    else if ((d >= a) && (d >= b) && (d >= c) && (d >= e) && (d >= f) && (d >= g) && (d >= h) &&
            (d >= i) && (d >= j) && (d >= k) && (d >= l) && (d >= m) && (d >= n) && (d >= o) && (d >= p)) maxv = d;
    else if ((e >= a) && (e >= b) && (e >= c) && (e >= d) && (e >= f) && (e >= g) && (e >= h) &&
            (e >= i) && (e >= j) && (e >= k) && (e >= l) && (e >= m) && (e >= n) && (e >= o) && (e >= p)) maxv = e;
    else if ((f >= a) && (f >= b) && (f >= c) && (f >= d) && (f >= e) && (f >= g) && (f >= h) &&
            (f >= i) && (f >= j) && (f >= k) && (f >= l) && (f >= m) && (f >= n) && (f >= o) && (f >= p)) maxv = f;
    else if ((g >= a) && (g >= b) && (g >= c) && (g >= d) && (g >= e) && (g >= f) && (g >= h) &&
            (g >= i) && (g >= j) && (g >= k) && (g >= l) && (g >= m) && (g >= n) && (g >= o) && (g >= p)) maxv = g;
    else if ((h >= a) && (h >= b) && (h >= c) && (h >= d) && (h >= e) && (h >= f) && (h >= g) &&
            (h >= i) && (h >= j) && (h >= k) && (h >= l) && (h >= m) && (h >= n) && (h >= o) && (h >= p)) maxv = h;
    else if ((i >= a) && (i >= b) && (i >= c) && (i >= d) && (i >= e) && (i >= f) && (i >= g) &&
            (i >= h) && (i >= j) && (i >= k) && (i >= l) && (i >= m) && (i >= n) && (i >= o) && (i >= p)) maxv = i;
    else if ((j >= a) && (j >= b) && (j >= c) && (j >= d) && (j >= e) && (j >= f) && (j >= g) &&
            (j >= h) && (j >= i) && (j >= k) && (j >= l) && (j >= m) && (j >= n) && (j >= o) && (j >= p)) maxv = j;
    else if ((k >= a) && (k >= b) && (k >= c) && (k >= d) && (k >= e) && (k >= f) && (k >= g) &&
            (k >= h) && (k >= i) && (k >= j) && (k >= l) && (k >= m) && (k >= n) && (k >= o) && (k >= p)) maxv = k;
    else if ((l >= a) && (l >= b) && (l >= c) && (l >= d) && (l >= e) && (l >= f) && (l >= g) &&
            (l >= h) && (l >= i) && (l >= j) && (l >= k) && (l >= m) && (l >= n) && (l >= o) && (l >= p)) maxv = l;
    else if ((m >= a) && (m >= b) && (m >= c) && (m >= d) && (m >= e) && (m >= f) && (m >= g) &&
            (m >= h) && (m >= i) && (m >= j) && (m >= k) && (m >= l) && (m >= n) && (m >= o) && (m >= p)) maxv = m;
    else if ((n >= a) && (n >= b) && (n >= c) && (n >= d) && (n >= e) && (n >= f) && (n >= g) &&
            (n >= h) && (n >= i) && (n >= j) && (n >= k) && (n >= l) && (n >= m) && (n >= o) && (n >= p)) maxv = n;
    else if ((o >= a) && (o >= b) && (o >= c) && (o >= d) && (o >= e) && (o >= f) && (o >= g) &&
            (o >= h) && (o >= i) && (o >= j) && (o >= k) && (o >= l) && (o >= m) && (o >= n) && (o >= p)) maxv = o;
    else maxv = p;

    if ((a2 <= b2) && (a2 <= c2) && (a2 <= d2) && (a2 <= e2) && (a2 <= f2) && (a2 <= g2) && (a2 <= h2) &&
        (a2 <= i2) && (a2 <= j2) && (a2 <= k2) && (a2 <= l2) && (a2 <= m2) && (a2 <= n2) && (a2 <= o2) && (a2 <= p2)) minv = a2;
    else if ((b2 <= a2) && (b2 <= c2) && (b2 <= d2) && (b2 <= e2) && (b2 <= f2) && (b2 <= g2) && (b2 <= h2) &&
            (b2 <= i2) && (b2 <= j2) && (b2 <= k2) && (b2 <= l2) && (b2 <= m2) && (b2 <= n2) && (b2 <= o2) && (b2 <= p2)) minv = b2;
    else if ((c2 <= a2) && (c2 <= b2) && (c2 <= d2) && (c2 <= e2) && (c2 <= f2) && (c2 <= g2) && (c2 <= h2) &&
            (c2 <= i2) && (c2 <= j2) && (c2 <= k2) && (c2 <= l2) && (c2 <= m2) && (c2 <= n2) && (c2 <= o2) && (c2 <= p2)) minv = c2;
    else if ((d2 <= a2) && (d2 <= b2) && (d2 <= c2) && (d2 <= e2) && (d2 <= f2) && (d2 <= g2) && (d2 <= h2) &&
            (d2 <= i2) && (d2 <= j2) && (d2 <= k2) && (d2 <= l2) && (d2 <= m2) && (d2 <= n2) && (d2 <= o2) && (d2 <= p2)) minv = d2;
    else if ((e2 <= a2) && (e2 <= b2) && (e2 <= c2) && (e2 <= d2) && (e2 <= f2) && (e2 <= g2) && (e2 <= h2) &&
            (e2 <= i2) && (e2 <= j2) && (e2 <= k2) && (e2 <= l2) && (e2 <= m2) && (e2 <= n2) && (e2 <= o2) && (e2 <= p2)) minv = e2;
    else if ((f2 <= a2) && (f2 <= b2) && (f2 <= c2) && (f2 <= d2) && (f2 <= e2) && (f2 <= g2) && (f2 <= h2) &&
            (f2 <= i2) && (f2 <= j2) && (f2 <= k2) && (f2 <= l2) && (f2 <= m2) && (f2 <= n2) && (f2 <= o2) && (f2 <= p2)) minv = f2;
    else if ((g2 <= a2) && (g2 <= b2) && (g2 <= c2) && (g2 <= d2) && (g2 <= e2) && (g2 <= f2) && (g2 <= h2) &&
            (g2 <= i2) && (g2 <= j2) && (g2 <= k2) && (g2 <= l2) && (g2 <= m2) && (g2 <= n2) && (g2 <= o2) && (g2 <= p2)) minv = g2;
    else if ((h2 <= a2) && (h2 <= b2) && (h2 <= c2) && (h2 <= d2) && (h2 <= e2) && (h2 <= f2) && (h2 <= g2) &&
            (h2 <= i2) && (h2 <= j2) && (h2 <= k2) && (h2 <= l2) && (h2 <= m2) && (h2 <= n2) && (h2 <= o2) && (h2 <= p2)) minv = h2;
    else if ((i2 <= a2) && (i2 <= b2) && (i2 <= c2) && (i2 <= d2) && (i2 <= e2) && (i2 <= f2) && (i2 <= g2) &&
            (i2 <= h2) && (i2 <= j2) && (i2 <= k2) && (i2 <= l2) && (i2 <= m2) && (i2 <= n2) && (i2 <= o2) && (i2 <= p2)) minv = i2;
    else if ((j2 <= a2) && (j2 <= b2) && (j2 <= c2) && (j2 <= d2) && (j2 <= e2) && (j2 <= f2) && (j2 <= g2) &&
            (j2 <= h2) && (j2 <= i2) && (j2 <= k2) && (j2 <= l2) && (j2 <= m2) && (j2 <= n2) && (j2 <= o2) && (j2 <= p2)) minv = j2;
    else if ((k2 <= a2) && (k2 <= b2) && (k2 <= c2) && (k2 <= d2) && (k2 <= e2) && (k2 <= f2) && (k2 <= g2) &&
            (k2 <= h2) && (k2 <= i2) && (k2 <= j2) && (k2 <= l2) && (k2 <= m2) && (k2 <= n2) && (k2 <= o2) && (k2 <= p2)) minv = k2;
    else if ((l2 <= a2) && (l2 <= b2) && (l2 <= c2) && (l2 <= d2) && (l2 <= e2) && (l2 <= f2) && (l2 <= g2) &&
            (l2 <= h2) && (l2 <= i2) && (l2 <= j2) && (l2 <= k2) && (l2 <= m2) && (l2 <= n2) && (l2 <= o2) && (l2 <= p2)) minv = l2;
    else if ((m2 <= a2) && (m2 <= b2) && (m2 <= c2) && (m2 <= d2) && (m2 <= e2) && (m2 <= f2) && (m2 <= g2) &&
            (m2 <= h2) && (m2 <= i2) && (m2 <= j2) && (m2 <= k2) && (m2 <= l2) && (m2 <= n2) && (m2 <= o2) && (m2 <= p2)) minv = m2;
    else if ((n2 <= a2) && (n2 <= b2) && (n2 <= c2) && (n2 <= d2) && (n2 <= e2) && (n2 <= f2) && (n2 <= g2) &&
            (n2 <= h2) && (n2 <= i2) && (n2 <= j2) && (n2 <= k2) && (n2 <= l2) && (n2 <= m2) && (n2 <= o2) && (n2 <= p2)) minv = n2;
    else if ((o2 <= a2) && (o2 <= b2) && (o2 <= c2) && (o2 <= d2) && (o2 <= e2) && (o2 <= f2) && (o2 <= g2) &&
            (o2 <= h2) && (o2 <= i2) && (o2 <= j2) && (o2 <= k2) && (o2 <= l2) && (o2 <= m2) && (o2 <= n2) && (o2 <= p2)) minv = o2;
    else minv = p2;

}



/**
@brief Implementa el algoritmo de DBC 4D secuencial tal y como se explica en Biswas 98 (Algorithm 1) pero ampliado a 4D
@param I [in] El volumen 3D de tama�o MxMxM y con valores entre 0 y G a la que se le va a calcular el DBC
@param M [in] El tama�o del volumen I (MxMxM)
@param G [in] El n�mero total de niveles de gris en el volumen I
@param Nr [out] Array con el DBC para cada tama�o de grid s
*/
void DBC_4D_CPU(const unsigned char* I, const int M, const int G, unsigned long* Nr) {
    /* copia el volumen I en los dos buffers que contienen el m�ximo y el m�nimo para cada tama�o s */
    unsigned char* Imax;
    unsigned char* Imin;

    try {
        Imax = new unsigned char[M * M * M * M];
        Imin = new unsigned char[M * M * M * M];
    }
    catch (std::bad_alloc& e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    unsigned char maxv, minv;

    double tiempo_total = 0;
    struct timeval start, stop;

    for (int rep = 0; rep < 10; rep++) { // repeticiones para obtener tiempos mayores para comparar
        memcpy(Imax, I, sizeof(unsigned char) * M * M * M * M);
        memcpy(Imin, I, sizeof(unsigned char) * M * M * M * M);

        gettimeofday(&start, 0);

        unsigned int s = 2;
        unsigned int size = M;
        unsigned char Nri = 0; // �ndice sobre el array Nr

        while (size > 2) {
            int sp = ceilf(((float)(G << (Nri + 1))) / (float)M); // tamaño del voxel en la direcci�n z
            float invsp = 1.0 / sp;
            int sm = s >> 1; // la mitad de s
            unsigned long sum = 0;
            unsigned long iM, kMM, lMMM;
            unsigned long ismM, ksmMM, lsmMMM;

            Nr[Nri] = 0;
            for (unsigned long l = 0; l < (M - 1); l += s) {
                lMMM = l * M * M * M;
                lsmMMM = (l + sm) * M * M * M;
                for (unsigned long k = 0; k < (M - 1); k += s) {
                    kMM = k * M * M;
                    ksmMM = (k + sm) * M * M;
                    for (unsigned long i = 0; i < (M - 1); i += s) {
                        iM = i * M;
                        ismM = (i + sm) * M;
                        for (unsigned long j = 0; j < (M - 1); j += s) {
                            max_min_if(Imax[lMMM + kMM + iM + j], Imax[lMMM + kMM + iM + (j+sm)], Imax[lMMM + kMM + ismM + j], Imax[lMMM + kMM + ismM + (j+sm)],
                                       Imax[lMMM + ksmMM + iM + j], Imax[lMMM + ksmMM + iM + (j+sm)], Imax[lMMM + ksmMM + ismM + j], Imax[lMMM + ksmMM + ismM + (j+sm)],
                                       Imax[lsmMMM + kMM + iM + j], Imax[lsmMMM + kMM + iM + (j+sm)], Imax[lsmMMM + kMM + ismM + j], Imax[lsmMMM + kMM + ismM + (j+sm)],
                                       Imax[lsmMMM + ksmMM + iM + j], Imax[lsmMMM + ksmMM + iM + (j+sm)], Imax[lsmMMM + ksmMM + ismM + j], Imax[lsmMMM + ksmMM + ismM + (j+sm)],
                                       Imin[lMMM + kMM + iM + j], Imin[lMMM + kMM + iM + (j + sm)], Imin[lMMM + kMM + ismM + j], Imin[lMMM + kMM + ismM + (j + sm)],
                                       Imin[lMMM + ksmMM + iM + j], Imin[lMMM + ksmMM + iM + (j + sm)], Imin[lMMM + ksmMM + ismM + j], Imin[lMMM + ksmMM + ismM + (j + sm)],
                                       Imin[lsmMMM + kMM + iM + j], Imin[lsmMMM + kMM + iM + (j + sm)], Imin[lsmMMM + kMM + ismM + j], Imin[lsmMMM + kMM + ismM + (j + sm)],
                                       Imin[lsmMMM + ksmMM + iM + j], Imin[lsmMMM + ksmMM + iM + (j + sm)], Imin[lsmMMM + ksmMM + ismM + j], Imin[lsmMMM + ksmMM + ismM + (j + sm)],
                                       maxv, minv);


                            Imax[kMM + iM + j] = maxv;
                            Imin[kMM + iM + j] = minv;

                            sum += ceilf((float)maxv * invsp) - ceilf((float)minv * invsp) + 1;
                        }
                    }
                }
            }

            Nr[Nri] = sum;
            s <<= 1; // s *= 2;
            size >>= 1; // size /= 2;
            Nri++;
        }

        gettimeofday(&stop, 0);
        tiempo_total += (1000000.0 * (stop.tv_sec - start.tv_sec) + stop.tv_usec - start.tv_usec) / 1000.0;

    } // fin de repeticiones
    printf("Tiempo CPU:  %6.2f ms \n", tiempo_total);

    delete Imax;
    delete Imin;
}





int main(int argc, char* argv[])
{
    // Array de nombres de ficheros de imagen
    const char* filenames[] = {
        "..\\imagenes\\image4D_8.vox", "..\\imagenes\\image4D_16.vox",
        "..\\imagenes\\image4D_32.vox", "..\\imagenes\\image4D_64.vox",
        "..\\imagenes\\image4D_128.vox"
    };
    const char* filenames_linux[] = {
        "../imagenes/image4D_8.vox", "../imagenes/image4D_16.vox",
        "../imagenes/image4D_32.vox", "../imagenes/image4D_64.vox",
        "../imagenes/image4D_128.vox"
    };
    int numFiles = 5;

    /* leer volumen VOX */
    unsigned char* I = NULL;
    int M;
    int G = 256;

    int Numr;
    unsigned long* Nr;

    int s[6] = { 2, 4, 8, 16, 32, 64 };

    for (int fileIndex = 0; fileIndex < numFiles; ++fileIndex) {

        std::cout << "Imagen  " << filenames[fileIndex] << std::endl;

#ifdef _WIN32
        leer_volumen4D_vox((char*)filenames[fileIndex], &I, &M);
#elif __linux__
        leer_volumen4D_vox((char*)filenames_linux[fileIndex], &I, &M);
#endif

        Numr = log(M) / log(2) - 1;
        Nr = new unsigned long[Numr];
        for (int i = 0;i < Numr; i++) Nr[i] = 0;

        std::cout << "Ejecutando DBC 4D CPU" << std::endl;
        DBC_4D_CPU(I, M, G, Nr);

       

        // visualizamos los resultados 
        for (int i = 0; i < Numr; i++) {
            std::cout << "s: " << (2 << i) << " -- Nr: " << Nr[i] << std::endl;
        }


        delete Nr;
        free(I); // libera el volumen, HACER S�LO SI SE HA LE�DO DESDE UN FICHERO DE VOX
    }

    return 0;

    
    
}
