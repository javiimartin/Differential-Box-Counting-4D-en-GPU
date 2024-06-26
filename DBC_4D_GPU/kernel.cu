#include <iostream>
#include <ctime>
#include <iomanip>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "bitmap_image.hpp" // clase para leer im�genes BMP descargada de: http://www.partow.net/programming/bitmap/index.html
#include "../common/lectura_datos.h"  // para lectura de im�genes y vol�menes 3D
#include "regression_line.h"

#include <stdio.h>

// seleccionar uno de estos dos defines para ejecutar la suma mediante sumas at�micas o mediante reducci�n
#define _ATOMIC_SUM 1  
//#define _REDUCTION_SUM 1 

// seleccionar uno de estos dos defines para hacer las transferencias de memoria utilizando memoria host pinned o memoria host normal (_PAGEABLE)
#define _PINNED 1 // ESTE ES EL QUE DA MEJORES TIEMPOS
//#define _PAGEABLE 1

// N�mero m�ximo de threads por bloque es 1024 para compute capability > 2.0
#define TPB 128 // threads per block mejor para Tesla K40c: 128
//#define TPB 64 // threads per block mejor para GTX 850m: 64

// seleccionar seg�n corresponda en base al valor de TPB
#define TPB_POTENCIA_2 1
//#define TPB_NO_POTENCIA_2 1



/**
@brief calcula el m�ximo y el m�nimo utilizando sentencias if
*/
__device__ void max_min(const unsigned char a, const unsigned char b, const unsigned char c, const unsigned char d,
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
@brief Kernel CUDA que implementa el DBC 3D para el tama�o de grid m�s peque�o, de manera que Imin se inicializa con los valores obtenidos
	   de Imax, evitando la copia inicial desde Imax a Imin
@param bits_M [in] 2^bits_M = M
@param sm [in] mitad del lado del box (lado del box = s)
@param sp [in] altura del box en la dimensi�n z
@param bits_s [i] 2^bits_s = s (s: lado del box)
*/

__global__ void DBCKernel_inicial(unsigned char* Imax, unsigned char* Imin, const int M, const unsigned char bits_M, const int G, unsigned int* Nr,
	const unsigned int sm, const unsigned int sp, const unsigned char bits_s,
	const unsigned char bits_TPB)
{
	#ifdef _REDUCTION_SUM
	__shared__ unsigned int SharedData[TPB]; // stores box-counting of each thread for using it in reduction sum
											 // este TPB tiene que ser fijo y siempre es mayor o igual a vTPB
	#endif

	

	register unsigned long long int tid = threadIdx.x; // identificaci�n del thread dentro del bloque
	register unsigned long long int idx = (blockIdx.x << bits_TPB /** vTPB*/) + tid; // identificaci�n del thread global

	/* ATENCI�N!!!: esta comprobaci�n solo hace falta si TPB no es potencia de 2 */
	/* si el TPB elegido finalmente es potencia de 2, entonces quitar este if para mejorar la ejecuci�n */
#ifdef TPB_NO_POTENCIA_2
	if (idx >= ((M >> 1) << ((bits_M - 1) + (bits_M - 1) + (bits_M - 1))))
		return;
#endif
	register unsigned int l = idx >> ((bits_M - bits_s) + (bits_M - bits_s) + (bits_M - bits_s)); // l index: idx/((m/s)*(m/s)*(m/s))
	register unsigned int offsetl = (idx & (((M >> bits_s) << ((bits_M - bits_s) + (bits_M - bits_s))) - 1)); // idx mod ((m/s)*(m/s)*(m/s)): offset inside 3D matrix l

	register unsigned int k = offsetl >> ((bits_M - bits_s) + (bits_M - bits_s)); // k index: offsetl / ((m/s)*(m/s))
	register unsigned int offset = (offsetl & (((M >> bits_s) << (bits_M - bits_s)) - 1)); // (offsetl mod ((m/s)*(m/s))), offset inside k slice

	register unsigned long long int i = offset >> (bits_M - bits_s); // i index: offset / (m/s)
	register unsigned long long int j = offset & ((M >> bits_s) - 1); // j index: offset mod (m/s)


	const register unsigned int jbs = j << bits_s;
	const register unsigned int ibsbM = (i << bits_s) << bits_M;
	const register unsigned int kbsbMM = ((k << bits_s) << bits_M) << bits_M;
	const register unsigned int lbsbMMM = (((l << bits_s) << bits_M) << bits_M) << bits_M;
	
	const register unsigned int jbssm = (j << bits_s) + sm;
	const register unsigned int ibssmbM = ((i << bits_s) + sm) << bits_M;
	const register unsigned int kbssmbMM = (((k << bits_s) + sm) << bits_M) << bits_M;
	const register unsigned int lbssmbMMM = ((((l << bits_s) + sm) << bits_M) << bits_M) << bits_M;

	
	register unsigned char valmax;
	register unsigned char valmin;
	max_min( Imax[lbsbMMM + kbsbMM + ibsbM + jbs], Imax[lbsbMMM + kbsbMM + ibsbM + jbssm], 
				Imax[lbsbMMM + kbsbMM + ibssmbM + jbs], Imax[lbsbMMM + kbsbMM + ibssmbM + jbssm],
				Imax[lbsbMMM + kbssmbMM + ibsbM + jbs], Imax[lbsbMMM + kbssmbMM + ibsbM + jbssm], 
				Imax[lbsbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbsbMMM + kbssmbMM + ibssmbM + jbssm],
				Imax[lbssmbMMM + kbsbMM + ibsbM + jbs], Imax[lbssmbMMM + kbsbMM + ibsbM + jbssm], 
				Imax[lbssmbMMM + kbsbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbsbMM + ibssmbM + jbssm],
				Imax[lbssmbMMM + kbssmbMM + ibsbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibsbM + jbssm], 
				Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbssm],

				Imax[lbsbMMM + kbsbMM + ibsbM + jbs], Imax[lbsbMMM + kbsbMM + ibsbM + jbssm],
				Imax[lbsbMMM + kbsbMM + ibssmbM + jbs], Imax[lbsbMMM + kbsbMM + ibssmbM + jbssm],
				Imax[lbsbMMM + kbssmbMM + ibsbM + jbs], Imax[lbsbMMM + kbssmbMM + ibsbM + jbssm],
				Imax[lbsbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbsbMMM + kbssmbMM + ibssmbM + jbssm],
				Imax[lbssmbMMM + kbsbMM + ibsbM + jbs], Imax[lbssmbMMM + kbsbMM + ibsbM + jbssm],
				Imax[lbssmbMMM + kbsbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbsbMM + ibssmbM + jbssm],
				Imax[lbssmbMMM + kbssmbMM + ibsbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibsbM + jbssm],
				Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbssm],
				valmax, valmin);


	Imax[lbsbMMM + kbsbMM + ibsbM + jbs] = valmax;
	Imin[lbsbMMM + kbsbMM + ibsbM + jbs] = valmin;

	/* usando sumas at�micas */
#ifdef _ATOMIC_SUM
	float invsp = 1.0 / sp;
	atomicAdd(Nr, ceilf((float) /*valmax[15]*//*maxv*/valmax * invsp) - ceilf((float) /*valmin[15]*//*minv*/valmin * invsp) + 1);
#endif
	/* fin usando sumas at�micas*/

	/* usando sumas por reducci�n */
#ifdef _REDUCTION_SUM
	float invsp = 1.0 / sp;
	SharedData[tid] = ceilf((float) /*valmax[15]*/maxv * invsp) - ceilf((float) /*valmin[15]*/minv * invsp) + 1;
	__syncthreads();
	for (unsigned int ss = (vTPB >> 1); ss > 0; ss >>= 1) {
		if (tid < ss) SharedData[tid] += SharedData[tid + ss];
		__syncthreads();
	}

	if (tid == 0) atomicAdd(Nr, *SharedData); // result of reduction sum is returned
#endif
/* fin usando sumas por reducci�n */
}


/**
@brief Kernel CUDA que implementa el DBC 4D
@param bits_M [in] 2^bits_M = M
@param sm [in] mitad del lado del box (lado del box = s)
@param sp [in] altura del box en la dimensi�n z
@param bits_s [i] 2^bits_s = s (s: lado del box)
*/
__global__ void DBCKernel(unsigned char* Imax, unsigned char* Imin, const int M,
	const unsigned char bits_M, const int G, unsigned int* Nr,
	const unsigned int sm, const unsigned int sp, const unsigned char bits_s,
	const unsigned char bits_TPB)
{
#ifdef _REDUCTION_SUM
	__shared__ unsigned int SharedData[TPB]; // stores box-counting of each thread for using it in reduction sum
											 // este TPB tiene que ser fijo y siempre es mayor o igual a vTPB
#endif

	register unsigned long long int tid = threadIdx.x; // identificaci�n del thread dentro del bloque
	register unsigned long long int idx = (blockIdx.x << bits_TPB /** vTPB*/) + tid; // identificaci�n del thread global

	/* ATENCI�N!!!: esta comprobaci�n solo hace falta si TPB no es potencia de 2 */
	/* si el TPB elegido finalmente es potencia de 2, entonces quitar este if para mejorar la ejecuci�n */
#ifdef TPB_NO_POTENCIA_2
	if (idx >= ((M >> 1) << ((bits_M - 1) + (bits_M - 1) + (bits_M - 1))))
		return;
#endif
	register unsigned int l = idx >> ((bits_M - bits_s) + (bits_M - bits_s) + (bits_M - bits_s)); // l index: idx/((m/s)*(m/s)*(m/s))
	register unsigned int offsetl = (idx & (((M >> bits_s) << ((bits_M - bits_s) + (bits_M - bits_s))) - 1)); // idx mod ((m/s)*(m/s)*(m/s)): offset inside 3D matrix l

	register unsigned int k = offsetl >> ((bits_M - bits_s) + (bits_M - bits_s)); // k index: offsetl / ((m/s)*(m/s))
	register unsigned int offset = (offsetl & (((M >> bits_s) << (bits_M - bits_s)) - 1)); // (offsetl mod ((m/s)*(m/s))), offset inside k slice

	register unsigned long long int i = offset >> (bits_M - bits_s); // i index: offset / (m/s)
	register unsigned long long int j = offset & ((M >> bits_s) - 1); // j index: offset mod (m/s)


	const register unsigned int jbs = j << bits_s;
	const register unsigned int ibsbM = (i << bits_s) << bits_M;
	const register unsigned int kbsbMM = ((k << bits_s) << bits_M) << bits_M;
	const register unsigned int lbsbMMM = (((l << bits_s) << bits_M) << bits_M) << bits_M;

	const register unsigned int jbssm = (j << bits_s) + sm;
	const register unsigned int ibssmbM = ((i << bits_s) + sm) << bits_M;
	const register unsigned int kbssmbMM = (((k << bits_s) + sm) << bits_M) << bits_M;
	const register unsigned int lbssmbMMM = ((((l << bits_s) + sm) << bits_M) << bits_M) << bits_M;


	register unsigned char valmax;
	register unsigned char valmin;
	max_min(Imax[lbsbMMM + kbsbMM + ibsbM + jbs], Imax[lbsbMMM + kbsbMM + ibsbM + jbssm],
		Imax[lbsbMMM + kbsbMM + ibssmbM + jbs], Imax[lbsbMMM + kbsbMM + ibssmbM + jbssm],
		Imax[lbsbMMM + kbssmbMM + ibsbM + jbs], Imax[lbsbMMM + kbssmbMM + ibsbM + jbssm],
		Imax[lbsbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbsbMMM + kbssmbMM + ibssmbM + jbssm],
		Imax[lbssmbMMM + kbsbMM + ibsbM + jbs], Imax[lbssmbMMM + kbsbMM + ibsbM + jbssm],
		Imax[lbssmbMMM + kbsbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbsbMM + ibssmbM + jbssm],
		Imax[lbssmbMMM + kbssmbMM + ibsbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibsbM + jbssm],
		Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbs], Imax[lbssmbMMM + kbssmbMM + ibssmbM + jbssm],

		Imin[lbsbMMM + kbsbMM + ibsbM + jbs], Imin[lbsbMMM + kbsbMM + ibsbM + jbssm],
		Imin[lbsbMMM + kbsbMM + ibssmbM + jbs], Imin[lbsbMMM + kbsbMM + ibssmbM + jbssm],
		Imin[lbsbMMM + kbssmbMM + ibsbM + jbs], Imin[lbsbMMM + kbssmbMM + ibsbM + jbssm],
		Imin[lbsbMMM + kbssmbMM + ibssmbM + jbs], Imin[lbsbMMM + kbssmbMM + ibssmbM + jbssm],
		Imin[lbssmbMMM + kbsbMM + ibsbM + jbs], Imin[lbssmbMMM + kbsbMM + ibsbM + jbssm],
		Imin[lbssmbMMM + kbsbMM + ibssmbM + jbs], Imin[lbssmbMMM + kbsbMM + ibssmbM + jbssm],
		Imin[lbssmbMMM + kbssmbMM + ibsbM + jbs], Imin[lbssmbMMM + kbssmbMM + ibsbM + jbssm],
		Imin[lbssmbMMM + kbssmbMM + ibssmbM + jbs], Imin[lbssmbMMM + kbssmbMM + ibssmbM + jbssm],
		valmax, valmin);

	Imax[lbsbMMM + kbsbMM + ibsbM + jbs] = valmax;
	Imin[lbsbMMM + kbsbMM + ibsbM + jbs] = valmin;

	/* usando sumas at�micas */
#ifdef _ATOMIC_SUM
	float invsp = 1.0 / sp;
	atomicAdd(Nr, ceilf((float) /*valmax[15]*//*maxv*/valmax * invsp) - ceilf((float) /*valmin[15]*//*minv*/valmin * invsp) + 1);
#endif
	/* fin usando sumas at�micas*/

	/* usando sumas por reducci�n */
#ifdef _REDUCTION_SUM
	float invsp = 1.0 / sp;
	SharedData[tid] = ceilf((float) /*valmax[15]*/maxv * invsp) - ceilf((float) /*valmin[15]*/minv * invsp) + 1;
	__syncthreads();
	for (unsigned int ss = (vTPB >> 1); ss > 0; ss >>= 1) {
		if (tid < ss) SharedData[tid] += SharedData[tid + ss];
		__syncthreads();
	}

	if (tid == 0) atomicAdd(Nr, *SharedData); // result of reduction sum is returned
#endif
/* fin usando sumas por reducci�n */
}

/**
@brief Helper function for using CUDA to compute DBC algorithm
*/

cudaError_t DBCWithCuda(const unsigned char* Imax, const int Numr,
	const int M, const unsigned char bits_M, const int G, unsigned int* Nr)
{
	unsigned char* dev_Imax = 0;
	unsigned char* dev_Imin = 0;
	unsigned int* dev_Nr = 0;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float time_transfers, total_time_kernels;
	float time_kernel;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_Imax, M * M * M * M * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Imin, M * M * M * M * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_Nr, Numr * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	total_time_kernels = 0;
	time_transfers = 0;
	for (int rep = 0; rep < 10; rep++) { // repeticiones para obtener tiempos mayores para comparar

		// Launch the kernel on the GPU
		unsigned int num_box;
		unsigned int tam_grid;
		dim3 grid, block(TPB, 1, 1);

		unsigned int sp;
		unsigned int s = 2;
		unsigned int sm;
		unsigned int size = M;
		unsigned char Nri = 0;
		int tpb;
		unsigned char b_tpb = log(TPB) / log(2);

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		// Imax se copia de host a device. Imin no hace falta copiarlo ya que se genera a partir de Imax en la primera llamada al kernel para el grid m�s peque�o
		cudaStatus = cudaMemcpy(dev_Imax, Imax, M * M * M * M * sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaStatus = cudaMemset(dev_Nr, 0, Numr * sizeof(unsigned int));
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_kernel, start, stop);
		time_transfers += time_kernel;

		// primera llamada se hace especial para generar Imin a partir de los valores iniciales de Imax y asi ahorrar la copia
		// inicial desde Imax a Imin
		sm = s >> 1; // la mitad de s
		sp = ceilf(((float)(G << (Nri + 1))) / (float)M); // tamaño del voxel en la direccion z
		//tam_grid = ceilf(((M * M * M * M) >> 4) / (float)TPB); // M/2 * M/2 * M/2 * M/2 = (tam_grid * TPB)
		grid.x = ceilf(((M * M * M * M) / (s * s * s * s)) / (float)TPB);
		grid.y = 1; grid.z = 1;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		DBCKernel_inicial<<<grid, block>>>(dev_Imax, dev_Imin, M, bits_M, G, &dev_Nr[Nri], sm, sp, Nri + 1, b_tpb);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_kernel, start, stop);
		total_time_kernels += time_kernel;
		cudaStatus = cudaGetLastError();
		

		Nri++;
		s <<= 1;
		size >>= 1;

		while (size > 2) {
			sm = s >> 1; // la mitad de s
			sp = ceilf(((float)(G << (Nri + 1))) / (float)M); // tama�o del voxel en la direcci�n z
			num_box = (M * M * M * M) / (s * s * s * s);
			if (num_box >= TPB) {
				grid.x = ceilf(num_box / (float)TPB); // M/s * M/s * M/s * M/s= (tam_grid * TPB)
				grid.y = 1; grid.z = 1;
				//tpb = TPB;
			}
			else {
				grid.x = 1; grid.y = 1; grid.z = 1;
				block.x = num_box; block.y = 1; block.z = 1;
				//tpb = num_box;
				b_tpb = log(num_box) / log(2);
			}

			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
			DBCKernel<<<grid, block>>>(dev_Imax, dev_Imin, M, bits_M, G, &dev_Nr[Nri], sm, sp, Nri + 1, b_tpb);
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time_kernel, start, stop);
			total_time_kernels += time_kernel;

			Nri++;
			s <<= 1;
			size >>= 1;
		}

		cudaDeviceSynchronize();

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		cudaStatus = cudaMemcpy(Nr, dev_Nr, Numr * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_kernel, start, stop);
		time_transfers += time_kernel;

	} // fin de repeticiones

	printf("Tiempo kernels:  %3.1f ms \n", total_time_kernels);
	printf("Tiempo transferencias:  %3.1f ms \n", time_transfers);

Error:
	cudaFree(dev_Imax);
	cudaFree(dev_Imin);
	cudaFree(dev_Nr);

	return cudaStatus;
}



void generar_volumen4D(unsigned char** I, int M) {
	int total_elements = M * M * M * M;
	*I = (unsigned char*)malloc(total_elements * sizeof(unsigned char));
	for (int i = 0; i < total_elements; ++i) {
		(*I)[i] = rand() % 256; // Números aleatorios entre 0 y 255
		//(*I)[i] = 255;
		//(*I)[i] = i % 255;
	}
}

/**
@brief Funci�n main
*/
int main()
{	
	// Array de nombres de ficheros de imagen
	char* filenames[] = {
		"..\\imagenes\\image4D_8.vox", "..\\imagenes\\image4D_16.vox",
		"..\\imagenes\\image4D_32.vox", "..\\imagenes\\image4D_64.vox",
		"..\\imagenes\\image4D_128.vox"
	};
	char* filenames_linux[] = {
		"../imagenes/image4D_8.vox", "../imagenes/image4D_16.vox",
		"../imagenes/image4D_32.vox", "../imagenes/image4D_64.vox",
		"../imagenes/image4D_128.vox"
	};
	int numFiles = 5;


	// leer volumen VOX 

	cudaError_t cudaStatus;
	
	int G = 256; // n�mero de niveles de gris en el volumen

	int Numr;
	unsigned int* Nr;

	int s[6] = { 2, 4, 8, 16, 32, 64 };

	for (int fileIndex = 0; fileIndex < numFiles; ++fileIndex) {
		unsigned char* I = NULL;
		int M;

		std::cout << "Imagen  " << filenames[fileIndex] << std::endl;

#ifdef _WIN32
		leer_volumen4D_vox(filenames[fileIndex], &I, &M);
#elif __linux__
		leer_volumen4D_vox(filenames_linux[fileIndex], &I, &M);
#endif

		Numr = log(M) / log(2) - 1;
		const unsigned char bits_M = Numr + 1; // 2^bits_M = M

		// copia la imagen I en los dos buffers que contienen el maximo y el minimo para cada tama�o s 
#ifdef _PINNED
		unsigned char* Imax;
		cudaStatus = cudaMallocHost((void**)&Imax, sizeof(unsigned char) * M * M * M * M);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n",
				cudaGetErrorString(cudaStatus));
			return 1;
		}
		cudaStatus = cudaMallocHost((void**)&Nr, sizeof(unsigned int) * Numr);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n",
				cudaGetErrorString(cudaStatus));
			return 1;
	}
#elif _PAGEABLE
		unsigned char* Imax = new unsigned char[M * M * M * M];
		Nr = new unsigned int[Numr];
#endif

		memcpy(Imax, I, sizeof(unsigned char) * M * M * M * M);
		for (int i = 0;i < Numr; i++) Nr[i] = 0;

		//std::cout << "Ejecutando DBC 4D CUDA" << std::endl;
		cudaStatus = DBCWithCuda(Imax, Numr, M, bits_M, G, Nr);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "DBCWithCuda failed!");
			return 1;
		}

		// visualizamos los resultados 
		for (int i = 0; i < Numr; i++) {
			std::cout << "s: " << (2 << i) << " -- Nr: " << Nr[i] << std::endl;
		}


		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

#ifdef _PINNED
		cudaFreeHost(Imax);
		cudaFreeHost(Nr);
#elif _PAGEABLE
		delete Imax;
		delete Nr;
#endif

		free(I); // libera el volumen
		
	}

	return 0;


}
