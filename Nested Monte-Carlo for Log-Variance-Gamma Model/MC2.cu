/**************************************************************
Inspired by

Lokman A. Abbas-Turki code


***************************************************************/

#include <stdio.h>
#include <omp.h>
#include <curand_kernel.h>

__device__ float kappad[20];
__device__ float sigmad[20];
__device__ float thetad[20];
__device__ float Strd[16];


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// for each maturity value, a list of strike is defined
void strikeInterval(float* K, float T) {

		float fidx = T * 12.0f + 1.0f;
		int i = 0;
		float coef = 1.0f;
		float delta;

		while (i < fidx) {
			coef *= (1.02f);
			i++;
		}

		delta = pow(coef, 1.0f / 8.0f);
		K[15] = coef;

		for (i = 1; i < 16; i++) {
			K[15 - i] = K[15 - i + 1] / delta;
		}
	}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

// Gamma variable sampling when the parameter a is smaller than 1
__device__ float gamma_johnk(float a, curandState *state) {
    float u, v, x, y;
    do {
        u = curand_uniform(state);
        v = curand_uniform(state);
        x = powf(u, 1.0f / a);
        y = powf(v, 1.0f / (1.0f - a));
    } while (x + y > 1.0f);
    return -logf(curand_uniform(state)) * x / (x + y);
}
// Gamma variable sampling when the parameter a is greater than 1
__device__ float gamma_best(float a, curandState *state) {
    float b = a - 1.0f;
    float c = 3.0f * a - 0.75f;
    float u, v, w, y, x, z;
    while (true) {
        u = curand_uniform(state);
        v = curand_uniform(state);
        w = u * (1.0f - u);
        y = sqrtf(c / w) * (u - 0.5f);
        x = b + y;
        if (x <= 0.0f) continue;
        z = 64.0f * w * w * w * v * v;
        if (z <= (1.0f - 2.0f * y * y / x)) return x;
        if (logf(z) <= 2.0f * (b * logf(x / b) - y)) return x;
    }
}

// Monte Carlo simulation kernel
__global__ void MC_k(float dt, float T, int Ntraj, curandState* state, float* sum, int* num){

	int pidx, same, numR;
	float t, logY, Y;
	float Z;
	float deltaX;
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	curandState localState = state[idx];
	float price;
	float sumR = 0.0f; 
	float sum2R = 0.0f;
	same = idx;
	float deltaS;	

	
	pidx = same/(20 * 20 * 20);
	float StrR = Strd[pidx];
	same -= (pidx* 20 * 20 * 20);
	pidx = same/(20 * 20);
	float kappaR = kappad[pidx];
	same -= (pidx*20 * 20);
	pidx = same/(20);
	float sigmaR = sigmad[pidx];
	same -= (pidx*20);
	pidx = same;
	float thetaR = thetad[pidx];
	float a = dt / kappaR;
	float w = logf((1.0f - thetaR * kappaR - kappaR * sigmaR * sigmaR / 2.0f))/ kappaR;

	numR = 0;
	for (int i = 0; i < Ntraj; i++) {
		t = 0.0f;
		logY = 0.0f;

		while(t<T){
			if (a < 1.0f)
				deltaS = gamma_johnk(a, &localState);
			else
				deltaS = gamma_best(a, &localState);

			Z = curand_normal(&localState);
			deltaX = sigmaR * Z * sqrtf(kappaR * deltaS) + thetaR * kappaR * deltaS;
			logY += deltaX;
			t += dt;
		}
		Y = expf(w * T + logY);
		price = fmaxf(0.0f, Y  - StrR) / Ntraj;
		sumR += price;
		sum2R += price * price * Ntraj;
		numR++;
		
	}
	sum[2*idx] = sumR*((float)Ntraj/numR);
	sum[2*idx + 1] = sum2R*((float)Ntraj / numR);
	num[idx] = numR;
	/* Copy state back to global memory */
	state[idx] = localState;
}

int main(void) {

	float kappa[20] = {0.035f, 0.065f, 0.085f, 0.105f, 0.135f, 0.165f, 0.205f, 0.255f, 0.305f, 0.355f, 0.405f, 0.455f, 0.505f, 0.555f, 0.605f, 0.655f, 0.705f, 0.755f, 0.805f, 0.855f};
	float sigma[20] = {0.045, 0.075, 0.105, 0.115, 0.125, 0.135, 0.145, 0.155, 0.165, 0.175, 0.185, 0.195, 0.205, 0.215, 0.225, 0.245, 0.265, 0.285, 0.305, 0.325}; 
	float theta[20] = {-0.355, -0.305, -0.255, -0.205, -0.155, -0.105, -0.055, -0.005, 0.025, 0.055, 0.065, 0.075, 0.085, 0.095, 0.105, 0.115, 0.125, 0.14, 0.155, 0.175};
	float Tmt[16] = { 1.0f / 12.0f,  2.0f / 12.0f, 3.0f / 12.0f, 4.0f / 12.0f, 5.0f / 12.0f, 6.0f / 12.0f, 7.0f / 12.0f,
					  8.0f / 12.0f, 9.0f / 12.0f, 10.0f / 12.0f, 11.0f / 12.0f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f };
	float Str[16];
	

	cudaMemcpyToSymbol(kappad, kappa, 20*sizeof(float));
	cudaMemcpyToSymbol(sigmad, sigma, 20*sizeof(float));
	cudaMemcpyToSymbol(thetad, theta, 20*sizeof(float));
	

	int pidx, same;
	int NTPB = 512; 
	int NB =  250; 
	int Ntraj = 64 * 512; 
	float dt = 1.0f/(2000.0f);
	float StrR, sigmaR, kappaR, price, error;

	curandState* states;
	cudaMalloc(&states, NB*NTPB*sizeof(curandState));
	init_curand_state_k <<<NB, NTPB>>> (states);
	float *sum;
	int* num;
	cudaMallocManaged(&sum, 2*NB*NTPB*sizeof(float));
	cudaMallocManaged(&num, NB * NTPB * sizeof(int));

	int numTraj;
	FILE* fpt;

	char strg[30];
	for(int i=0; i<16; i++){
		strikeInterval(Str, Tmt[i]);
		cudaMemcpyToSymbol(Strd, Str, 16*sizeof(float));
		MC_k<<<NB,NTPB>>>(dt, Tmt[i], Ntraj, states, sum, num);
		cudaDeviceSynchronize();
		for(int j=0; j<16; j++){
			StrR = Str[j];
			sprintf(strg, "Tmt%.4fStr%.4f.csv", Tmt[i], StrR);
			fpt = fopen(strg, "w+");
			fprintf(fpt, "kappa,sigma, theta, price, 95cI, numTraj\n");
			for(int k=0; k< 20*20*20; k++){
				same = k + j*(20*20*20);
				numTraj = num[same];
				pidx = j;
				price = sum[2*same];
				error = 1.96f*sqrtf(sum[2*same+1] - (price * price)) / sqrtf((float)Ntraj);
				same -= (pidx* 20*20*20);
				pidx = same/(20*20);
				kappaR = kappa[pidx];
				same -= (pidx*20*20);
				pidx = same/(20);
				sigmaR = sigma[pidx];
				same -= (pidx*20);
				pidx = same;
				fprintf(fpt, "%f, %f, %f, %f, %f, %d\n", kappaR, sigmaR, theta[same], price, error, numTraj);
			}
			fclose(fpt);
		}
	}

	cudaFree(states);
	cudaFree(sum);
	cudaFree(num);

	return 0;
}