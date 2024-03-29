#include <stdio.h>      // printf
#include <iostream>     // stoi
#include <cstdlib>      // random

#include <cublas_v2.h>
#include <ctime>        // time measurement, time()
#include <cmath>        // fabs()

void random_initialize(float* x, int size) {
    /*
        - Randomly initialize vector x
        - Range of value is [-1, 1]
        - Note that it is "column-major" order
    */
    for (int i = 0; i < size; i++) {
        x[i] = (float)rand()/(float)RAND_MAX - 0.5;
    }
}

void reset_vector(float* x, int size) {
    /*
        - Randomly initialize vector x
        - Range of value is [-1, 1]
        - Note that it is "column-major" order
    */
    for (int i = 0; i < size; i++) {
        x[i] = 0;
    }
}

void print_vector(float* x, int row, int col) {
    /*
       Note that matrix is stored in "column-major" order
    */
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%.2f ", x[j*row + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void matmul_cpu(float* A, float* B, float* C, int M, int K, int N) {
    /*
       Note that matrix is stored in "column-major" order
    */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += A[k*M + i] * B[j*K + k];
            }
            C[j*M + i] = psum;
        }
    }
}

void cublas_gemm(float* A, float* B, float* C, int M, int K, int N) {
    // initialize cuBLAS library
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Ref: https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm
    // sgemm = single-precision gemm
    // C = alpha * A * B + beta * C
    const float alpha_val = 1.0;
    const float* alpha = &alpha_val;
    const float beta_val = 0.0;
    const float* beta = &beta_val;
    // Leading dimension; row size in 2D array
    int lda = M;
    int ldb = K;
    int ldc = M;

    status = cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc
    );

    if (status != CUBLAS_STATUS_SUCCESS){
        printf("cublas Sgemm failed...\n");
        exit(1);
    }

    // Destroy cublas
    cublasDestroy(handle);
}

void compare_matrix(float* mat, float* ref, int M, int N) {
    bool correct = true;
    float tmp_1, tmp_2;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            tmp_1 = mat[j*M + i];
            tmp_2 = ref[j*M + i];
            if (fabs(tmp_1 - tmp_2) > 0.0001) {
                printf("Error at (%d, %d). Result is %.3f, and ref is %.3f\n", i, j, tmp_1, tmp_2);
                correct = false;
                break;
            }
        }
        if (!correct) {
            break;
        }
    }
    if (correct) {
        printf("\t All correct\n");
    }

}

int main(int argc, char** argv) {
    // Configuration
    // A: (M, K), B: (K, N), C: (M, N)
    int M, N, K;
    int iter;
    if (argc >= 4) {
            M = std::stoi(argv[1]);
            K = std::stoi(argv[2]);
            N = std::stoi(argv[3]);
            if (argc == 5) {
                iter = std::stoi(argv[4]);
            }
            else {
                iter = 1;
            }
    }
    else {
        M = 4;
        K = 16;
        N = 8;
        iter = 1;
    }
    
    unsigned int mem_size_A = sizeof(float) * M * K;
    unsigned int mem_size_B = sizeof(float) * K * N;
    unsigned int mem_size_C = sizeof(float) * M * N;
    
    /*
       Initialize input operands in host(=CPU)
    */
    float* A_h = (float*) malloc (mem_size_A);
    float* B_h = (float*) malloc (mem_size_B);
    float* C_h = (float*) malloc (mem_size_C);
    float* C_ref = (float*) malloc (mem_size_C);    // reference of the result
    
    srand(time(NULL));
    random_initialize(A_h, M*K);
    random_initialize(B_h, K*N);
    //print_vector(A_h, M, K);
    //print_vector(B_h, K, N);
    
    clock_t start_cpu, end_cpu;
    float latency_cpu;
    
    for (int i = 0; i < 3; i++) {
        reset_vector(C_ref, M*N);
        matmul_cpu(A_h, B_h, C_ref, M, K, N);
    }

    reset_vector(C_ref, M*N);

    start_cpu = clock();
    matmul_cpu(A_h, B_h, C_ref, M, K, N);
    end_cpu = clock();

    latency_cpu = 1000*(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("***********************************\n");
    printf("CPU naive gemm\n");
    printf("***********************************\n");
    printf("\t Latency: %.3f [ms]\n", latency_cpu);
    printf("\n");

    /*
        Move to device(=GPU)
    */
    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc((void**)&A_d, mem_size_A);
    cudaMalloc((void**)&B_d, mem_size_B);
    cudaMalloc((void**)&C_d, mem_size_C);
    cudaMemcpy(A_d, A_h, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, mem_size_B, cudaMemcpyHostToDevice);

    /*
       GEMM using cuBLAS
    */
    // Warp-up operation
    for (int i = 0; i < 3; i++) {
        reset_vector(C_h, M*N);
        cublas_gemm(A_d, B_d, C_d, M, K, N);
    }

    cudaEvent_t start_cuda, end_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&end_cuda);
    float latency;
    printf("***********************************\n");
    printf("cuBLAS gemm\n");
    printf("***********************************\n");

    for (int i = 0; i < iter; i++) {
        reset_vector(C_h, M*N);
        
        cudaEventRecord(start_cuda, 0);
        cublas_gemm(A_d, B_d, C_d, M, K, N);
        cudaEventRecord(end_cuda, 0);
        cudaEventSynchronize(end_cuda);
        cudaEventElapsedTime(&latency, start_cuda, end_cuda);

        cudaMemcpy(C_h, C_d, mem_size_C, cudaMemcpyDeviceToHost);
    
        /*
           Check correctness
        */
        //print_vector(C_ref, M, N);
        //print_vector(C_h, M, N);
        printf("%dth iter\n", i+1);
        //compare_matrix(C_h, C_ref, M, N);
        printf("\t Latency: %.3f [ms]\n", latency);
    }

    // Event destroy
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(end_cuda);

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_ref);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
