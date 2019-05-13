#include "math_func.h"
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>
#include <omp.h>
#include <Eigen/Dense>

#include "util.h"

#define NUM_THREADS 4

using namespace std;
using namespace Eigen;

namespace micronet {

void gemm_nn(int M, int N, int K, float ALPHA,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
        const float *A, int lda,
        const float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void matmul_nn(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc);
void matmul_nt(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc);
void matmul_tn(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc);
void matmul_tt(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
        const float *A, int lda,
        const float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        matmul_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        matmul_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        matmul_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        matmul_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void add_scalar(int n, float scalar, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] += scalar;
    }
}

void add(int n, const float* a, float alpha, const float* b, float beta, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = alpha * a[i] + beta * b[i];
    }
}

float sum(int n, float scalar, const float* a) {
    float tmp = scalar;
    for (int i = 0; i < n; ++i) {
        tmp += a[i];
    }
    return tmp;
}

float* mat_transpose(int n_rows, int n_cols, const float* mat) {
    float* trans_mate = new float[n_rows*n_cols];
    for (int i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_cols; ++j) {
            trans_mate[j*n_rows+i] = mat[i*n_cols+j];
        }
    }
    return trans_mate;
}

float sdot_8(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	float s, t[8];
	t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = t[6] = t[7] = 0.0f;
	for (i = 0; i < n8; i += 8) {
		t[0] += x[i+0] * y[i+0];
		t[1] += x[i+1] * y[i+1];
		t[2] += x[i+2] * y[i+2];
		t[3] += x[i+3] * y[i+3];
		t[4] += x[i+4] * y[i+4];
		t[5] += x[i+5] * y[i+5];
		t[6] += x[i+6] * y[i+6];
		t[7] += x[i+7] * y[i+7];
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	s += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
	return s;
}

#ifdef __SSE__
#include <xmmintrin.h>

float sdot_sse(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}
#endif // __SSE__

void mat_mul3(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha) {
    int i, j, n_b_rows = n_a_cols;
    float* bT = mat_transpose(n_b_rows, n_b_cols, b);
    for (i = 0; i < n_a_rows; ++i)
		for (j = 0; j < n_b_cols; ++j)
			c[i*n_b_cols+j] += alpha * sdot_8(n_a_cols, a+i*n_a_cols, bT+j*n_b_rows);
    delete[] bT;
}

#ifdef __SSE__
void mat_mul4(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha) {
    int i, j, ii, jj, x = 16, n_b_rows = n_a_cols;
	float* bT = mat_transpose(n_b_rows, n_b_cols, b);
	for (i = 0; i < n_a_rows; i += x) {
		for (j = 0; j < n_b_cols; j += x) {
			int je = n_b_cols < j + x? n_b_cols : j + x;
			int ie = n_a_rows < i + x? n_a_rows : i + x;
			for (ii = i; ii < ie; ++ii)
				for (jj = j; jj < je; ++jj)
					c[ii*n_b_cols+jj] += alpha * sdot_sse(n_a_cols, a+ii*n_a_cols, bT+jj*n_b_rows);
		}
	}
	delete[] bT;
}
#endif // __SSE__

void eigen_filled(MatrixXf& X, const float* x) {
    int rows = X.rows(), cols = X.cols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X(i, j) = x[i*cols+j];
        }
    }
}

void eigen_defilled(MatrixXf& X, float* x, float alpha) {
    int rows = X.rows(), cols = X.cols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            x[i*cols+j] += alpha * X(i, j);
        }
    }
}

void mat_mul5(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha) {
    MatrixXf A(n_a_rows, n_a_cols), B(n_a_cols, n_b_cols);
    eigen_filled(A, a);
    eigen_filled(B, b);
    //Timer t;
    MatrixXf C = A * B;
    //cout << t.elapsed() << endl;
    eigen_defilled(C, c, alpha);
}

void matmul_nn(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc) {
    int n_a_rows = M, n_a_cols = K, n_b_cols = N;

    #ifdef __SSE__
    mat_mul4(n_a_rows, n_a_cols, A, n_b_cols, B, C, ALPHA);
    return;
    #endif // __SSE__

    mat_mul3(n_a_rows, n_a_cols, A, n_b_cols, B, C, ALPHA);
}

void matmul_nt(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc) {
    float* b_tmp = mat_transpose(N, ldb, B);
    int n_a_rows = M, n_a_cols = K, n_b_cols = N;

    #ifdef __SSE__
    mat_mul4(n_a_rows, n_a_cols, A, n_b_cols, b_tmp, C, ALPHA);
    delete[] b_tmp;
    return;
    #endif // __SSE__

    mat_mul3(n_a_rows, n_a_cols, A, n_b_cols, b_tmp, C, ALPHA);
    delete[] b_tmp;
}

void matmul_tn(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc) {
    float* a_tmp = mat_transpose(K, M, A);
    int n_a_rows = M, n_a_cols = K, n_b_cols = N;

    #ifdef __SSE__
    mat_mul4(n_a_rows, n_a_cols, a_tmp, n_b_cols, B, C, ALPHA);
    delete[] a_tmp;
    return;
    #endif // __SSE__

    mat_mul3(n_a_rows, n_a_cols, a_tmp, n_b_cols, B, C, ALPHA);
    delete[] a_tmp;
}

void matmul_tt(int M, int N, int K, float ALPHA,
               const float* A, int lda, const float* B, int ldb,
               float* C, int ldc) {
    float* a_tmp = mat_transpose(K, M, A);
    float* b_tmp = mat_transpose(N, ldb, B);
    int n_a_rows = M, n_a_cols = K, n_b_cols = N;

    #ifdef __SSE__
    mat_mul4(n_a_rows, n_a_cols, a_tmp, n_b_cols, b_tmp, C, ALPHA);
    delete[] a_tmp;
    delete[] b_tmp;
    return;
    #endif // __SSE__

    mat_mul3(n_a_rows, n_a_cols, a_tmp, n_b_cols, b_tmp, C, ALPHA);
    delete[] a_tmp;
    delete[] b_tmp;
}

inline float get_pixel(int row, int col, int n_cols, const float* img) {
    return img[row*n_cols+col];
}

void bilinear_interpolation(int n_rows, int n_cols, int n_channels, const float* img,
                            float h_rate, float w_rate, float* result_img) {
    int rows = int(n_rows * h_rate);
	int cols = int(n_cols * w_rate);
    for (int c = 0; c < n_channels; ++c) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                int px = (int)(x / w_rate);
			    int py = (int)(y / h_rate);
			    //if (px >= n_cols - 1 || py >= n_rows - 1) break;

			    float fx1 = (float)x / (float)w_rate - (float)px;
			    float fx2 = 1 - fx1;
			    float fy1 = (float)y / (float)h_rate - (float)py;
			    float fy2 = 1 - fy1;

			    float w1 = fx2*fy2;
			    float w2 = fx1*fy2;
			    float w3 = fx2*fy1;
			    float w4 = fx1*fy1;

			    float p1 = get_pixel(py, px, n_cols, img);
			    float p2 = px < n_cols - 1? get_pixel(py, px+1, n_cols, img): get_pixel(py, px, n_cols, img);
			    float p3 = py < n_rows - 1? get_pixel(py+1, px, n_cols, img): get_pixel(py, px, n_cols, img);
			    float p4 = px < n_cols - 1 && py < n_rows - 1? get_pixel(py+1, px+1, n_cols, img): get_pixel(py, px, n_cols, img);

			    result_img[y*cols+x] = w1*p1 + w2*p2 + w3*p3 + w4*p4;
            }
        }
        img += n_rows * n_cols;
        result_img += rows * cols;
    }
}

} // namespace micronet

