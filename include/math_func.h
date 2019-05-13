#ifndef MATH_FUNC_H
#define MATH_FUNC_H

namespace micronet {

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
                    const float *A, int lda,
                    const float *B, int ldb,
                    float BETA,
                    float *C, int ldc);

void add_scalar(int n, float scalar, float* y);
void add(int n, const float* a, float alpha, const float* b, float beta, float* y);
float sum(int n, float scalar, const float* a);

void mat_mul3(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha);
#ifdef __SSE__
void mat_mul4(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha);
#endif // __SSE__
void mat_mul5(int n_a_rows, int n_a_cols, const float* a, int n_b_cols, const float* b, float* c, float alpha);

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

void bilinear_interpolation(int n_rows, int n_cols, int n_channels, const float* img,
                            float h_rate, float w_rate, float* result_img);

} // namespace micronet

#endif // MATH_FUNC_H
