/*
 * @Date: 2021-01-13 20:33:10
 * @LastEditors: mrz
 * @LastEditTime: 2021-01-23 10:13:25
 * @FilePath: /matrix_transport/transpose.h
 */

#ifndef _transpose_h
#define _transpose_h


// no change
void transpose0(double *dst, double *src, int dim);
void create_matrix(double *A, int dim, int circle);
void print_matrix(double *A, int dim);

//change transport

//1.循环分块
void transpose1(double *dst, double *src, int dim, int B);
void trans_block4x4(const __m256d s1, const __m256d s2, const __m256d s3, const __m256d s4, double *d1, 
double *d2, double *d3, double *d4);

//只用AVX指令集
void transpose2(double *x, double *y, int N);

//分块+块内使用AVX指令集
void transpose3(double *x, double *y, int N, int B);

//划分为8x8的子块，这样子块的每一行都恰好是一个Cache Line的大小
void transpose4(double *y, double *x, int N);
void tran_block8x8(const double *s1,const double *s2,const double* s3,const double *s4,
                          const double *s5,const double *s6,const double *s7,const double *s8,
                          double *d1,double *d2,double *d3,double *d4,
                          double *d5,double *d6,double *d7,double *d8);

//直接写入内存
void trans_block4x4_dir_ram(__m256d s1,__m256d s2,__m256d s3,__m256d s4, double *d1, 
double *d2, double *d3, double *d4);
void tran_block8x8_dir_ram(const double *s1,const double *s2,const double* s3,const double *s4,
                          const double *s5,const double *s6,const double *s7,const double *s8,
                          double *d1,double *d2,double *d3,double *d4,
                          double *d5,double *d6,double *d7,double *d8);

void transpose5(double *y, double *x, int n);

//分块+划分为8x8的子块+直接写入内存
void transpose6(double *y, double *x, int n, int B);
#endif