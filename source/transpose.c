/*
 * @Date: 2021-01-13 20:34:44
 * @LastEditors: mrz
 * @LastEditTime: 2021-01-23 14:19:00
 * @FilePath: /matrix_transport/transpose.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include "transpose.h"

void transpose0(double *dst, double *src, int dim)
{
    int i, j;
    for (i = 0; i < dim; i++)
        for (j = 0; j < dim; j++)
            dst[j * dim + i] = src[i * dim + j];
}

void create_matrix(double *A, int dim, int circle)
{
    unsigned int times = (unsigned int)time(NULL);
    srand(times * (circle+1));
    int l, n;
    for (l = 0; l < dim; l++)
    {
        for (n = 0; n < dim; n++)
        {
            A[l * dim + n] = (double)rand()/(RAND_MAX) ;
        }
    }
}

void print_matrix(double *A, int dim)
{
    int l, n;
    for (l = 0; l < dim; l++)
    {
        for (n = 0; n < dim; n++)
        {
            printf("%lf\t", A[l * dim + n]);
        }
        printf("\n");
    }
}

//change transpose

//分块
void transpose1(double *dst, double *src, int dim, int B)
{
    int i, j, i1, j1;
    for (i = 0; i < dim; i += B)
    {
        for (j = 0; j < dim; j += B)
        {
            for (i1 = i; i1 < i + B; i1++)
            {
                for (j1 = j; j1 < j + B; j1++)
                {
                    dst[j1 * dim + i1] = src[i1 * dim + j1];
                }
            }
        }
    }
}


/*
我们假设4x4的子块是
a0	a1	a2	a3
b0	b1	b2	b3
c0	c1	c2	c3
d0	d1	d2	d3
目标子块即为
a0	b0	c0	d0
a1	b1	c1	d1
a2	b2	c2	d2
a3	b3	c3	d3
*/
void trans_block4x4(__m256d s1,__m256d s2,__m256d s3,__m256d s4, double *d1, 
double *d2, double *d3, double *d4)
{
    __m256d t1,t2,t3,t4,t5,t6,t7,t8;

    t1=_mm256_permute4x64_pd(s1,0b01001110);//将第一行数据进行交换得到a2,a3,a0,a1
    t2=_mm256_permute4x64_pd(s2,0b01001110);//将第二行数据进行交换得到b2,b3,b0,b1
    t3=_mm256_permute4x64_pd(s3,0b01001110);//将第三行数据进行交换得到c2,c3,c0,c1
    t4=_mm256_permute4x64_pd(s4,0b01001110);//将第四行数据进行交换得到d2,d3,d0,d1
    t5=_mm256_blend_pd(s1,t3,0b1100);//合并原序列第一行和重排序列第三行得到a0,a1,c0,c1
    t6=_mm256_blend_pd(s2,t4,0b1100);//b0,b1,d0,d1
    t7=_mm256_blend_pd(t1,s3,0b1100);//a2,a3,c2,c3
    t8=_mm256_blend_pd(t2,s4,0b1100);//b2,b3,d2,d3
    s1=_mm256_unpacklo_pd(t5,t6);//生成转置子块，并写入对应位置a0,b0,c0,d0
    s2=_mm256_unpackhi_pd(t5,t6);//a1,b1,c1,d1
    s3=_mm256_unpacklo_pd(t7,t8);//a2,b2,c2,d2
    s4=_mm256_unpackhi_pd(t7,t8);//a3,b3,c3,d3
    _mm256_store_pd(d1,s1);
    _mm256_store_pd(d2,s2);
    _mm256_store_pd(d3,s3);
    _mm256_store_pd(d4,s4);    
}

//AVX指令集
void transpose2(double *y, double *x, int N)
{
    double *p1,*p2,*p3,*p4;
    double *d1,*d2,*d3,*d4,*t=y;
    d1=y;d2=y+N;d3=y+2*N;d4=y+3*N;
    p1=x;p2=x+N;p3=x+2*N;p4=x+3*N;
    t+=4;
    for(int i=0;i<N/4;i++)
    {
        for(int j=0;j<N/4;j++)
        {
            __m256d s1,s2,s3,s4;
            s1=_mm256_load_pd(p1);
            s2=_mm256_load_pd(p2);
            s3=_mm256_load_pd(p3);
            s4=_mm256_load_pd(p4);
            trans_block4x4(s1, s2, s3, s4, d1, d2, d3, d4);
            p1+=4;p2+=4;p3+=4;p4+=4;
            d1+=4*N;d2+=4*N;d3+=4*N;d4+=4*N;
        }
        p1+=3*N;p2+=3*N;p3+=3*N;p4+=3*N;
        d1=t;d2=t+N;d3=t+2*N;d4=t+3*N;
        t+=4;
    } 
}

//AVX指令集加分块
void transpose3(double *y, double *x, int N, int B)
{
    double *p1,*p2,*p3,*p4;
    double *d1,*d2,*d3,*d4,*t=y,*t2;
    int n = B/4;

    int i, j, i1, j1; 
    for (i = 0; i < N; i += B)
    {
        for (j = 0; j < N; j += B)
        {
            d1=y+j*N+i;d2=d1+N;d3=d2+N;d4=d3+N;
            p1=x+i*N+j;p2=p1+N;p3=p2+N;p4=p3+N;
            t = d1+4;
            t2 = p1 + 4*N;
            for(i1 = 0; i1 < n; i1++)
                {
                    for(j1 = 0; j1 < n; j1++)
                        {
                            __m256d s1,s2,s3,s4;
                            s1=_mm256_load_pd(p1);
                            s2=_mm256_load_pd(p2);
                            s3=_mm256_load_pd(p3);
                            s4=_mm256_load_pd(p4);
                            trans_block4x4(s1, s2, s3, s4, d1, d2, d3, d4);
                            p1+=4;p2+=4;p3+=4;p4+=4;
                            d1+=4*N;d2+=4*N;d3+=4*N;d4+=4*N;
                        }
                        p1=t2;p2=p1+N;p3=p2+N;p4=p3+N;
                        d1=t;d2=d1+N;d3=d2+N;d4=d3+N;
                        t+=4;
                        t2+=4*N;
                } 
        }
    }
}




// change matrix 8*8

void tran_block8x8(const double *s1,const double *s2,const double* s3,const double *s4,
                          const double *s5,const double *s6,const double *s7,const double *s8,
                          double *d1,double *d2,double *d3,double *d4,
                          double *d5,double *d6,double *d7,double *d8)
{
    __m256d t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16;
    t1=_mm256_load_pd(s1);
    t5=_mm256_load_pd(s1+4);
    t2=_mm256_load_pd(s2);
    t6=_mm256_load_pd(s2+4);
    t3=_mm256_load_pd(s3);
    t7=_mm256_load_pd(s3+4);
    t4=_mm256_load_pd(s4);
    t8=_mm256_load_pd(s4+4);
    t9=_mm256_load_pd(s5);
    t13=_mm256_load_pd(s5+4);
    t10=_mm256_load_pd(s6);
    t14=_mm256_load_pd(s6+4);
    t11=_mm256_load_pd(s7);
    t15=_mm256_load_pd(s7+4);
    t12=_mm256_load_pd(s8);
    t16=_mm256_load_pd(s8+4);
    trans_block4x4(t1,t2,t3,t4,d1,d2,d3,d4);
    trans_block4x4(t5,t6,t7,t8,d5,d6,d7,d8);
    trans_block4x4(t9,t10,t11,t12,d1+4,d2+4,d3+4,d4+4);
    trans_block4x4(t13,t14,t15,t16,d5+4,d6+4,d7+4,d8+4);
}


void transpose4(double *y, double *x, int n){
    double *s1=x,*s2=x+n,*s3=x+2*n,*s4=x+3*n,
           *s5=x+4*n,*s6=x+5*n,*s7=x+6*n,*s8=x+7*n,
           *d1=y,*d2=y+n,*d3=y+2*n,*d4=y+3*n,
           *d5=y+4*n,*d6=y+5*n,*d7=y+6*n,*d8=y+7*n;
    for(int i=0;i<n;i+=8)
    {
        d1=y+i,d2=d1+n,d3=d1+2*n,d4=d1+3*n,
        d5=d1+4*n,d6=d1+5*n,d7=d1+6*n,d8=d1+7*n;
        int j=n;
        while(j>=8)
        {
            tran_block8x8(s1,s2,s3,s4,s5,s6,s7,s8,d1,d2,d3,d4,d5,d6,d7,d8);
            s1+=8;s2+=8;s3+=8;s4+=8;
            s5+=8;s6+=8;s7+=8;s8+=8;
            d1+=8*n;d2+=8*n;d3+=8*n;d4+=8*n;
            d5+=8*n;d6+=8*n;d7+=8*n;d8+=8*n;
            j-=8;
        }
        s1+=7*n;s2+=7*n;s3+=7*n;s4+=7*n;
        s5+=7*n;s6+=7*n;s7+=7*n;s8+=7*n;
    }
}


//AVX直接写内存

void trans_block4x4_dir_ram(__m256d s1,__m256d s2,__m256d s3,__m256d s4, double *d1, 
double *d2, double *d3, double *d4)
{
    __m256d t1,t2,t3,t4,t5,t6,t7,t8;

    t1=_mm256_permute4x64_pd(s1,0b01001110);//将第一行数据进行交换得到a2,a3,a0,a1
    t2=_mm256_permute4x64_pd(s2,0b01001110);//将第二行数据进行交换得到b2,b3,b0,b1
    t3=_mm256_permute4x64_pd(s3,0b01001110);//将第三行数据进行交换得到c2,c3,c0,c1
    t4=_mm256_permute4x64_pd(s4,0b01001110);//将第四行数据进行交换得到d2,d3,d0,d1
    t5=_mm256_blend_pd(s1,t3,0b1100);//合并原序列第一行和重排序列第三行得到a0,a1,c0,c1
    t6=_mm256_blend_pd(s2,t4,0b1100);//b0,b1,d0,d1
    t7=_mm256_blend_pd(t1,s3,0b1100);//a2,a3,c2,c3
    t8=_mm256_blend_pd(t2,s4,0b1100);//b2,b3,d2,d3
    s1=_mm256_unpacklo_pd(t5,t6);//生成转置子块，并写入对应位置a0,b0,c0,d0
    s2=_mm256_unpackhi_pd(t5,t6);//a1,b1,c1,d1
    s3=_mm256_unpacklo_pd(t7,t8);//a2,b2,c2,d2
    s4=_mm256_unpackhi_pd(t7,t8);//a3,b3,c3,d3
    _mm256_stream_pd(d1,s1);
    _mm256_stream_pd(d2,s2);
    _mm256_stream_pd(d3,s3);
    _mm256_stream_pd(d4,s4);    
}

void tran_block8x8_dir_ram(const double *s1,const double *s2,const double* s3,const double *s4,
                          const double *s5,const double *s6,const double *s7,const double *s8,
                          double *d1,double *d2,double *d3,double *d4,
                          double *d5,double *d6,double *d7,double *d8)
{
    __m256d t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16;
    t1=_mm256_load_pd(s1);
    t5=_mm256_load_pd(s1+4);
    t2=_mm256_load_pd(s2);
    t6=_mm256_load_pd(s2+4);
    t3=_mm256_load_pd(s3);
    t7=_mm256_load_pd(s3+4);
    t4=_mm256_load_pd(s4);
    t8=_mm256_load_pd(s4+4);
    t9=_mm256_load_pd(s5);
    t13=_mm256_load_pd(s5+4);
    t10=_mm256_load_pd(s6);
    t14=_mm256_load_pd(s6+4);
    t11=_mm256_load_pd(s7);
    t15=_mm256_load_pd(s7+4);
    t12=_mm256_load_pd(s8);
    t16=_mm256_load_pd(s8+4);
    trans_block4x4_dir_ram(t1,t2,t3,t4,d1,d2,d3,d4);
    trans_block4x4_dir_ram(t5,t6,t7,t8,d5,d6,d7,d8);
    trans_block4x4_dir_ram(t9,t10,t11,t12,d1+4,d2+4,d3+4,d4+4);
    trans_block4x4_dir_ram(t13,t14,t15,t16,d5+4,d6+4,d7+4,d8+4);
}

void transpose5(double *y, double *x, int n){
    double *s1=x,*s2=x+n,*s3=x+2*n,*s4=x+3*n,
           *s5=x+4*n,*s6=x+5*n,*s7=x+6*n,*s8=x+7*n,
           *d1=y,*d2=y+n,*d3=y+2*n,*d4=y+3*n,
           *d5=y+4*n,*d6=y+5*n,*d7=y+6*n,*d8=y+7*n;
    for(int i=0;i<n;i+=8)
    {
        d1=y+i,d2=d1+n,d3=d1+2*n,d4=d1+3*n,
        d5=d1+4*n,d6=d1+5*n,d7=d1+6*n,d8=d1+7*n;
        int j=n;
        while(j>=8)
        {
            tran_block8x8_dir_ram(s1,s2,s3,s4,s5,s6,s7,s8,d1,d2,d3,d4,d5,d6,d7,d8);
            s1+=8;s2+=8;s3+=8;s4+=8;
            s5+=8;s6+=8;s7+=8;s8+=8;
            d1+=8*n;d2+=8*n;d3+=8*n;d4+=8*n;
            d5+=8*n;d6+=8*n;d7+=8*n;d8+=8*n;
            j-=8;
        }
        s1+=7*n;s2+=7*n;s3+=7*n;s4+=7*n;
        s5+=7*n;s6+=7*n;s7+=7*n;s8+=7*n;
    }
}


void transpose6(double *y, double *x, int n, int B)
{
    double *s1,*s2,*s3,*s4,
           *s5,*s6,*s7,*s8,
           *d1,*d2,*d3,*d4,
           *d5,*d6,*d7,*d8,
           *t, *t2;
    int b = B/8;

    int i, j, i1, j1; 
    for (i = 0; i < n; i += B)
    {
        for (j = 0; j < n; j += B)
        {
            s1=x+i*n+j,s2=s1+n,s3=s1+2*n,s4=s1+3*n,
            s5=s1+4*n,s6=s1+5*n,s7=s1+6*n,s8=s1+7*n,
            d1=y+j*n+i,d2=d1+n,d3=d1+2*n,d4=d1+3*n,
            d5=d1+4*n,d6=d1+5*n,d7=d1+6*n,d8=d1+7*n;
            t = d1+8;
            t2 = s1 + 8*n;
            for(i1 = 0; i1 < b; i1++)
                {
                    for(j1 = 0; j1 < b; j1++)
                        {
                            tran_block8x8_dir_ram(s1,s2,s3,s4,s5,s6,s7,s8,d1,d2,d3,d4,d5,d6,d7,d8);
                            s1+=8;s2+=8;s3+=8;s4+=8;
                            s5+=8;s6+=8;s7+=8;s8+=8;
                            d1+=8*n;d2+=8*n;d3+=8*n;d4+=8*n;
                            d5+=8*n;d6+=8*n;d7+=8*n;d8+=8*n;
                        }
                        s1=t2;s2=s1+n;s3=s2+n;s4=s3+n;s5=s4+n;s6=s5+n;s7=s6+n;s8=s7+n;
                        d1=t;d2=d1+n;d3=d2+n;d4=d3+n;d5=d4+n;d6=d5+n;d7=d6+n;d8=d7+n;
                        t+=8;
                        t2+=8*n;
                } 
        }
    }
}
