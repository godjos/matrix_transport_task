/*
 * @Date: 2021-01-13 20:31:49
 * @LastEditors: mrz
 * @LastEditTime: 2021-01-23 10:29:57
 * @FilePath: /matrix_transport/main.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#include "transpose.h"

int main(int argc, char **argv)
{
    int i, dim = 16384, p = 0, circle = 10, arg_index = 1, t = 0, B = 8;
    double *dst, *src;
    clock_t start, finish;
    double duration = 0;
    
    while(arg_index < argc){
        if(strcmp(argv[arg_index], "-dim") == 0){
            dim = atoi(argv[++arg_index]);
            arg_index++;
        }
        else if(strcmp(argv[arg_index], "-p") == 0){
            p = atoi(argv[++arg_index]);
            arg_index++;
        }
        else if(strcmp(argv[arg_index], "-c") == 0){
            circle = atoi(argv[++arg_index]);
            arg_index++;
        }
        else if(strcmp(argv[arg_index], "-B") == 0){
            B = atoi(argv[++arg_index]);
            arg_index++;
        }
        else if(strcmp(argv[arg_index], "-t") == 0){
            t = atoi(argv[++arg_index]);
            if (t < 2){
                dst = (double *)malloc(dim * dim * sizeof(double));
                src = (double *)malloc(dim * dim * sizeof(double));
            }
            else {
                dst = (double*)_mm_malloc(sizeof(double)*dim*dim,32);
                src = (double*)_mm_malloc(sizeof(double)*dim*dim,32); 
            }
            arg_index++;
        }
        else {
            printf("Example:use no changed function transport  10*10 matrix, use 10 times circle and print result is %s -dim 10 -p 1 -c 10 -t 0", argv[0]);
            arg_index++;
        }
    }
       
    for (i = 0; i < circle; i++)
    {
        create_matrix(src, dim, i);
        if (t == 0){
            start = clock();
            transpose0(dst, src, dim);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;
        }
        else if (t == 1){
            start = clock();
            transpose1(dst, src, dim, B);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;
        }
        else if (t == 2){
            start = clock();
            transpose2(dst, src, dim);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;        
        }
        else if (t == 3){
            start = clock();
            transpose3(dst, src, dim, B);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;        
        }
        else if (t == 4){
            start = clock();
            transpose4(dst, src, dim);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;        
        }
        else if (t == 5){
            start = clock();
            transpose5(dst, src, dim);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;        
        }
        else if (t == 6){
            start = clock();
            transpose6(dst, src, dim, B);
            finish = clock();
            printf("%d circle: %f seconds\n", i, (double)(finish - start) / CLOCKS_PER_SEC);
            duration = duration + (double)(finish - start) / CLOCKS_PER_SEC;        
        }                         
    }   
    printf("average: %f seconds\n", duration / circle);

    //print result
    if(p == 1){
        print_matrix(src, dim);
        printf("\ntransport_matrix\n");
        print_matrix(dst, dim);
    }


    free(dst);
    free(src);
    return 0;
}
