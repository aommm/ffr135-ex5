#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>


// Matrix utility functions

float vec_vec_mult(float *vec1, float *vec2, int d); // (Dot product; scalar)
float *mat_vec_mult(float **mat, float *vec, int n, int p); // (p vector)
float *vec_mat_mult(float *vec, float **mat, int n, int p); // (p vector)
float **mat_mat_mult(float **mat1, float **mat2, int m, int n, int p); // (mxp matrix)

float *vec_scalar_mult(float *vec, float s, int d);
float **mat_scalar_mult(float **mat, float s, int m, int n);

float *vec_vec_add(float *vec1, float *vec2, int d);
float *vec_vec_sub(float *vec1, float *vec2, int d);
float **mat_mat_add(float **mat1, float **mat2, int m, int n);
float **mat_mat_sub(float **mat1, float **mat2, int m, int n);

float *vec_vec_concat(float *vec1, float *vec2, int d1, int d2);

float **new_mat(int m, int n);
float **new_mat_zero(int m, int n);
float *new_vec(int m);
float *new_vec_zero(int d);

float **copy_mat(float **mat, int m, int n);
void free_mat(float **mat, int m);

void print_mat(float **mat, int m, int n);
void print_vec(float *vec, int d);

#endif // MATRIX_H_INCLUDED
