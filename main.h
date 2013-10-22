#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED

#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// Main parts of program
void init();
void training();
void simulation();

void forward(int n);
void backward(int n);
void next();

// Init
float *read_patterns(int p, char *file_name);
float **generate_weight_matrix(int m, int n);
float *generate_weight_vector(int m);
float random_weight();

// Backward
void update_lambda(int j, float *phi_vec, float *u);

// Math stuff
float g(float x);
float g_prime(float x);
float h();

// Various
float add_to_file(char *file_name, int i, float f);

#endif // MAIN_H_INCLUDED
