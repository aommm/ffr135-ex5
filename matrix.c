#include "matrix.h"

// Multiplication
float vec_vec_mult(float *vec1, float *vec2, int d) {
    float r = 0;
    for(int i=0;i<d;i++) {
        if ( (vec1[i]>10000) || (vec2[i]>10000) ) errror("ERROR ETC");
        r += vec1[i]*vec2[i];

    }
    return r;
}
float *mat_vec_mult(float **mat, float *vec, int n, int p) {
    float *r = new_vec(n);
    for(int i=0;i<n;i++) {
        float sum = 0;
        for (int k=0;k<p;k++) {
            if ( (mat[i][k]>10000) || (vec[k]>10000) ) errror("ERROR ETC");
            sum += mat[i][k]*vec[k];

        }
        r[i] = sum;
    }
    return r;
}
float *vec_mat_mult(float *vec, float **mat, int n, int p) {
    float *r = new_vec(p);
    for(int i=0;i<p;i++) {
        float sum = 0;
        for (int j=0;j<n;j++) {
            if ( (mat[j][i]>10000) || (vec[j]>10000) ) errror("ERROR ETC");
            sum += vec[j]*mat[j][i];
        }
        r[i] = sum;
    }
    return r;
}
float **mat_mat_mult(float **mat1, float **mat2, int m, int n, int p) {
    float **r = new_mat(m, p);
    for(int i=0;i<m;i++) {
        for (int j=0;j<p;j++) {
            float sum = 0;
            for (int k=0;k<n;k++) {
                if ( (mat1[i][k]>10000) || (mat2[k][j]>10000) ) errror("ERROR ETC");
                sum += mat1[i][k]*mat2[k][j];

            }
            r[i][j] = sum;
        }
    }
    return r;
}

// Scalar multiplication
float *vec_scalar_mult(float *vec, float s, int d) {
    float *r = new_vec(d);
    for(int i=0;i<d;i++) {
        if ( (vec[i]>10000) ) errror("ERROR ETC");
        r[i] = s * vec[i];

    }
    return r;
}
float **mat_scalar_mult(float **mat, float s, int m, int n) {
    float **r = new_mat(m, n);
    for(int i=0;i<m;i++) {
        for (int j=0;j<n;j++) {
            if ( (mat[i][j]>10000) ) errror("ERROR ETC");
            r[i][j] = s * mat[i][j];

        }
    }
    return r;
}

// Addition
float *vec_vec_add(float *vec1, float *vec2, int d) {
    float *r = new_vec(d);
    for (int i=0;i<d;i++) {
        if ( (vec1[i]>10000) || (vec2[i]>10000) ) errror("ERROR ETC");
        r[i] = vec1[i] + vec2[i];

    }
    return r;
}
float **mat_mat_add(float **mat1, float **mat2, int m, int n) {
    float **r = new_mat(m, n);
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if ( (mat1[i][j]>10000) || (mat2[i][j]>10000) ) errror("ERROR ETC");
            r[i][j] = mat1[i][j] + mat2[i][j];

        }
    }
    return r;
}
float *vec_vec_sub(float *vec1, float *vec2, int d) {
    float *r = new_vec(d);
    for (int i=0;i<d;i++) {
        if ( (vec1[i]>10000) || (vec2[i]>10000) ) errror("ERROR ETC");
        r[i] = vec1[i] - vec2[i];

    }
    return r;
}
float **mat_mat_sub(float **mat1, float **mat2, int m, int n) {
    float **r = new_mat(m, n);
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if ( (mat1[i][j]>10000) || (mat2[i][j]>10000) ) errror("ERROR ETC");
            r[i][j] = mat1[i][j] - mat2[i][j];

        }
    }
    return r;
}

float *vec_vec_concat(float *vec1, float *vec2, int d1, int d2) {
    float *r = new_vec(d1+d2);
    for (int i=0; i<d1; i++) r[i] = vec1[i];
    for (int i=0; i<d2; i++) r[i+d1] = vec2[i];
    return r;
}

float **new_mat(int m, int n) {
    float **mat = malloc(m * sizeof(float*));
    if (mat == NULL) errror("ERROR: new_mat malloc failed");
    for (int i=0;i<m;i++) mat[i] = new_vec(n);
    return mat;
}
float **new_mat_zero(int m, int n) {
    float **mat = malloc(m * sizeof(float*));
    if (mat == NULL) errror("ERROR: new_mat_zero malloc failed");
    for (int i=0;i<m;i++) mat[i] = new_vec_zero(n);
    return mat;
}
float *new_vec(int d) {
    float *r = malloc(d * sizeof(float));
    if (r == NULL) errror("ERROR: new_vec malloc failed");
    return r;
}
float *new_vec_zero(int d) {
    float *r = malloc(d * sizeof *r);
    if (r == NULL) errror("ERROR: r malloc failed");
    for (int i=0;i<d; i++) r[i] = 0;
    return r;
}



void free_mat(float **mat, int m) {
    // printf("free_mat %p\n", mat); fflush(stdout);

    for (int i=0;i<m;i++) {
        // printf("gonna free row %d, addr:%p", i, &mat[m]); fflush(stdout);
        free(&mat[m]);
        // printf(" (done!)\n", i); fflush(stdout);
    }
    // printf("freeing mat\n"); fflush(stdout);
    // free(mat);
    // printf("freed mat\n"); fflush(stdout);
}
float **copy_mat(float **mat, int m, int n) {
    float **r = new_mat(m, n);
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if ( (mat[i][j]>10000) ) {errror("ERROR ETC"); fflush(stderr); fflush(stdout); exit(1); }
            r[i][j] = mat[i][j];
        }
    }
    return r;
}

void print_mat(float **mat, int m, int n) {
    for (int i=0; i<m; i++)
        for (int j=0;j<n;j++)
            printf("%f%c", mat[i][j], ((j+1)==n? '\n' : '\t'));
    fflush(stdout);
}
void print_vec(float *vec, int d) {
    for (int i=0; i<d; i++)
        printf("%f%c", vec[i], ((i+1)==d? '\n' : '\t'));
    fflush(stdout);
}

void errror(char *msg) {
   errror(msg);
    fflush(stderr);
    fflush(stdout);
    exit(1);
}
void prrint(char *msg) {
    printf("%s", msg);
    fflush(stdout);
}
