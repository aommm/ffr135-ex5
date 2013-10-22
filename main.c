#include "main.h"

// Network parameters
int m = 2; // dimension of input units vector
int q = 5; // dimension of hidden units vector
int p_t = 1000; // number of patterns in training set

bool rand_mode = true;
float rand_coeff = 0.1;
float eta = 0.001;

int n_max = 20000;

// Network states
float *I; // Laser intensity vector

float *xi;      // Input units vector
float *V;       // Hidden units vector
float *V_next;  // Hidden units vector (next step)
float zeta;     // Output unit expected value
float O;        // Output unit actual value

float **Wa; // hidden-hidden weights
float **Wb; // input-hidden weights
float *Wc;  // output weights

float ***lambda; // Change-in-activation-states-thingy
float ***lambda_next; // Change-in-activation-states-thingy

float sdev = 46.8754;
float mean = 59.894;

int main() {
    printf("helo");
    fflush(stdout);


    init();

    float **initial_a_matrix = copy_mat(Wa, q, q);
    float **initial_b_matrix = copy_mat(Wb, q, (m+1));

    printf("\n initial_a_matrix:\n");
    print_mat(initial_a_matrix, q, q);
    fflush(stdout);

    training();

    printf("hela");
    fflush(stdout);

    float **matrix_a_diff = mat_mat_sub(Wa, initial_a_matrix, q, q);
    float **matrix_b_diff = mat_mat_sub(Wb, initial_b_matrix, q, (m+1));

    printf("\n Wa:\n");
    print_mat(Wa, q, q);


    printf("\n Wa diff:\n");
    print_mat(matrix_a_diff, q, q);
    printf("\n Wb diff:\n");
    print_mat(matrix_b_diff, q, (m+1));
    fflush(stdout);


//    simulate(1);
    // return 0;
}

void training() {
    int p = 0;
    float H_sum = 0;
    unlink("energy.txt");
    FILE *file = fopen("energy.txt", "w");

    for (int n=1; n<=n_max; n++) {

        if (p>=998) p = 0;
        xi[0] = I[p];
        xi[1] = I[p+1];
        zeta = I[p+2];
        p++;

        forward(n);
        H_sum += h();
        fprintf(file, "%d\t%f\n", n, (H_sum/n));
        printf("n: %d", n);
        fflush(stdout);
        backward(n);
        printf("n2");
        fflush(stdout);
        next();
        printf("n3");
        fflush(stdout);

        if (n%100==0) {
//            printf("hej! V:\n");
            // print_vec(V, q);
            if (0) printf("helo delo");
//            printf("n: %d, gdiff: %f\n", n, greatest_diff(V, q));

        }

    }
    printf("Energy: %f, last output: %f\n", (H_sum/n_max), (O*sdev+mean));
    fflush(stdout);

    // system("pause");
    fclose(file);
}

void simulate(int k_max) {

    char file_name[20];
    sprintf(file_name, "energy_k%d.txt", k_max);
    FILE *file = fopen(file_name, "w");

    float *output = new_vec(k_max+2);
    output[0] = I[0];
    output[1] = I[1];

    int p = 0;
    int k = 0;
    for (int n=1; n<=n_max; n++) {

        if (p>=998) p = 0;
        if (k==k_max) {
            output[0] = I[p];
            output[1] = I[p+1];
            k=0;
        }
        xi[0] = output[k];
        xi[1] = output[k+1];
        k++;
        p++;

        forward(n);
//        O = k+5;
        fprintf(file, "%d\t%f\n", n, O);
        output[k+2] = O;

    }
    fclose(file);
}

void forward(int n) {

    printf("a");
    fflush(stdout);
    // exit(0);
    V_next = new_vec_zero(q);
    printf("b");
    fflush(stdout);
    if (V_next == NULL) {perror("ERROR: V_next malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    // Calculate hidden states
    for (int j=0;j<q;j++) {
        V_next[j] = g(vec_vec_mult(Wa[j], V, q) + vec_vec_mult(Wb[j], xi, (m+1)));
    }
    printf("c");
    fflush(stdout);
    // Calculate output state
    O = vec_vec_mult(Wc, V, q);

}

void backward(int n) {
    printf("d");
    fflush(stdout);
    // Perform concatenations of weights and states
    float **wjs = malloc(q * sizeof *wjs);
    if (wjs == NULL) {perror("ERROR: wjs malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    for(int j=0;j<q;j++) wjs[j] = vec_vec_concat(Wa[j], Wb[j], q, (m+1));
    printf("e");
    fflush(stdout);
    float *u = vec_vec_concat(V, xi, q, (m+1));
    printf("f");
    fflush(stdout);
    // Use vector instead of diagonal matrix
    float *phi_vec = new_vec(q);
    if (phi_vec == NULL) {perror("ERROR: phi_vec malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
//    printf("hej");
    for(int j=0; j<q; j++) phi_vec[j] = g_prime(vec_vec_mult(wjs[j], u, (q+m+1)));
    printf("g");
    fflush(stdout);
//    printf("phi:\n");
//    for(int j=0;j<q;j++) print_vec(phi_vec, q);
//    exit(1);

    // Update all neurons
    for (int j=0; j<q; j++) {

        update_lambda(j, phi_vec, u);

        // Calculate error
        float E = zeta - O;
        // Calculate delta-weights
        float *soon_enough = vec_mat_mult(Wc, lambda[j], q, (q+m+1));
        float *almost_done = vec_scalar_mult(soon_enough, eta, (q+m+1));
        float *delta_wjs = vec_scalar_mult(almost_done, E, (q+m+1));

//        printf("E:%f\nsoon_enough:\n", E);
//        print_vec(soon_enough, (q+m+1));
//        printf("almost_done:\n");
//        print_vec(almost_done, (q+m+1));

//        printf("(n=%d, j=%d) delta_wjs:\n", n, j);
//        print_vec(delta_wjs, (q+m+1));

        free(soon_enough);
        free(almost_done);

        // Apply delta-weights

        for (int i=0; i<q; i++) { // Hidden weights
//            printf("applying delta to hidden weight: %f\n", delta_wjs[i]);
            Wa[j][i] += delta_wjs[i];
        }

        for (int i=0; i<(m+1); i++) { // Input weights
//            printf("applying delta to input weight: %f\n", delta_wjs[i+q]);
            Wb[j][i] += delta_wjs[i+q];
        }

        free(delta_wjs);
    }
//    printf("--- next up: n=%d\n",n+1);
    printf("h");
    fflush(stdout);
    // Finally, clean up
    free_mat(wjs, q);
    free(u);
    printf("i");
    fflush(stdout);
}

void next() {

    // "Go to" next state
    free(V);
    V = V_next;
    for (int j=0;j<q;j++) {
        printf("lambda[%d]: %p\n", j, lambda[j]); fflush(stdout);
        free_mat(lambda[j], q);
        // lambda[j] = lambda_next[j];
    }
    exit(1);
}

void update_lambda(int j, float *phi_vec, float *u) {
    // Calculate new lambda
    float **product = mat_mat_mult(Wa, lambda[j], q, q, (q+m+1));

    printf("product %p:\n", product);
    for (int i=0;i<q;i++) {
        printf("row addr:%p\n", &product[i]); fflush(stdout);
    }
//    print_mat(product, q, (q+m+1));

    // Add u to only j'th row
    float *jth_row = vec_vec_add(product[j], u, (q+m+1));
    free(product[j]);
    product[j] = jth_row;

//    printf("product after add:\n");
//    print_mat(product, q, (q+m+1));

    // "phi matrix multiplication"
    for (int a=0;a<q; a++) { // row
        for(int b=0; b<(q+m+1); b++) { // col
            product[a][b] *= phi_vec[a];
        }
    }

//    printf("product after mult:\n");
//    print_mat(product, q, (q+m+1));

//    exit(1);

    // free(product[j]);
    // printf("freed %d\n", j);
    // fflush(stdout);

    for (int i=0;i<q;i++) {

    }

    lambda_next[j] = product;
}

float g(float x) {
    return tanh(x);
}
float g_prime(float x) {
    return (1-pow(tanh(x), 2));
}
float h() {
    float E = zeta-O;
    E *= sdev;
    E += mean;
    return (pow(E, 2) / 2);
}

// Initialise program
void init() {
    srand(time(NULL));
    rand();

    I = read_patterns(1000, "laser.norm.txt");

    xi = malloc((m+1) * sizeof *xi);
    if (xi == NULL) {perror("ERROR: xi malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    xi[2] = 1; // Add hax node, for threshold
    V = new_vec_zero(q);
    if (V == NULL) {perror("ERROR: V malloc failed"); fflush(stderr); fflush(stdout); exit(1);}

    Wa = generate_weight_matrix(q,q);
    Wb = generate_weight_matrix(q,(m+1));
    Wc = generate_weight_vector(q);

    lambda = malloc(q * sizeof *lambda);
    if (lambda == NULL) {perror("ERROR: lambda malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    lambda_next = malloc(q * sizeof *lambda_next);
    if (lambda_next == NULL) {perror("ERROR: lambda_next malloc failed"); fflush(stderr); fflush(stdout); exit(1);}

    for (int j=0; j<q; j++) {
        lambda[j] = new_mat_zero(q, (q+m+1));
        if (lambda[j] == NULL) {perror("ERROR: lambda[j] malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
        lambda_next[j] = new_mat_zero(q, (q+m+1));
        if (lambda_next[j] == NULL) {perror("ERROR: lambda_next[j] malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    }

}

float **generate_weight_matrix(int m, int n) {
    float **weights = malloc(m * sizeof *weights);
    if (weights == NULL) {perror("ERROR: weights mat malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    for (int i=0; i<m; i++) {
        weights[i] = malloc(n * sizeof *weights[i]);
        if (weights[i] == NULL) {perror("ERROR: weights[i] mat malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    }

    for (int i=0;i<m; i++) {
        for (int j=0;j<n; j++) {
            weights[i][j] = random_weight();
        }
    }
    return weights;
}
float *generate_weight_vector(int m) {
    float *weights = malloc(m * sizeof *weights);
    if (weights == NULL) {perror("ERROR: weights vec malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    for (int i=0;i<m; i++) {
        weights[i] = random_weight();
    }
    return weights;
}
float random_weight() {
    if (!rand_mode) return 0.01;
    float rando = (((float)rand()/RAND_MAX) * rand_coeff) - (rand_coeff/2);
    return rando;
}

// Read 1 value for each row
float *read_patterns(int p, char *file_name) {
    FILE *f = fopen(file_name, "r");
    if (f == NULL) {
        printf("%s \n", file_name);
        printf("Error opening file!\n");
        exit(1);
    }
    float *pat = malloc(p * sizeof *pat);
    if (pat == NULL) {perror("ERROR: pat malloc failed"); fflush(stderr); fflush(stdout); exit(1);}
    for(int i=0;i<p;i++) {
        fscanf(f, "%f", &pat[i]);
    }
    fclose(f);
    return pat;
}
