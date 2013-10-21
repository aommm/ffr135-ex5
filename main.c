#include "main.h"

// Network parameters
int m = 2; // dimension of input units vector
int q = 5; // dimension of hidden units vector
int p_t = 1000; // number of patterns in training set
float rand_coeff = 0.3;
float rand_d = 0;// -0.15;?
float eta = 0.1;

int n_max = 2000;

// Network states
float *I; // Laser intensity vector

float *xi;      // Input units vector
float *V;       // Hidden units vector
float *V_next;  // Hidden units vector (next step)
float zeta;     // Output unit expected value
float O;        // Output unit actual value

float **Wa; // hidden-hidden weights
float **Wb; // input-hidden weights
float *Wo;  // output weights

float ***lambda; // Change-in-activation-states-thingy
float ***lambda_next; // Change-in-activation-states-thingy

int main() {
    init();
    training();
    simulate(1);
}

void training() {
    int p = 0;
    float H_sum = 0;
    unlink("energy.txt");
    FILE *file = fopen("energy.txt", "w");

    for (int n=0;n<n_max; n++) {

        if (p>=998) p = 0;
        xi[0] = I[p];
        xi[1] = I[p+1];
        zeta = I[p+2];
        p++;

        forward(n);
        H_sum += h();
        fprintf(file, "%d\t%f\n", n, (H_sum/n));
        backward(n);
    }
    printf("Energy: %f\n", (H_sum/n_max));
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
    for (int n=0;n<n_max; n++) {

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
        O = k+5;
        fprintf(file, "%d\t%f\n", n, O);
        output[k+2] = O;

    }
    fclose(file);
}

void forward(int n) {
    float *V_next = new_vec(q);
    // Calculate hidden states
    for (int j=0;j<q;j++) {
        V_next[j] = g(vec_vec_mult(Wa[j], V, q) + vec_vec_mult(Wb[j], xi, (m+1)));
    }
    // "Go to" next state
    free(V);
    V = V_next;
    // Calculate output state
    O = vec_vec_mult(Wo, V_next, q);
}

void backward(int n) {
    // Perform concatenations of weights and states
    float **wjs = malloc(q * sizeof *wjs);
    for(int j=0;j<q;j++) wjs[j] = vec_vec_concat(Wa[j], Wb[j], q, (m+1));
    float *u = vec_vec_concat(V, xi, q, (m+1));

    // Use vector instead of diagonal matrix
    float *phi_vec = new_vec(q);
    for(int j=0;j<q;j++) phi_vec[j] = g_prime(vec_vec_mult(wjs[j], u, (q+m+1)));

    // Update all neurons
    for (int j=0; j<q; j++) {

        update_lambda(j, phi_vec, u);

        // Calculate error
        float E = zeta - O;
        // Calculate delta-weights
        float *soon_enough = vec_mat_mult(Wo, lambda[j], q, (q+m+1));
        float *almost_done = vec_scalar_mult(soon_enough, eta, (q+m+1));
        float *delta_wjs = vec_scalar_mult(almost_done, E, (q+m+1));
        free(soon_enough);
        free(almost_done);

        // Apply delta-weights
        for (int i=0; i<q; i++) { // Hidden weights
            Wa[j][i] += delta_wjs[i];
        }
        for (int i=0; i<(m+1); i++) { // Input weights
            Wb[j][i] += delta_wjs[i+q];
        }

        free(delta_wjs);
    }


//    exit(1);

    // Finally, clean up
    free_mat(wjs, q);
    free(u);
}

void update_lambda(int j, float *phi_vec, float *u) {
    // Calculate new lambda
    float **product = mat_mat_mult(Wa, lambda[j], q, q, (q+m+1));

    // Add u to only j'th row
    float *added_u = vec_vec_add(product[j], u, (q+m+1));
    free(product[j]);
    product[j] = added_u;


//    printf("product:\n");
//    print_mat(product, q, (q+m+1));
//    exit(1);

    // "phi matrix multiplication"
    for (int a=0;a<q; a++) { // row
        for(int b=0; b<q; b++) { // col
            product[a][b] *= phi_vec[a];
        }
    }
    // Phi matrix multiplication
//    float **almost_done = mat_mat_mult(phi, product, q, q, (q+m+1));
//    free(product);
    free_mat(lambda[j], q);
    lambda[j] = product;
}

float g(float x) {
    return tanh(x);
}
float g_prime(float x) {
    return (1-pow(tanh(x), 2));
}
float h() {
    float E = zeta-O;
    return (pow(E, 2) / 2);
}

// Initialise program
void init() {
    srand(time(NULL));
    rand();

    I = read_patterns(1000, "laser.txt");

    xi = malloc((m+1) * sizeof *xi);
    xi[2] = 1; // Add hax node, for threshold
    V = calloc(q, sizeof *V);

    Wa = generate_weight_matrix(q,q);
    Wb = generate_weight_matrix(q,(m+1));
    Wo = generate_weight_vector(q);

    lambda = malloc(q * sizeof *lambda);
    lambda_next = malloc(q * sizeof *lambda_next);

    for (int j=0; j<q; j++) {
//        lambda[j] = generate_weight_matrix(q, (q+m+1)); // ? initial values ?

        lambda[j] = new_mat_zero(q, (q+m+1)); // ? initial values ?
    }

}

float **generate_weight_matrix(int m, int n) {
    float **weights = malloc(m * sizeof(float*));
    for (int i=0; i<m; i++)
        weights[i] = malloc(n * sizeof(float));

    for (int i=0;i<m; i++) {
        for (int j=0;j<n; j++) {
            weights[i][j] = random_weight();
        }
    }
    return weights;
}
float **generate_weight_vector(int m) {
    float *weights = malloc(m * sizeof(float));
    for (int i=0;i<m; i++) {
        weights[i] = random_weight();
    }
    return weights;
}
float random_weight() {
    return -0.01;
    float rando = (((float)rand()/RAND_MAX) * rand_coeff) - rand_d;
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
    float *pat = malloc(sizeof(float) * p);
    for(int i=0;i<p;i++) {
        fscanf(f, "%f", &pat[i]);
    }
    fclose(f);
    return pat;
}
