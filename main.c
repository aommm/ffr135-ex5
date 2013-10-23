#include "main.h"

// Network parameters
int m = 2; // dimension of input units vector
int q = 5; // dimension of hidden units vector
int p_t = 1000; // number of patterns
int p_v = 9093;

bool normalise_data = true;
bool validation_mode = false;
bool rand_mode = true;

//float rand_coeff = 0.1;
//float eta = 0.001;

float rand_coeff = 1;
float eta = 0.1;
int n_max = 20000;

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

float sdev = 46.8754; float sdev_v = 47.0727;
float mean = 59.894;  float mean_v = 59.8247;

int main(int argc, char** argv) {

    if (argc>1) {
        rand_coeff = atof(argv[1]);
    } if (argc>2) {
        eta = atof(argv[2]);
    } if (argc>3) {
        n_max = atof(argv[3]);
    } if (argc>4) {
        validation_mode = !!atoi(argv[4]);
        if (validation_mode) printf("asdf %d");
        exit(1);
    }

    init();

    // float **initial_a_matrix = copy_mat(Wa, q, q);
    // float **initial_b_matrix = copy_mat(Wb, q, (m+1));
    // prrint("\n Wa matrix before:\n");
    // print_mat(initial_a_matrix, q, q);

    train();

    // float **matrix_a_diff = mat_mat_sub(Wa, initial_a_matrix, q, q);
    // float **matrix_b_diff = mat_mat_sub(Wb, initial_b_matrix, q, (m+1));
    // prrint("\n Wa matrix after:\n");
    // print_mat(Wa, q, q);
    // prrint("\n Wa diff:\n");
    // print_mat(matrix_a_diff, q, q);
    // prrint("\n Wb diff:\n");
    // print_mat(matrix_b_diff, q, (m+1));

    simulate(1);
    return 0;
}

void train() {
    int p = 0;
    float H_sum = 0;
    unlink("energy.txt");
    FILE *file_e = fopen("energy.txt", "w");
    unlink("weights.txt");
    FILE *file_w = fopen("weights.txt", "w");


    for (int n=1; n<=n_max; n++) {

        if (p>=998) p = 0;
        xi[0] = I[p];
        xi[1] = I[p+1];
        zeta = I[p+2];
        p++;

        forward(n);
        H_sum += h();
        fprintf(file_e, "%d\t%f\n", n, (H_sum/n));
        backward(n);
        next();

        if (n%100==0) {
            // Print weights to file
            for (int i=0;i<q;i++)
                for (int j=0;j<q; j++)
                    fprintf(file_w, "   %f,", Wa[i][j]);
            for (int i=0;i<q;i++)
                for (int j=0;j<(m+1); j++)
                    fprintf(file_w, "%f,", Wb[i][j]);
            for (int i=0;i<q;i++)
                fprintf(file_w, "%f%c", Wc[i], ((i+1)==q? '\n' : ','));
        }

    }
    printf("Energy: %f, last output: %f\n", (H_sum/n_max), get_O()); fflush(stdout);

    // system("pause");
    fclose(file_e);
    fclose(file_w);
}

void simulate(int k_max) {

    char file_name[20];
    sprintf(file_name, "output_k%d.txt", k_max);
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
        fprintf(file, "%f\n", get_O());
        output[k+2] = O;
    }
    fclose(file);
}

void forward(int n) {

    V_next = new_vec_zero(q);
    if (V_next == NULL) errror("ERROR: V_next malloc failed");
    // Calculate hidden states
    for (int j=0;j<q;j++) {
        V_next[j] = g(vec_vec_mult(Wa[j], V, q) + vec_vec_mult(Wb[j], xi, (m+1)));
    }
    // Calculate output state
    O = vec_vec_mult(Wc, V, q);

}

void backward(int n) {
    // Perform concatenations of weights and states
    float **wjs = malloc(q * sizeof *wjs);
    if (wjs == NULL) errror("ERROR: wjs malloc failed");
    for(int j=0;j<q;j++) wjs[j] = vec_vec_concat(Wa[j], Wb[j], q, (m+1));
    float *u = vec_vec_concat(V, xi, q, (m+1));
    // Use vector instead of diagonal matrix
    float *phi_vec = new_vec(q);
    if (phi_vec == NULL) errror("ERROR: phi_vec malloc failed");
    for(int j=0; j<q; j++) phi_vec[j] = g_prime(vec_vec_mult(wjs[j], u, (q+m+1)));

    // Update all neurons
    for (int j=0; j<q; j++) {
        update_lambda(j, phi_vec, u);
        // Calculate error
        float E = zeta - O;
        // Calculate delta-weights
        float *soon_enough = vec_mat_mult(Wc, lambda[j], q, (q+m+1));
        float *almost_done = vec_scalar_mult(soon_enough, eta, (q+m+1));
        float *delta_wjs = vec_scalar_mult(almost_done, E, (q+m+1));

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
    // Finally, clean up
    free_mat(wjs, q);
    free(u);
}

void next() {

    // "Go to" next state
    free(V);
    V = V_next;
    for (int j=0;j<q;j++) {
        // printf("lambda[%d]: %p\n", j, lambda[j]); fflush(stdout);
        free_mat(lambda[j], q);
        lambda[j] = lambda_next[j];
    }
}

void update_lambda(int j, float *phi_vec, float *u) {
    // Calculate new lambda
    float **product = mat_mat_mult(Wa, lambda[j], q, q, (q+m+1));

    // Add u to only j'th row
    float *jth_row = vec_vec_add(product[j], u, (q+m+1));
    free(product[j]);
    product[j] = jth_row;

    // "phi matrix multiplication"
    for (int a=0;a<q; a++) // row
        for(int b=0; b<(q+m+1); b++) // col
            product[a][b] *= phi_vec[a];

    lambda_next[j] = product;
}

float g(float x) {
    return tanh(x);
}
float g_prime(float x) {
    return (1-pow(tanh(x), 2));
}
float h() {
    float E = get_zeta()-get_O();
    return (pow(E, 2) / 2);
}
float get_O() { // Returns unnormalised O value
    return normalise_data ? (validation_mode? (O*sdev_v + mean_v) : (O*sdev + mean)) : O;
}
float get_zeta() { // Returns unnormalised O value
    return normalise_data ? (validation_mode? (zeta*sdev_v + mean_v) : (zeta*sdev + mean)) : zeta;
}

// Initialise program
void init() {
    srand(time(NULL));
    rand();

    if (normalise_data) {
        I = validation_mode? read_patterns(p_v, "laser_cont.norm.txt") : read_patterns(p_t, "laser.norm.txt");
    } else {
        I = validation_mode? read_patterns(p_v, "laser_cont.txt") : read_patterns(p_t, "laser.txt");
    }

    xi = malloc((m+1) * sizeof *xi);
    if (xi == NULL) errror("ERROR: xi malloc failed");
    xi[2] = 1; // Add hax node, for threshold
    V = new_vec_zero(q);
    if (V == NULL) errror("ERROR: V malloc failed");

    Wa = generate_weight_matrix(q,q);
    Wb = generate_weight_matrix(q,(m+1));
    Wc = generate_weight_vector(q);

    lambda = malloc(q * sizeof *lambda);
    if (lambda == NULL) errror("ERROR: lambda malloc failed");
    lambda_next = malloc(q * sizeof *lambda_next);
    if (lambda_next == NULL) errror("ERROR: lambda_next malloc failed");

    for (int j=0; j<q; j++) {
        lambda[j] = new_mat_zero(q, (q+m+1));
        if (lambda[j] == NULL) errror("ERROR: lambda[j] malloc failed");
        lambda_next[j] = new_mat_zero(q, (q+m+1));
        if (lambda_next[j] == NULL) errror("ERROR: lambda_next[j] malloc failed");
    }

}

float **generate_weight_matrix(int m, int n) {
    float **weights = malloc(m * sizeof *weights);
    if (weights == NULL) errror("ERROR: weights mat malloc failed");
    for (int i=0; i<m; i++) {
        weights[i] = malloc(n * sizeof *weights[i]);
        if (weights[i] == NULL) errror("ERROR: weights[i] mat malloc failed");
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
    if (weights == NULL) errror("ERROR: weights vec malloc failed");
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
    if (pat == NULL) {}
    for(int i=0;i<p;i++) {
        fscanf(f, "%f", &pat[i]);
    }
    fclose(f);
    return pat;
}
