#include <stdio.h>
#include "matrix.h"

int main() {

	int **a = new_mat(10, 10);

	printf("a: %p\n", a);
	printf("&a: %p\n", &a);
	printf("a[0]: %p\n", a[0]);
	printf("&a[0]: %p\n", &a[0]);

	for (int i=0;i<10;i++) {
		free(a[i]);
	} free(a);


	printf("a: %p\n", a);
	printf("&a: %p\n", &a);
	printf("a[0]: %p\n", a[0]);
	printf("&a[0]: %p\n", &a[0]);
	


	return 0;
}