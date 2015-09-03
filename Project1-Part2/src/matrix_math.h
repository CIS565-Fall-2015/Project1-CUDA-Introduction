
#define MATRIX_WIDTH (5)

void mat_add (float * A, float * B, float * C, int width);
void mat_sub (float * A, float * B, float * C, int width);
void mat_mul (float * A, float * B, float * C, int width);

void init();
void run();
void free();

void printMatrix (float * M, int width);