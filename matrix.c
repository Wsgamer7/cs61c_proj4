#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
 */

/*
 * Generates a random double between `low` and `high`.
 */
double rand_double(double low, double high)
{
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/*
 * Generates a random matrix with `seed`.
 */
void rand_matrix(matrix *result, unsigned int seed, double low, double high)
{
    srand(seed);
    for (int i = 0; i < result -> rows; i++)
    {
        for (int j = 0; j < result -> cols; j++)
        {
            set(result, i, j, rand_double(low, high));
        }
    }
}
/*
 *Trans 2D position to 1D
 */
int to1D(int row, int col, int rows)
{
    return row + col * rows;
}

int allocate_matrix_raw(matrix **mat, int rows, int cols) {
    if (rows <= 0 || cols <= 0)
    {
        return -1;
    }
    matrix *mat_got = (matrix *)malloc(sizeof(matrix *));
    if (mat_got == NULL)
    {
        return -1;
    }
    (*mat) = mat_got;
    mat_got -> rows = rows;
    mat_got -> cols = cols;
    mat_got -> ref_cnt = 0;
    mat_got -> parent = NULL;
    mat_got -> is_1d = 0;
    if (rows == 1 && cols == 1)
    {
        mat_got -> is_1d = 1;
    }
    /*allocate for data*/
    int data_length = rows * cols;
    mat_got -> data = (double **)malloc(data_length * sizeof(double *));
    if (mat_got -> data == NULL)
    {
        return -1;
    }
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. Remember to set all fieds of the matrix struct.
 * `parent` should be set to NULL to indicate that this matrix is not a slice.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. <<??????>>If you don't set python error messages here upon
 * failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix(matrix **mat, int rows, int cols)
{
    /* TODO: YOUR CODE HERE */
    int allocate_pass = allocate_matrix_raw(mat, rows, cols);
    if (allocate_pass != 0) {
        return -1;
    }

    // inti all element in matrix to 0
    int data_length = rows * cols;
    for (int i = 0; i < data_length; i++)
    {
        double *a_num = (double *)malloc(sizeof(double));
        if (a_num == NULL)
        {
            return -1;
        }
        *a_num = 0;
        *(((*mat) -> data) + i) = a_num;
    }
    return 0;
}

/*
 * Allocate space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * This is equivalent to setting the new matrix to be
 * from[row_offset:row_offset + rows, col_offset:col_offset + cols]
 * If you don't set python error messages here upon failure, then remember to set it in numc.c.
 * Return 0 upon success and non-zero upon failure.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int row_offset, int col_offset,
                        int rows, int cols)
{
    /* TODO: YOUR CODE HERE */
    if (row_offset < 0 || col_offset < 0 ||
        (row_offset >= from -> rows) || (col_offset >= from -> cols))
    {
        return -1;
    }
    int allocate_pass = allocate_matrix_raw(mat, rows, cols);
    if (allocate_pass != 0) {
        return -1;
    }
    matrix* mat_got = *mat;
    mat_got -> ref_cnt = 1;
    from -> ref_cnt += 1;

    if ((row_offset + rows <= from -> rows) && (col_offset+ cols <= from -> cols)) {
        mat_got -> parent = *mat;
    }

    // ref share element or inti to 0
    for (int col = 0; col < cols; col++)
    {
        for (int row = 0; row < rows; row++)
        {
            // ref to from
            int to_1d = to1D(row, col, rows);
            if ((col < from -> cols - col_offset) && (row < from -> rows - row_offset))
            {
                int from_1d = to1D(row + row_offset, col + col_offset, from -> rows);
                *((mat_got -> data) + to_1d) = *((from -> data) + from_1d);
                continue;
            }

            // inti to 0
            double *a_num = (double *)malloc(sizeof(double));
            if (a_num == NULL)
            {
                return -1;
            }
            *a_num = 0;
            *((mat_got -> data) + to_1d) = a_num;
        }
    }
    return 0;
}

/*
 * This function will be called automatically by Python when a numc matrix loses all of its
 * reference pointers.
 * You need to make sure that you only free `mat -> data` if no other existing matrices are also
 * referring this data array.
 * See the spec for more information.
 */
void deallocate_matrix(matrix *mat)
{
    /* TODO: YOUR CODE HERE */
    if (mat -> ref_cnt == 0) {
        int data_length = (mat -> cols) * (mat -> rows);
        for (int i = 0; i < data_length; i++) {
            free(*(mat -> data + i));
        }
        free(mat -> data);
    }
    /* TODO: free the matrix when all ref_matrices are freed*/
}

/*
 * Return the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col)
{
    /* TODO: YOUR CODE HERE */
    int index = to1D(row, col, mat -> rows);
    return **(mat -> data + index);
}

/*
 * Set the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val)
{
    /* TODO: YOUR CODE HERE */
    int index = to1D(row, col, mat -> rows);
    **(mat -> data + index) = val;
}

/*
 * Set all entries in mat to val
 */
void fill_matrix(matrix *mat, double val)
{
    /* TODO: YOUR CODE HERE */
    double *value = (double*) malloc(sizeof(double));
    *value = val;
    int data_length = (mat -> cols) * (mat -> rows);
    for (int i = 0; i < data_length; i++) {
        *(mat -> data + i) = value;
    }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    /* TODO: YOUR CODE HERE */
    if ((mat1 -> cols != mat2 -> cols) || (mat1 -> rows != mat2 -> rows)) {
        return -1;
    }
    int rows = mat1 -> rows;
    int cols = mat1 -> cols;

    int data_length = rows * cols;
    for (int i = 0; i < data_length; i++)
    {
        double *i_num = (double *)malloc(sizeof(double));
        if (i_num == NULL)
        {
            return -1;
        }
        *i_num = **(mat1 -> data + i) + **(mat2 -> data + i);
        *((result -> data) + i) = i_num;
    }
    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    /* TODO: YOUR CODE HERE */
    if ((mat1 -> cols != mat2 -> cols) || (mat1 -> rows != mat2 -> rows)) {
        return -1;
    }
    int rows = mat1 -> rows;
    int cols = mat1 -> cols;

    int data_length = rows * cols;
    for (int i = 0; i < data_length; i++)
    {
        double *i_num = (double *)malloc(sizeof(double));
        if (i_num == NULL)
        {
            return -1;
        }
        *i_num = **(mat1 -> data + i) - **(mat2 -> data + i);
        *((result -> data) + i) = i_num;
    }
    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2)
{
    /* TODO: YOUR CODE HERE */
    if (mat1 -> cols != mat2 -> rows) {
        return -1;
    }
    int rows = mat1 -> rows;
    int cols = mat2 -> cols; 

    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            set(result, row, col, 0);
            for (int k = 0; k < mat1 -> cols; k++) {
                double value = get(result, row, col) + get(mat1, row, k) * get(mat2, k, col);
                set(result, row, col, value);
            }
        }
    }
    return 0;
}
int cp_matrix(matrix *result, matrix *from) {
    if ((result -> cols != from -> cols) || (result -> rows != from -> rows)) {
        return -1;
    }
    int data_length = result -> cols * result -> rows;
    for (int i = 0; i < data_length; i++)
    {
        *(result -> data + i) = *(from -> data + i);
    }
    result -> parent = from;
    result -> ref_cnt = 1;
    from -> ref_cnt += 1;
    return 0;
}
/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow)
{
    /* TODO: YOUR CODE HERE */
    if (pow <= 0) {
        return -1;
    }
    if (pow == 1) {
        cp_matrix(result, mat);
    }
    if (pow % 2 == 0) {
        matrix* last_result = (matrix*) malloc(sizeof(matrix));
        int last_pow_pass = pow_matrix(last_result, mat, pow / 2);
        if (last_pow_pass != 0) {
            return -1;
        }
        int mul_pass = mul_matrix(result, last_result, last_result);
        deallocate_matrix(last_result);
    } else {
        matrix* last_result = (matrix*) malloc(sizeof(matrix));
        int last_pow_pass = pow_matrix(last_result, mat, pow - 1);
        if (last_pow_pass != 0) {
            return -1;
        }
        int mul_pass = mul_matrix(result, last_result, mat);
        deallocate_matrix(last_result);
    }
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat)
{
    /* TODO: YOUR CODE HERE */
    int data_length = result -> cols * result -> rows;
    for (int i = 0; i < data_length; i++)
    {
        **(result -> data + i) = - (**(mat -> data + i));
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat)
{
    /* TODO: YOUR CODE HERE */
    int data_length = result -> cols * result -> rows;
    for (int i = 0; i < data_length; i++)
    {
        **(result -> data + i) = abs(**(mat -> data + i));
    }
    return 0;
}
