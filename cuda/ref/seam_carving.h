/*
    seam_carving.h
    
    header file for seam carving
*/

#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

namespace cuda {

/**
 * @brief Gives the product of two square matrices (of equal dimensions)
 * @param sq_matrix_1 First square matrix
 * @param sq_matrix_2 Second square matrix
 * @param sq_matrix_result Pointer to store the resultand matrix
 * @param sq_dimension Dimension of the square matrix
 */
  void matrix_multiplication(float *sq_matrix_1, float *sq_matrix_2,
      float *sq_matrix_result, unsigned int sq_dimension);
}

#endif
