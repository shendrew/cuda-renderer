#ifndef MATRIX_H
#define MATRIX_H

#include "./std_includes.h"

/*
Base data type that holds most coordinate info.
*/
struct Matrix {
    const size_t rows;
    const size_t cols;

    // constructors
    Matrix();
    Matrix(size_t r, size_t c);
    Matrix(std::initializer_list<float> vec);
    Matrix(std::initializer_list<std::initializer_list<float>> mat);

    // utility
    float& at(size_t r, size_t c) const;
    void print();

    // operators
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
    Matrix operator*(const float num);
    Matrix operator*(const Matrix& other);

protected:
    std::vector<std::vector<float>> m_data;
    Matrix(int r, int c, std::initializer_list<float> vec);
    Matrix(int r, int c, std::initializer_list<std::initializer_list<float>> mat);
};

#define vx at(0, 0)
#define vy at(1, 0)
#define vz at(2, 0)
#define vw at(3, 0)

#endif  //MATRIX_H