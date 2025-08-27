#ifndef MATRIX_H
#define MATRIX_H

#include "std_includes.h"

/*
Base data type that holds most coordinate info.
*/
class Matrix {
    size_t rows;
    size_t cols;

public:
    // ctor
    Matrix(size_t r, size_t c);
    Matrix(std::initializer_list<float> vec);
    Matrix(std::initializer_list<std::initializer_list<float>> mat);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(Matrix&& other);
    // dtor
    ~Matrix();

    // utility
    float& at(size_t r, size_t c) const;
    void print();

    // operators
    float* operator[](size_t r) const;
    Matrix operator+(const Matrix& other);
    Matrix operator-(const Matrix& other);
    Matrix operator*(const float num);
    Matrix operator*(const Matrix& other);
    
    float* value_ptr() const {
        return m;
    }

    inline float& vx() { return this->at(0, 0); }
    inline float& vy() { return this->at(1, 0); }
    inline float& vz() { return this->at(2, 0); }
    inline float& vw() { return this->at(3, 0); }

protected:
    float *m; // pointer to 2d array in a single block m[i * cols + j]
};

using Vec2 = Matrix;
using Vec3 = Matrix;
using Vec4 = Matrix;


inline Matrix Ortho(float left, float right, float bottom, float top, float nearVal, float farVal) {
    Matrix result(4, 4); // Assuming Matrix constructor initializes to identity or zeros.
                         // If it doesn't, ensure it's identity first.

    // Pure scaling components
    result[0][0] = 2.0f / (right - left);
    result[1][1] = 2.0f / (top - bottom);
    result[2][2] = -2.0f / (farVal - nearVal); // Note the negative for Z as typically OpenGL uses Z-mapping to [-1,1]

    // Ensure all other elements are zero, except for result[3][3] which should be 1.0 for perspective division (though not strictly necessary for pure scaling)
    result[0][1] = result[0][2] = result[0][3] = 0.0f;
    result[1][0] = result[1][2] = result[1][3] = 0.0f;
    result[2][0] = result[2][1] = result[2][3] = 0.0f;
    result[3][0] = result[3][1] = result[3][2] = 0.0f;
    result[3][3] = 1.0f; // Essential for correct homogeneous coordinates

    return result;
}

#endif  //MATRIX_H