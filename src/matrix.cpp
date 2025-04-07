#include "matrix.h"

Matrix::Matrix() : rows(0), cols(0), m_data() {}

Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), m_data(r, std::vector<float>(c, 0)) {}

// constructor for column vector
Matrix::Matrix(std::initializer_list<float> vec) : rows(vec.size()), cols(1) {
    for (auto &row_i : vec) {
        m_data.push_back({row_i});
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> mat) : rows(mat.size()), cols(mat.begin() != mat.end() ? (*mat.begin()).size() : 0){
    for (auto &row_i : mat) { 
        if (row_i.size() != cols) {
            std::cerr << "Error: invalid matrix format" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        m_data.push_back(row_i);
    }
}

/* derived constructors */
Matrix::Matrix(int r, int c, std::initializer_list<float> vec) : rows(r), cols(c) {
    m_data.push_back(vec);
}

Matrix::Matrix(int r, int c, std::initializer_list<std::initializer_list<float>> mat) : rows(r), cols(c) {
    for (auto &row_i : mat) { 
        if (row_i.size() != cols) {
            std::cerr << "Error: invalid matrix format" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        m_data.push_back(row_i);
    }
}

/* utility */
float& Matrix::at(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        std::cerr << "Error: matrix element access out of range" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return const_cast<float&>(m_data[r][c]);
}

void Matrix::print() {
    for (int ri=0; ri<rows; ri++) {
        for (int ci=0; ci<cols; ci++) {
            std::cout << m_data[ri][ci] << " ";
        }
        std::cout << std::endl;
    }
}

/* math */
Matrix Matrix::operator+(const Matrix& other) {
    if (!(this->rows == other.rows && this->cols == other.cols)) {
        std::cerr << "Error: dimension mismatch" << "{" << this->rows << "," << this->cols << "}"
            << "versus" << "{" << other.rows << "," << other.cols << "}" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    Matrix result = Matrix(rows, cols);
    for (int r=0; r<rows; r++) {
        for (int c=0; c<cols; c++) {
            result.at(r, c) = this->at(r, c) + other.at(r, c);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) {
    if (!(this->rows == other.rows && this->cols == other.cols)) {
        std::cerr << "Error: dimension mismatch" << "{" << this->rows << "," << this->cols << "}"
            << "versus" << "{" << other.rows << "," << other.cols << "}" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    Matrix result = Matrix(rows, cols);
    for (int r=0; r<rows; r++) {
        for (int c=0; c<cols; c++) {
            result.at(r, c) = this->at(r, c) - other.at(r, c);
        }
    }
    return result;
}

Matrix Matrix::operator*(const float num) {
    Matrix result = Matrix(rows, cols);
    for (int r=0; r<rows; r++) {
        for (int c=0; c<cols; c++) {
            result.at(r, c) = this->at(r, c) * num;
        }
    }
    return result;
}

// matrix multiplication
Matrix  Matrix::operator*(const Matrix& other) {
    if (this->cols != other.rows) {
        std::cerr << "Error: matrix multiplication dimension mismatch" << "{" << this->rows << "," << this->cols << "}"
            << "versus" << "{" << other.rows << "," << other.cols << "}" << std::endl;
    }

    int middle = this->cols;

    // A x B = result dim = (A.row, B.col)
    Matrix result = Matrix(this->rows, other.cols);
    for (int r=0; r < this->rows; r++) {
        for (int c=0; c < other.cols; c++) {
            float sum = 0;
            for (int i=0; i < middle; i++) {
                sum += this->at(r, i) * other.at(i, c);
            }
            result.at(r, c) = sum;
        }
    }

    return result;
}