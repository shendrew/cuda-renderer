#include "matrix.h"

Matrix::Matrix(size_t r, size_t c) : rows{r}, cols{c}, m{new float[r*c]} {}

// constructor for column vector
Matrix::Matrix(std::initializer_list<float> vec) : rows{vec.size()}, cols{1}, m{new float[rows]} {
    int i = 0;
    for (auto &row : vec) {
        m[i] = row;
        i++;
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> mat) : rows{mat.size()}, cols{mat.begin() != mat.end() ? mat.begin()->size() : 0}, m{new float[rows * cols]} {
    int i = 0;
    for (auto &row : mat) { 
        if (row.size() != cols) {
            std::cerr << "Error: invalid matrix format" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        int j = 0;
        for (auto &val: row) {
            m[i*cols + j] = val;
            j++;
        }
        i++;
    }
}

// copy
Matrix::Matrix(const Matrix& other) : rows{other.rows}, cols{other.cols}, m{new float[rows * cols]} {
    std::copy(other.m, other.m + (rows * cols), m);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this; // self-assignment check

    delete[] m;
    rows = other.rows;
    cols = other.cols;
    m = new float[rows * cols];
    std::copy(other.m, other.m + (rows * cols), m);
    return *this;
}

// move
Matrix::Matrix(Matrix&& other) : rows{other.rows}, cols{other.cols}, m{other.m} {
    other.m = nullptr; // avoid double deletion
}

Matrix& Matrix::operator=(Matrix&& other) {
    if (this == &other) return *this; // self-assignment check

    Matrix temp(other);
    std::swap(rows, temp.rows);
    std::swap(cols, temp.cols);
    std::swap(m, temp.m);
    return *this;
}

Matrix::~Matrix() {
    delete[] m;
}

/* utility */
float& Matrix::at(size_t r, size_t c) const {
    if (r >= rows || c >= cols) {
        std::cerr << "Error: matrix element access out of range" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return m[r * cols + c];
}

void Matrix::print() {
    for (int ri=0; ri<rows; ri++) {
        for (int ci=0; ci<cols; ci++) {
            std::cout << this->at(ri, ci) << " ";
        }
        std::cout << std::endl;
    }
}

// syntax sugar for accessing rows (potentially unsafe)
float* Matrix::operator[](size_t r) const {
    if (r >= rows) {
        std::cerr << "Error: matrix row access out of range" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return &m[r * cols];
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