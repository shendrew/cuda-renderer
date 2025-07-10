#include "matrix.h"
#include "std_includes.h"

void testMatrix() {
    Matrix a = {{1, 2, 3}, {4, 5, 6}};

    Matrix b = {{1, 2}, {3, 4}, {5, 6}};

    // std::cout << "A: " << a.rows << " " << a.cols << std::endl;
    a.print();

    // std::cout << "B: " << b.rows << " " << b.cols << std::endl;
    b.print();

    Matrix c = a*b;

    c.print();

    Matrix* d = new Matrix(4, 1);
    d->print();
    // std::cout << d->rows << std::endl;

    delete d;

    std::cout << "---------" << std::endl;

    Vec2 v2 = {{1,2,3}, {3, 4,5 }};
    v2.print();
}

// int main() {
//     testMatrix();
// }