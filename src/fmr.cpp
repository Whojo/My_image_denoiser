#include <Eigen/Dense>
#include <complex>
#include <iostream>

#include "fft.hh"

using Eigen::Matrix;
using MatrixC = Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>;
using Eigen::Map;

int main(int argc, char *argv[])
{
    double coeffs[] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2,
                        2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4 };

    std::complex<double> coeffs_ft[50];
    std::complex<double> coeffs_ift[50];

    fft2(coeffs, coeffs_ft, 5, 5);

    std::cout << Map<MatrixC>(coeffs_ft, 5, 5) << std::endl;

    ifft2(coeffs_ft, coeffs_ift, 5, 5);

    std::cout << Map<MatrixC>(coeffs_ift, 5, 5) << std::endl;

    return 0;
}
