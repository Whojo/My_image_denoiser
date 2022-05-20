#include "fft.hh"

#include <Eigen/Dense>
#include <algorithm>
#include <complex>
#include <iostream>
#include <numbers>
#include <vector>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::RowVectorXcd;
using Eigen::Map;

auto pi = std::numbers::pi_v<double>;

void copy_with_stride(std::uint8_t *src, double *dst, size_t size, int offset,
                      unsigned stride)
{
    for (size_t i = 0; i < size; i++)
        dst[i] = src[offset + i * stride];
}

void fft2(double *coeffs, std::complex<double> *coeffs_ft, size_t height,
          size_t width)
{
    Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        mat_coeffs(coeffs, height, width);

    Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>>
        mat_coeffs_ft(coeffs_ft, height, width);

    MatrixXcd tmp_mat_coeffs_ft = mat_coeffs_ft;

    auto factors_width = RowVectorXcd(width);
    auto factors_height = RowVectorXcd(height);

    for (int k = 0; k < width; k++)
        factors_width(k) = std::polar<double>(1, -2.0f * pi * k / width);

    for (int k = 0; k < height; k++)
        factors_height(k) = std::polar<double>(1, -2.0f * pi * k / height);

    for (size_t i = 0; i < height; i++)
    {
        for (size_t k = 0; k < width; k++)
        {
            auto &coeff_ft = tmp_mat_coeffs_ft(i, k);

            coeff_ft.imag(0.f);
            coeff_ft.real(0.f);

            for (size_t j = 0; j < width; j++)
                coeff_ft += mat_coeffs(i, j) * factors_width((j * k) % width);
        }
    }

    for (size_t i = 0; i < width; i++)
    {
        for (size_t k = 0; k < height; k++)
        {
            auto &coeff_ft = mat_coeffs_ft(k, i);

            coeff_ft.imag(0.f);
            coeff_ft.real(0.f);

            for (size_t j = 0; j < height; j++)
                coeff_ft +=
                    tmp_mat_coeffs_ft(j, i) * factors_height((j * k) % height);
        }
    }
}

void ifft2(std::complex<double> *coeffs, std::complex<double> *coeffs_ift,
           size_t height, size_t width)
{
    Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                            Eigen::Dynamic, Eigen::RowMajor>>
        mat_coeffs(coeffs, height, width);

    Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>>
        mat_coeffs_ft(coeffs_ift, height, width);

    MatrixXcd tmp_mat_coeffs_ft = mat_coeffs_ft;

    auto factors_width = RowVectorXcd(width);
    auto factors_height = RowVectorXcd(height);

    for (int k = 0; k < width; k++)
        factors_width(k) = std::polar<double>(1, 2.0f * pi * k / width);

    for (int k = 0; k < height; k++)
        factors_height(k) = std::polar<double>(1, 2.0f * pi * k / height);

    for (size_t i = 0; i < height; i++)
    {
        for (size_t k = 0; k < width; k++)
        {
            auto &coeff_ft = tmp_mat_coeffs_ft(i, k);

            coeff_ft.imag(0.f);
            coeff_ft.real(0.f);

            for (size_t j = 0; j < width; j++)
                coeff_ft += mat_coeffs(i, j) * factors_width((j * k) % width);

            coeff_ft /= width;
        }
    }

    for (size_t i = 0; i < width; i++)
    {
        for (size_t k = 0; k < height; k++)
        {
            auto &coeff_ft = mat_coeffs_ft(k, i);

            for (size_t j = 0; j < height; j++)
                coeff_ft +=
                    tmp_mat_coeffs_ft(j, i) * factors_height((j * k) % height);

            coeff_ft /= height;
        }
    }
}

void fft2_rgb(std::uint8_t *img_rgb, std::complex<double> *r_coeffs,
              std::complex<double> *g_coeffs, std::complex<double> *b_coeffs,
              size_t height, size_t width)
{
    size_t size = height * width;
    auto *comp_coeffs = new double[size];

    // r component
    copy_with_stride(img_rgb, comp_coeffs, size, 0, 3);
    fft2(comp_coeffs, r_coeffs, height, width);

    // g component
    copy_with_stride(img_rgb, comp_coeffs, size, 1, 3);
    fft2(comp_coeffs, g_coeffs, height, width);

    // b component
    copy_with_stride(img_rgb, comp_coeffs, size, 2, 3);
    fft2(comp_coeffs, b_coeffs, height, width);

    delete[] comp_coeffs;
}
