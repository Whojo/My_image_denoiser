#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <math.h>

#include "Eigen/src/Core/util/Constants.h"
#include "fft.hh"
#include "stb_image.h"
#include "stb_image_write.h"

using Eigen::Matrix;
using MatrixC = Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>;
using Eigen::ArrayXf;
using Eigen::Map;

void rgb_to_gray(std::uint8_t *rgb, double *coeffs, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        double r = rgb[3 * i];
        double g = rgb[3 * i + 1];
        double b = rgb[3 * i + 2];

        coeffs[i] = (r + g + b) / 3.0;
    }
}

Eigen::ArrayXXd get_filter_base(size_t win_size)
{
    size_t half = win_size / 2;

    Eigen::ArrayXXd base(win_size, win_size);

    for (size_t i = 0; i < win_size; i++)
        for (size_t j = 0; j < win_size; j++)
            base(i, j) = (i - half) * (i - half) + (j - half) * (j - half);

    return base;
}

void compute_gauss_notch_response(const Eigen::ArrayXXd &base,
                                  Eigen::ArrayXXd &filter, double A, double B)
{
    filter = 1 - A * (-B * base).exp();
}

void fft_shift(std::complex<double> *ft_coeffs, size_t height, size_t width)
{
    Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>>
        mat_coeffs(ft_coeffs, height, width);

    size_t offset_w = (width + 1) / 2;
    size_t offset_h = (height + 1) / 2;

    for (size_t i = 0; i < height; i++)
    {
        auto row = mat_coeffs.row(i);
        std::rotate(row.begin(), row.begin() + offset_w, row.end());
    }

    for (size_t i = 0; i < width; i++)
    {
        auto col = mat_coeffs.col(i);
        std::rotate(col.begin(), col.begin() + offset_h, col.end());
    }
}

void ifft_shift(std::complex<double> *ft_coeffs, size_t height, size_t width)
{
    Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>>
        mat_coeffs(ft_coeffs, height, width);

    size_t offset_w = width / 2;
    size_t offset_h = height / 2;

    for (size_t i = 0; i < height; i++)
    {
        auto row = mat_coeffs.row(i);
        std::rotate(row.begin(), row.begin() + offset_w, row.end());
    }

    for (size_t i = 0; i < width; i++)
    {
        auto col = mat_coeffs.col(i);
        std::rotate(col.begin(), col.begin() + offset_h, col.end());
    }
}

void correct_spikes(std::complex<double> *ft_coeffs, size_t height,
                    size_t width, double theta, size_t win_size, size_t radius)
{
    size_t half = win_size / 2;
    double radius_sqr = radius * radius;
    Eigen::ArrayXXd base = get_filter_base(win_size);
    Eigen::ArrayXXd filter(win_size, win_size);

    Map<Eigen::Array<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>>
        mat_coeffs(ft_coeffs, height, width);

    double w = (static_cast<double>(width) - 1) / 2.0;
    double h = (static_cast<double>(height) - 1) / 2.0;

    for (size_t i = half; i < height - half - 1; i++)
    {
        for (size_t j = half; j < width - half - 1; j++)
        {
            if ((i - h) * (i - h) + (j - w) * (j - w) < radius_sqr)
                continue;

            auto block_magn =
                mat_coeffs.block(i - half, j - half, win_size, win_size).abs();

            if ((std::abs(mat_coeffs(i, j)) / block_magn.mean()) > theta)
            {
                auto min = block_magn.minCoeff();
                compute_gauss_notch_response(base, filter, 0.9,
                                             12.5 / (min * min));
                mat_coeffs.block(i - half, j - half, win_size, win_size) *=
                    filter;
            }
        }
    }
}

void complex_to_gray(std::complex<double> *ift_coeffs, std::uint8_t *rgb_gray,
                     size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        std::uint8_t val = std::clamp(ift_coeffs[i].real(), 0.0, 255.0);

        rgb_gray[3 * i] = val;
        rgb_gray[3 * i + 1] = val;
        rgb_gray[3 * i + 2] = val;
    }
}

void fft_log_image(std::complex<double> *ft_coeffs, std::uint8_t *rgb_gray,
                   size_t size)
{
    double *magnitudes = new double[size];

    double max = 0.0;

    for (size_t i = 0; i < size; i++)
    {
        double magn = log(std::abs(ft_coeffs[i]));
        max = std::max(magn, max);
        magnitudes[i] = magn;
    }

    for (size_t i = 0; i < size; i++)
    {
        std::uint8_t val =
            static_cast<std::uint8_t>(255.0 * magnitudes[i] / max);

        rgb_gray[3 * i] = val;
        rgb_gray[3 * i + 1] = val;
        rgb_gray[3 * i + 2] = val;
    }

    delete[] magnitudes;
}
void fft_log_image_corrected(std::complex<double> *ft_coeffs,
                             std::uint8_t *rgb_gray, size_t size)
{
    double *magnitudes = new double[size];

    double max = 0.0;

    for (size_t i = 0; i < size; i++)
    {
        double magn = log(std::abs(ft_coeffs[i]));
        max = std::max(magn, max);
        magnitudes[i] = magn;
    }

    for (size_t i = 0; i < size; i++)
    {
        std::uint8_t val =
            static_cast<std::uint8_t>(255.0 * magnitudes[i] / max);

        rgb_gray[3 * i] = val;
        rgb_gray[3 * i + 1] = val;
        rgb_gray[3 * i + 2] = val;
    }

    delete[] magnitudes;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
        return 1;

    int width, height, num_chan;

    std::uint8_t *img_rgb = stbi_load(argv[1], &width, &height, &num_chan, 3);

    size_t size = width * height;

    double *gray = new double[size];
    std::complex<double> *ft_coeffs = new std::complex<double>[size];
    std::complex<double> *ift_coeffs = new std::complex<double>[size];

    rgb_to_gray(img_rgb, gray, size);

    fft2(gray, ft_coeffs, height, width);

    fft_shift(ft_coeffs, height, width);

    correct_spikes(ft_coeffs, height, width, 5, 11, 10);

    ifft_shift(ft_coeffs, height, width);

    ifft2(ft_coeffs, ift_coeffs, height, width);

    complex_to_gray(ift_coeffs, img_rgb, size);

    stbi_write_png("denoised.png", width, height, 3, img_rgb, 3 * width);

    stbi_image_free(img_rgb);
    delete[] gray;

    return 0;
}
