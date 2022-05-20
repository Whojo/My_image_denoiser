#include <complex>
#include <cstddef>
#include <cstdint>

void fft2(double *coeffs, std::complex<double> *coeffs_ft, size_t height,
          size_t width);

void ifft2(std::complex<double> *coeffs, std::complex<double> *coeffs_ift,
           size_t height, size_t width);

void fft2_rgb(std::uint8_t *img_rgb, std::complex<double> *r_coeffs,
              std::complex<double> *g_coeffs, std::complex<double> *b_coeffs,
              size_t height, size_t width);
