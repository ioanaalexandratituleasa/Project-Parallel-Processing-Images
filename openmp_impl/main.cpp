#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>
#include <iomanip>
#include <omp.h>

struct Image {
    int width, height;
    std::vector<unsigned char> data;
    Image(int w, int h) : width(w), height(h), data(w* h, 0) {}
    unsigned char& at(int row, int col) { return data[row * width + col]; }
    unsigned char  at(int row, int col) const { return data[row * width + col]; }
};

Image load_image(const char* path) {
    int w, h, channels;
    unsigned char* pixels = stbi_load(path, &w, &h, &channels, 1); // 1 = greyscale
    if (!pixels) {
        std::cerr << "Eroare la incarcarea imaginii: " << path << "\n";
        exit(1);
    }
    Image img(w, h);
    memcpy(img.data.data(), pixels, w * h);
    stbi_image_free(pixels);
    std::cout << "Imagine incarcata: " << w << "x" << h << " px\n";
    return img;
}

static inline unsigned char median_of_window(const Image& src, int row, int col, int half) {
    unsigned char window[625];
    int count = 0;
    int r0 = std::max(0, row - half), r1 = std::min(src.height - 1, row + half);
    int c0 = std::max(0, col - half), c1 = std::min(src.width - 1, col + half);
    for (int r = r0; r <= r1; ++r)
        for (int c = c0; c <= c1; ++c)
            window[count++] = src.at(r, c);
    std::nth_element(window, window + count / 2, window + count);
    return window[count / 2];
}

Image median_filter_sequential(const Image& src, int kernel_size) {
    Image dst(src.width, src.height);
    int half = kernel_size / 2;
    for (int r = 0; r < src.height; ++r)
        for (int c = 0; c < src.width; ++c)
            dst.at(r, c) = median_of_window(src, r, c, half);
    return dst;
}

Image median_filter_openmp(const Image& src, int kernel_size, int num_threads) {
    Image dst(src.width, src.height);
    int half = kernel_size / 2;

    omp_set_num_threads(num_threads);

#pragma omp parallel for schedule(static)
    for (int r = 0; r < src.height; ++r)
        for (int c = 0; c < src.width; ++c)
            dst.at(r, c) = median_of_window(src, r, c, half);

    return dst;
}

using Clock = std::chrono::high_resolution_clock;
template<typename Fn>
double measure_ms(Fn&& fn, int repetitions = 3) {
    double total = 0;
    for (int i = 0; i < repetitions; ++i) {
        auto t0 = Clock::now(); fn();
        auto t1 = Clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / repetitions;
}

void run_benchmark(const Image& src) {
    int max_threads = omp_get_max_threads();
    const std::vector<int> kernels = { 3, 5, 7, 11, 15 };
    const std::vector<int> thread_counts = { 1, 2, 4, 8, 16 };

    // ── Tabel 1: speedup vs kernel ──
    std::cout << "\n--- Speedup vs Kernel Size (threads=" << max_threads << ") ---\n";
    std::cout << std::setw(8) << "Kernel"
        << std::setw(14) << "Sequential"
        << std::setw(14) << "OpenMP"
        << std::setw(10) << "Speedup" << "\n"
        << std::string(46, '-') << "\n";

    for (int k : kernels) {
        double t_seq = measure_ms([&] { median_filter_sequential(src, k); });
        double t_omp = measure_ms([&] { median_filter_openmp(src, k, max_threads); });
        std::cout << std::setw(8) << k
            << std::setw(13) << std::fixed << std::setprecision(2) << t_seq << "ms"
            << std::setw(13) << t_omp << "ms"
            << std::setw(9) << t_seq / t_omp << "x\n";
    }

    // ── Tabel 2: scalabilitate vs threaduri ──
    std::cout << "\n--- Scalabilitate vs Thread Count (kernel=7) ---\n";
    std::cout << std::setw(10) << "Threads"
        << std::setw(14) << "OpenMP"
        << std::setw(10) << "Speedup"
        << std::setw(12) << "Efficiency" << "\n"
        << std::string(46, '-') << "\n";

    double t_seq_base = measure_ms([&] { median_filter_sequential(src, 7); });
    for (int nt : thread_counts) {
        if (nt > omp_get_max_threads()) break;
        double t_omp = measure_ms([&] { median_filter_openmp(src, 7, nt); });
        double su = t_seq_base / t_omp;
        std::cout << std::setw(10) << nt
            << std::setw(13) << t_omp << "ms"
            << std::setw(9) << su << "x"
            << std::setw(10) << (su / nt * 100) << "%\n";
    }
}

int main() {
    const char* path = "C:\\Users\\Elena\\Desktop\\UNI\\Facultate\\an 3 sem 2\\PARADIS\\PROIECT-PARADIS-4\\mri\\brain_tumor_dataset\\yes\\Y1.jpg";
    Image src = load_image(path);
    Image big(1024, 1024);
    for (int r = 0; r < 1024; ++r)
        for (int c = 0; c < 1024; ++c)
            big.at(r, c) = src.at(r * src.height / 1024, c * src.width / 1024);
    src = big;

    std::cout << "Imagine redimensionata: 1024x1024\n";
    std::cout << "Threaduri disponibile: " << omp_get_max_threads() << "\n";
    run_benchmark(src);
    return 0;
}