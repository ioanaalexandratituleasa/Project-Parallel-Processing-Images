#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>

//Median de fereastra

static inline unsigned char median_of_window(
    const unsigned char* data, int width, int height,
    int row, int col, int half)
{
    unsigned char window[625];
    int count = 0;
    int r0 = std::max(0, row - half), r1 = std::min(height - 1, row + half);
    int c0 = std::max(0, col - half), c1 = std::min(width - 1, col + half);
    for (int r = r0; r <= r1; ++r)
        for (int c = c0; c <= c1; ++c)
            window[count++] = data[r * width + c];
    std::nth_element(window, window + count / 2, window + count);
    return window[count / 2];
}

//Sequential baseline

using Clock = std::chrono::high_resolution_clock;

double run_sequential(const std::vector<unsigned char>& img,
    int width, int height, int kernel_size, int reps = 3)
{
    int half = kernel_size / 2;
    std::vector<unsigned char> out(width * height);
    double total = 0;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = Clock::now();
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                out[r * width + c] = median_of_window(
                    img.data(), width, height, r, c, half);
        auto t1 = Clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / reps;
}

//TBB parallel_for pe linii

double run_tbb_1d(const std::vector<unsigned char>& img,
    std::vector<unsigned char>& out,
    int width, int height, int kernel_size,
    int num_threads, int reps = 3)
{
    int half = kernel_size / 2;
    tbb::global_control gc(
        tbb::global_control::max_allowed_parallelism, num_threads);

    double total = 0;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = Clock::now();

        tbb::parallel_for(0, height, [&](int r) {
            for (int c = 0; c < width; ++c)
                out[r * width + c] = median_of_window(
                    img.data(), width, height, r, c, half);
            });

        auto t1 = Clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / reps;
}

//TBB parallel_for pe tile 2D

double run_tbb_2d(const std::vector<unsigned char>& img,
    std::vector<unsigned char>& out,
    int width, int height, int kernel_size,
    int num_threads, int reps = 3)
{
    int half = kernel_size / 2;
    tbb::global_control gc(
        tbb::global_control::max_allowed_parallelism, num_threads);

    double total = 0;
    for (int rep = 0; rep < reps; ++rep) {
        auto t0 = Clock::now();

        tbb::parallel_for(
            tbb::blocked_range2d<int>(0, height, 32, 0, width, 64),
            [&](const tbb::blocked_range2d<int>& range) {
                for (int r = range.rows().begin(); r < range.rows().end(); ++r)
                    for (int c = range.cols().begin(); c < range.cols().end(); ++c)
                        out[r * width + c] = median_of_window(
                            img.data(), width, height, r, c, half);
            }
        );

        auto t1 = Clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return total / reps;
}

//Main

int main() {
    const char* path = "C:\\Users\\Elena\\Desktop\\UNI\\Facultate\\an 3 sem 2\\PARADIS\\PROIECT-PARADIS-4\\mri\\brain_tumor_dataset\\yes\\Y1.jpg";

    int w, h, ch;
    unsigned char* px = stbi_load(path, &w, &h, &ch, 1);
    if (!px) { std::cerr << "Eroare la incarcarea imaginii!\n"; return 1; }

    int width = 1024, height = 1024;
    std::vector<unsigned char> img(width * height);
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
            img[r * width + c] = px[(r * h / height) * w + (c * w / width)];
    stbi_image_free(px);

    std::cout << "Imagine incarcata: " << w << "x" << h << " px\n";
    std::cout << "Imagine redimensionata: 1024x1024\n";
    std::cout << "Threaduri disponibile: "
        << tbb::info::default_concurrency() << "\n";

    std::vector<unsigned char> out(width * height);
    const std::vector<int> kernels = { 3, 5, 7, 11, 15 };
    const std::vector<int> thread_counts = { 1, 2, 4, 8 };
    int max_threads = tbb::info::default_concurrency();

    // Tabel 1: Speedup vs Kernel Size
    std::cout << "\n--- Speedup vs Kernel Size (threads=" << max_threads << ") ---\n";
    std::cout << std::setw(8) << "Kernel"
        << std::setw(14) << "Sequential"
        << std::setw(14) << "TBB-1D"
        << std::setw(14) << "TBB-2D"
        << std::setw(10) << "Su-1D"
        << std::setw(10) << "Su-2D" << "\n"
        << std::string(70, '-') << "\n";

    for (int k : kernels) {
        double t_seq = run_sequential(img, width, height, k);
        double t_1d = run_tbb_1d(img, out, width, height, k, max_threads);
        double t_2d = run_tbb_2d(img, out, width, height, k, max_threads);

        std::cout << std::setw(8) << k
            << std::setw(13) << std::fixed << std::setprecision(2) << t_seq << "ms"
            << std::setw(13) << t_1d << "ms"
            << std::setw(13) << t_2d << "ms"
            << std::setw(9) << t_seq / t_1d << "x"
            << std::setw(9) << t_seq / t_2d << "x\n";
    }

    // Tabel 2: Scalabilitate vs Thread Count (kernel=7)
    std::cout << "\n--- Scalabilitate vs Thread Count (kernel=7) ---\n";
    std::cout << std::setw(10) << "Threads"
        << std::setw(14) << "TBB-1D"
        << std::setw(10) << "Speedup"
        << std::setw(12) << "Efficiency"
        << std::setw(14) << "TBB-2D"
        << std::setw(10) << "Speedup"
        << std::setw(12) << "Efficiency" << "\n"
        << std::string(82, '-') << "\n";

    double t_seq_base = run_sequential(img, width, height, 7);
    for (int nt : thread_counts) {
        if (nt > max_threads) break;
        double t_1d = run_tbb_1d(img, out, width, height, 7, nt);
        double t_2d = run_tbb_2d(img, out, width, height, 7, nt);
        double su_1d = t_seq_base / t_1d;
        double su_2d = t_seq_base / t_2d;

        std::cout << std::setw(10) << nt
            << std::setw(13) << t_1d << "ms"
            << std::setw(9) << su_1d << "x"
            << std::setw(10) << (su_1d / nt * 100) << "%"
            << std::setw(13) << t_2d << "ms"
            << std::setw(9) << su_2d << "x"
            << std::setw(10) << (su_2d / nt * 100) << "%\n";
    }

    return 0;
}