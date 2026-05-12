#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cstring>

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

double run_sequential(const std::vector<unsigned char>& img,
    int width, int height, int kernel_size, int reps = 3)
{
    int half = kernel_size / 2;
    std::vector<unsigned char> out(width * height);
    double t0 = MPI_Wtime();
    for (int rep = 0; rep < reps; ++rep)
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                out[r * width + c] = median_of_window(
                    img.data(), width, height, r, c, half);
    return (MPI_Wtime() - t0) * 1000.0 / reps;
}

//MPI parallel median filter

double run_mpi(const std::vector<unsigned char>& img,
    int width, int height, int kernel_size,
    int rank, int size, int reps = 3)
{
    int half = kernel_size / 2;

    int rows_per_proc = height / size;
    int remainder = height % size;

    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int my_start = 0;
    for (int i = 0; i < rank; ++i)
        my_start += rows_per_proc + (i < remainder ? 1 : 0);

    std::vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * width;
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
    }

    std::vector<unsigned char> result;
    if (rank == 0) result.resize(width * height);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int rep = 0; rep < reps; ++rep) {
        std::vector<unsigned char> local_out(my_rows * width);
        for (int r = 0; r < my_rows; ++r) {
            int global_r = my_start + r;
            for (int c = 0; c < width; ++c)
                local_out[r * width + c] = median_of_window(
                    img.data(), width, height, global_r, c, half);
        }

        MPI_Gatherv(local_out.data(), my_rows * width, MPI_UNSIGNED_CHAR,
            rank == 0 ? result.data() : nullptr,
            recvcounts.data(), displs.data(), MPI_UNSIGNED_CHAR,
            0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return (MPI_Wtime() - t0) * 1000.0 / reps;
}

// Main

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = 1024, height = 1024;
    std::vector<unsigned char> img_data(width * height, 0);

    if (rank == 0) {
        const char* path = "C:\\Users\\Elena\\Desktop\\UNI\\Facultate\\an 3 sem 2\\PARADIS\\PROIECT-PARADIS-4\\mri\\brain_tumor_dataset\\yes\\Y1.jpg";
        int w, h, ch;
        unsigned char* px = stbi_load(path, &w, &h, &ch, 1);
        if (!px) {
            std::cerr << "Eroare la incarcarea imaginii!\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Imagine incarcata: " << w << "x" << h << " px\n";
        std::cout << "Imagine redimensionata: 1024x1024\n";
        std::cout << "Procese MPI: " << size << "\n";

        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                img_data[r * width + c] = px[(r * h / height) * w + (c * w / width)];
        stbi_image_free(px);
    }

    // Procesul 0 trimite imaginea la toate procesele
    MPI_Bcast(img_data.data(), width * height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    //Tabel 1: Speedup vs Kernel Size
    const std::vector<int> kernels = { 3, 5, 7, 11, 15 };

    if (rank == 0) {
        std::cout << "\n--- Speedup vs Kernel Size (procese=" << size << ") ---\n";
        std::cout << std::setw(8) << "Kernel"
            << std::setw(14) << "Sequential"
            << std::setw(14) << "MPI"
            << std::setw(10) << "Speedup" << "\n"
            << std::string(46, '-') << "\n";
    }

    for (int k : kernels) {
        double t_seq = 0;
        if (rank == 0)
            t_seq = run_sequential(img_data, width, height, k);

        double t_mpi = run_mpi(img_data, width, height, k, rank, size);

        if (rank == 0) {
            double speedup = t_seq / t_mpi;
            std::cout << std::setw(8) << k
                << std::setw(13) << std::fixed << std::setprecision(2) << t_seq << "ms"
                << std::setw(13) << t_mpi << "ms"
                << std::setw(9) << speedup << "x\n";
        }
    }

    //Tabel 2: Scalabilitate (kernel=7)
    if (rank == 0) {
        double t_seq = run_sequential(img_data, width, height, 7);
        double t_mpi = run_mpi(img_data, width, height, 7, rank, size);
        double speedup = t_seq / t_mpi;
        double efficiency = speedup / size * 100.0;

        std::cout << "\n--- Scalabilitate (kernel=7) ---\n";
        std::cout << std::setw(10) << "Procese"
            << std::setw(14) << "Sequential"
            << std::setw(14) << "MPI"
            << std::setw(10) << "Speedup"
            << std::setw(12) << "Efficiency" << "\n"
            << std::string(60, '-') << "\n";
        std::cout << std::setw(10) << size
            << std::setw(13) << t_seq << "ms"
            << std::setw(13) << t_mpi << "ms"
            << std::setw(9) << speedup << "x"
            << std::setw(10) << efficiency << "%\n";
    }
    else {
        run_mpi(img_data, width, height, 7, rank, size);
    }

    MPI_Finalize();
    return 0;
}