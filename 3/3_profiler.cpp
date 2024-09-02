/**
 * Distributed Solution for Q3- Prefix sum
 * author: Omerprogrammer
 *
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

std::vector<double> parallelPrefixSum(std::vector<double>& local_arr, int rank, int size) {
    int local_size = local_arr.size();
    double local_sum = 0;

    // Compute the local prefix sum
    for (int i = 0; i < local_size; ++i) {
        local_sum += local_arr[i];
        local_arr[i] = local_sum;
    }

    // Gather the local sums to calculate the prefix sum offset
    std::vector<double> all_sums(size, 0.0);
    MPI_Allgather(&local_sum, 1, MPI_DOUBLE, all_sums.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    double prefix_sum_offset = 0;
    for (int i = 0; i < rank; ++i) {
        prefix_sum_offset += all_sums[i];
    }

    // Add the offset to each element in the local prefix sum
    for (int i = 0; i < local_size; ++i) {
        local_arr[i] += prefix_sum_offset;
    }

    return local_arr;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 0;
    std::vector<double> arr;

    if (rank == 0) {
        if (argc != 2) {
            std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::ifstream input(argv[1]);
        if (!input) {
            std::cerr << "Error: Unable to open input file: " << argv[1] << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        input >> N;
        if (N <= 0) {
            std::cerr << "Error: Invalid array size " << N << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        arr.resize(N);
        for (int i = 0; i < N; ++i) {
            if (!(input >> arr[i])) {
                std::cerr << "Error: Failed to read element " << i << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Fixes edge case sync errors on LG Gram's issues
    MPI_Barrier(MPI_COMM_WORLD);

    // Calculate local array size and distribute the array
    int local_N = N / size;
    int remainder = N % size;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);

    for (int i = 0; i < size; ++i) {
        sendcounts[i] = local_N + (i < remainder ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    local_N = sendcounts[rank];
    std::vector<double> local_arr(local_N);

    // Fixes edge case sync errors on LG Gram's issues
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(arr.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_arr.data(), local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Fixes edge case sync errors on LG Gram's issues
    MPI_Barrier(MPI_COMM_WORLD);

    double completion_time;
    double start_time = MPI_Wtime();

    // Compute parallel prefix sum
    local_arr = parallelPrefixSum(local_arr, rank, size);

    completion_time = MPI_Wtime() - start_time;
    double execution_time;
    MPI_Reduce(&completion_time, &execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to root process
    // Fixes edge case sync errors on LG Gram's issues
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(local_arr.data(), local_N, MPI_DOUBLE, arr.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Fixes edge case sync errors on LG Gram's issues
    MPI_Barrier(MPI_COMM_WORLD);

    // Print result
    // if (rank == 0) {
    //     for (int i = 0; i < N; ++i) {
    //         std::cout << std::fixed << std::setprecision(2) << arr[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    if (rank == 0)
        std::cout << execution_time << std::endl;

    MPI_Finalize();
    return 0;
}
