/**
 * Distributed code to execute Q5 - Parallel Matrix Chain Multiplication Problem
 * :author: flightvin
 */

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

vector <long long> read_file(int argc, char* argv[]) {
    assert(argc >= 2);

    string filepath = argv[1];

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    int N;
    file >> N;

    vector <long long> dimensions(N+1);

    for (int i = 0; i<N+1; i++) {
        file >> dimensions[i];
    }
    return dimensions;
}

int main(int argc, char* argv[]) {

    MPI_Init(NULL, NULL);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    vector <long long> dimensions_vec;

    if (my_rank == 0) {
        dimensions_vec = read_file(argc, argv);
    }

    int N = dimensions_vec.size() - 1;
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int num_dims = N + 1;
    dimensions_vec.resize(num_dims);
    MPI_Bcast(dimensions_vec.data(), num_dims, MPI_LONG_LONG, 0, MPI_COMM_WORLD);


    double completion_time;
    double start_time = MPI_Wtime();

    // initialize DP array
    vector <vector <long long>> dp_array(num_dims, vector <long long> (num_dims, 0));

    // Parallel Matrix Chain Multiplication Calculation
    for (int len = 2; len <= N; len++) {
        vector <int> counts(comm_size, 0);
        vector <int> displacements(comm_size, 0);

        int total = N - len + 1;
        vector <long long> gathered_results = vector <long long>(total, 0);

        int block_size = total / comm_size;
        int remainder = total % comm_size;

        int start_idx = my_rank * block_size + min(my_rank, remainder);
        int end_idx;
        if (my_rank < remainder) {
            end_idx = start_idx + block_size + 1;
        } else {
            end_idx = start_idx + block_size;
        }

        int local_size = end_idx - start_idx;
        vector <long long> local_results = vector <long long>(local_size, 0);

        displacements[0] = 0;
        for (int i = 0; i < comm_size; i++) {
            counts[i] = total / comm_size + (i < remainder ? 1 : 0);
            if (i > 0) {
                displacements[i] = displacements[i - 1] + counts[i - 1];
            }
        }

        for (int i = start_idx; i < end_idx; i++) {
            int j = i + len;
            dp_array[i][j] = LLONG_MAX;

            for (int k = i + 1; k < j; k++) {
                dp_array[i][j] = min(
                    dp_array[i][j], 

                    (dimensions_vec[i] * dimensions_vec[k] * dimensions_vec[j]) +
                        dp_array[i][k] + dp_array[k][j] 
                    );
            }

            local_results[i - start_idx] = dp_array[i][j];
        }

        // wait for all processes to complete
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgatherv(local_results.data(), local_size, MPI_LONG_LONG, gathered_results.data(), counts.data(), displacements.data(), MPI_LONG_LONG, MPI_COMM_WORLD);

        for (int i = 0; i < total; i++) {
            int j = i + len;
            dp_array[i][j] = gathered_results[i];
        }
    }

    // // Final Result
    // if (my_rank == 0) {
    //     cout << dp_array[0][N] << endl;
    // }
    completion_time = MPI_Wtime() - start_time;
    double execution_time;
    MPI_Reduce(&completion_time, &execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
        std::cout << execution_time << std::endl;

    MPI_Finalize();
    return 0;
}
