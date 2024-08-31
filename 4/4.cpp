/**
 * Distributed code to execute Q4 - Matrix Inversion
 * :author: flightvin
 */

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

vector<vector<double>> read_file(int argc, char* argv[]) {
    assert(argc >= 2);

    string filepath = argv[1];

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    int N;
    file >> N;

    vector<vector<double>> matrix(N, vector<double>(N));

    for (int i = 0; i<N; i++) {
        for (int j = 0; j<N; j++) {
            file >> matrix[i][j];
        }
    }
    return matrix;
}

double** allocate_square_array(int N) {
    double** arr = new double*[N];
    for (int i = 0; i<N; i++) {
        arr[i] = new double[N];
    }
    return arr;
}

int owner_process(int row_index, int num_processes){
    return row_index % num_processes;
}

/**
 * Handle of the i-th row is given to the (i % num_processes) process
 */
bool check_if_row_in_scope_of_process(int process_rank, int num_processes, int row_index) {
    return process_rank == owner_process(row_index, num_processes);
}

void swap_rows_in_square_matrix(double** arr, int N, int i, int j) {
    for (int k = 0; k<N; k++) {
        double temp = arr[i][k];
        arr[i][k] = arr[j][k];
        arr[j][k] = temp;
    }
}

void gaussian_elimination(
    int process_rank,
    int num_processes,
    int N,
    double** matrix,
    double** identity
) {
    for (int i = 0; i<N; i++) {
        // Determine the process that owns the pivot row
        int pivot_owner_process_rank = owner_process(i, num_processes);

        // Broadcast the pivot value
        if (check_if_row_in_scope_of_process(process_rank, num_processes, i)) {
            double pivot_val = matrix[i][i];

            // Check if the pivot is non-zero
            if (fabs(pivot_val) < 1e-12) {
                MPI_Abort(MPI_COMM_WORLD, 1);
                return;
            }

            for (int j = 0; j<N; j++) {
                matrix[i][j] /= pivot_val;
                identity[i][j] /= pivot_val;
            }
        }

        // Broadcast the updated pivot row to all processes
        MPI_Bcast(&matrix[i][0], N, MPI_DOUBLE, pivot_owner_process_rank, MPI_COMM_WORLD);
        MPI_Bcast(&identity[i][0], N, MPI_DOUBLE, pivot_owner_process_rank, MPI_COMM_WORLD);

        // Eliminate rows below the pivot
        for (int j = i + 1; j<N; j++) {
            // Check if the current row is owned by this process
            if (check_if_row_in_scope_of_process(process_rank, num_processes, j)) {
                double factor = matrix[j][i];
                for (int k = 0; k < N; k++) {
                    matrix[j][k] -= factor * matrix[i][k];
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }

        // Wait for all processes to complete this step
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void back_substitution(
    int process_rank,
    int num_processes,
    int N,
    double** matrix,
    double** identity
) {
    for (int i = N - 1; i>=0; i--) {
        int pivot_owner_process_rank = owner_process(i, num_processes);

        MPI_Bcast(&matrix[i][0], N, MPI_DOUBLE, pivot_owner_process_rank, MPI_COMM_WORLD);
        MPI_Bcast(&identity[i][0], N, MPI_DOUBLE, pivot_owner_process_rank, MPI_COMM_WORLD);

        for (int j = i - 1; j>=0; j--) {
            if (check_if_row_in_scope_of_process(process_rank, num_processes, j)) {
                double factor = matrix[j][i];
                for (int k = 0; k<N; k++) {
                    matrix[j][k] -= factor * matrix[i][k];
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(2);

    MPI_Init(NULL, NULL);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    vector<vector<double>> matrix_vec;

    if (my_rank == 0) {
        matrix_vec = read_file(argc, argv);
    }

    int N = matrix_vec.size();
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double** matrix = allocate_square_array(N);
    if (my_rank == 0) {
        for (int i = 0; i<N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = matrix_vec[i][j];
            }
        }
    }

    for (int i = 0; i<N; i++)
        MPI_Bcast(&matrix[i][0], N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double** identity = allocate_square_array(N);
    for (int i = 0; i<N; i++) {
        for (int j = 0; j<N; j++) {
            identity[i][j] = (i == j) ? 1 : 0;
        }
    }

    gaussian_elimination(my_rank, comm_size, N, matrix, identity);
    back_substitution(my_rank, comm_size, N, matrix, identity);

    if (my_rank == 0) {
        for (int i = 0; i<N; i++) {
            for (int j = 0; j<N; j++) {
                cout << identity[i][j] << " ";
            }
            cout << endl;
        }
    }

    for (int i = 0; i < N; i++) {
        delete[] matrix[i];
        delete[] identity[i];
    }
    delete[] matrix;
    delete[] identity;

    MPI_Finalize();
    return 0;
}
