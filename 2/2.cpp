/**
 * Distributed code to execute Q2 - Julia Set
 * :author: flightvin
 */

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

vector <double>  read_file(int argc, char* argv[]){
    assert(argc >= 2);

    string filepath = argv[1];

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    int N, M, k;
    double c_r, c_i;

    file >> N >> M >> k;
    file >> c_r >> c_i;
    file.close();

    vector <double> result;
    result.push_back(static_cast<double>(N));
    result.push_back(static_cast<double>(M));
    result.push_back(static_cast<double>(k));
    result.push_back(c_r);
    result.push_back(c_i);
    return result;
}

void square_complex_number(double* real_pointer, double* imag_pointer) {
    double a = *real_pointer;
    double b = *imag_pointer;

    *real_pointer = a * a - b * b;
    *imag_pointer = 2 * a * b;
}

void julia_next_iteration(double* real_pointer, double* imag_pointer, double c_r, double c_i) {
    square_complex_number(real_pointer, imag_pointer);

    *real_pointer = *real_pointer + c_r;
    *imag_pointer = *imag_pointer + c_i;
}


bool is_magnitude_under_thresh(double real, double imag, double T){
    if ((real * real + imag * imag) <= T * T) {
        return true;
    }

    return false;
}

int compute_for_point_per_process(double real, double imag, int K, double T, double c_r, double c_i){
    if (!is_magnitude_under_thresh(real, imag, T)){
        return 0;
    }
    
    for (int i = 0; i < K; i++){
        julia_next_iteration(&real, &imag, c_r, c_i);

        if (!is_magnitude_under_thresh(real, imag, T)){
            return 0;
        }
    }

    return 1;
}

void attach_grid_coordinates(double* real_pointer, double* imag_pointer, int i, int j, int N, int M){
    *real_pointer = static_cast<double>(-1.5) + (static_cast<double>(i-1) * 3.0)/static_cast<double>(M - 1);
    *imag_pointer = static_cast<double>(1.5) - (static_cast<double>(j-1) * 3.0)/static_cast<double>(N - 1); 
}

int check_for_current_grid_coordinate(
    int i, 
    int j,
    int N,
    int M,
    int K, 
    double T,
    double c_r, 
    double c_i
){
    double real, imag;
    attach_grid_coordinates(&real, &imag, i, j, N, M);
    return compute_for_point_per_process(real, imag, K, T, c_r, c_i);
}


string check_for_current_grid_coordinate_as_str(
    int i, 
    int j,
    int N,
    int M,
    int K, 
    double T,
    double c_r, 
    double c_i
){
    int actual_res = check_for_current_grid_coordinate(i, j, N, M, K, T, c_r, c_i);
    if (actual_res == 0){
        return "0 ";
    }

    return "1 ";
}

void attach_start_and_end_indices(int* start_idx, int* end_idx, int total, int my_rank, int comm_size){
	int min_distro_per_process = total/comm_size;

	int ideal_start_idx = (my_rank * min_distro_per_process) + 1;
	int ideal_end_idx = ((my_rank + 1) * min_distro_per_process);

	int num_tasks_left = total - min_distro_per_process * comm_size;

	if (num_tasks_left > my_rank){
		*start_idx = ideal_start_idx + my_rank;
		*end_idx = ideal_end_idx + my_rank + 1;
	} else {
		*start_idx = ideal_start_idx + num_tasks_left;
		*end_idx = ideal_end_idx + num_tasks_left;
	}
}

string compute_answer_per_proc(
    int start_index, 
    int end_index,
    int N,
    int M,
    int K,
    double T,
    double c_r, 
    double c_i
){
	int start_row = (start_index + M - 1) / M;
	int end_row = (end_index + M - 1) / M;

	string res = "";
	
	int start_column = start_index - M * (start_row - 1);
	int end_column = end_index - M * (end_row - 1);

    if (start_row == end_row){
        for (int col = start_column; col <= end_column; col++){
            res += check_for_current_grid_coordinate_as_str(col, start_row, N, M, K, T, c_r, c_i);
        }
    } else {

        for (int col = start_column; col <= M; col++){
            res += check_for_current_grid_coordinate_as_str(col, start_row, N, M, K, T, c_r, c_i);
        }
        res += "\n";

        for (int row = start_row + 1; row < end_row; row++){
            for (int col = 1; col <= M; col ++){
                res += check_for_current_grid_coordinate_as_str(col, row, N, M, K, T, c_r, c_i);
            }
            res += "\n";
        }

        for (int col = 1; col <= end_column; col++){
            res += check_for_current_grid_coordinate_as_str(col, end_row, N, M, K, T, c_r, c_i);
        }
    }

    if (end_column == M){
        res += "\n";
    }

    // res += "\n\n\n\n\n";

	return res;
}

int main(int argc, char* argv[]) {
	// initialize the required MPI things
    MPI_Init(NULL, NULL);
 
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

	// some variables shared by every process
	double T = 2;
	vector <double> input_args;

	// the process that reads the input
	if (my_rank == 0){
		input_args = read_file(argc, argv);
	}

	// Distributed Comms
	// **********************************************************************

	// broadcast vector size
    int vector_size = input_args.size();
    MPI_Bcast(&vector_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// load input vector for every process
    input_args.resize(vector_size);
    MPI_Bcast(input_args.data(), vector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// read the arguments
	int N, M, K;
	double c_r, c_i;
	N = static_cast<int>(input_args[0]); M = static_cast<int>(input_args[1]); K = static_cast<int>(input_args[2]);
	c_r = input_args[3]; c_i = input_args[4];

	// now every process computes results
	int start_idx, end_idx;
	attach_start_and_end_indices(&start_idx, &end_idx, N*M, my_rank, comm_size);
	// start_idx and end_idx will be from 1 to N*M
	// **********************************************************************



	// Actual distributed computation step 
	// **********************************************************************
	string res = compute_answer_per_proc(start_idx, end_idx, N, M, K, T, c_r, c_i);
	// **********************************************************************

    // wait for everyone to complete
    MPI_Barrier(MPI_COMM_WORLD);

	// Printing the answer
	// **********************************************************************
    int res_len = res.size();
    vector<int> lengths(comm_size);
    MPI_Gather(&res_len, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> displs(comm_size);
    int total_length = 0;
    if (my_rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < comm_size; ++i) {
            displs[i] = displs[i-1] + lengths[i-1];
        }
        total_length = displs[comm_size-1] + lengths[comm_size-1];
    }

    vector<char> all_strings(total_length);
    MPI_Gatherv(res.c_str(), res_len, MPI_CHAR, all_strings.data(), lengths.data(), displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        for (int i = 0; i < comm_size; ++i) {
            string proc_result(all_strings.data() + displs[i], lengths[i]);
            cout << proc_result; cout.flush();
        }
    }
	// **********************************************************************

	MPI_Finalize();

    return 0;
}
