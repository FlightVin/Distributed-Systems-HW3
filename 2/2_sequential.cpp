/**
 * Sequential code to execute Q2 - Julia Set
 * :author: flightvin
 */

#include <bits/stdc++.h>
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
    *real_pointer = static_cast<double>(-1.5) + (static_cast<double>(i-1) * 3.0)/static_cast<double>(N - 1);
    *imag_pointer = static_cast<double>(1.5) - (static_cast<double>(j-1) * 3.0)/static_cast<double>(M - 1); 
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

int main(int argc, char* argv[]) {
    vector <double> input_args = read_file(argc, argv);
    
    int N, M, K;
    double c_r, c_i;
    N = static_cast<int>(input_args[0]); M = static_cast<int>(input_args[1]); K = static_cast<int>(input_args[2]);
    c_r = input_args[3]; c_i = input_args[4];

    double T = 2;

    for (int i = 1; i<=N; i++){
        for (int j = 1; j<=M; j++){
            cout << check_for_current_grid_coordinate(j, i, N, M, K, T, c_r, c_i) << " ";
        }
        cout<<endl;
    }

    return 0;
}
