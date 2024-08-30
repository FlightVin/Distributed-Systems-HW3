/**
 * Distributed code to execute Q4 - Matrix inversion
 * :author: flightvin
 */


// https://dl.acm.org/doi/pdf/10.1145/321420.321434
// https://www.cs.utexas.edu/~flame/pubs/SIAMMatrixInversion.pdf
// Use gauss jordan

#include <bits/stdc++.h>
using namespace std;

void row_reduce(vector<vector<double>>& augmented) {
    int n = augmented.size();
    for (int i = 0; i < n; i++) {
        double diag = augmented[i][i];
        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= diag;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = augmented[k][i];
                for (int j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
}

vector<vector<double>> inverse_matrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    vector<vector<double>> augmented(n, vector<double>(2 * n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
        }
        for (int j = 0; j < n; j++) {
            augmented[i][j + n] = (i == j) ? 1 : 0;
        }
    }

    row_reduce(augmented);

    vector<vector<double>> inverse(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = augmented[i][j + n];
        }
    }

    return inverse;
}

vector <vector<double>> read_file(int argc, char* argv[]){
    assert(argc >= 2);

    string filepath = argv[1];

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filepath << endl;
        return {};
    }

    int N;
    file >> N;

    vector <vector<double>> matrix(N, vector<double> (N));

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            file >> matrix[i][j];
        }
    }
    return matrix;
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(2);

    vector <vector<double>> matrix = read_file(argc, argv);
    
    vector<vector<double>> inverse = inverse_matrix(matrix);

    int N = inverse.size();
    for (int i = 0; i<N; i++){
        for (int j = 0; j<N; j++){
            cout<<inverse[i][j]<<" ";
        }
        cout<<endl;
    }

    return 0;
}