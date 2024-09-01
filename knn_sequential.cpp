#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>

struct Point {
    double x, y;
};

struct DistancePoint {
    double distance;
    Point point;
};

double calculateDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    int N, M, K;
    std::vector<Point> P, Q;

    std::ifstream input(argv[1]);
    if (!input) {
        std::cerr << "Error: Unable to open input file." << std::endl;
        return 1;
    }
    input >> N >> M >> K;
    P.resize(N);
    Q.resize(M);

    for (int i = 0; i < N; i++) {
        input >> P[i].x >> P[i].y;
    }
    for (int i = 0; i < M; i++) {
        input >> Q[i].x >> Q[i].y;
    }
    input.close();

    for (int q = 0; q < M; q++) {
        std::vector<DistancePoint> distances(N); 
        for (int i = 0; i < N; i++) {
            distances[i].distance = calculateDistance(Q[q], P[i]);
            distances[i].point = P[i];
        }

        std::partial_sort(distances.begin(), distances.begin() + K, distances.end(),
                         [](const DistancePoint& a, const DistancePoint& b) {
                             return a.distance < b.distance;
                         });

        for (int i = 0; i < K; i++) {
            std::cout << std::fixed << std::setprecision(2)
                      << distances[i].point.x << " "
                      << distances[i].point.y << std::endl;
        }
    }

    return 0;
}
