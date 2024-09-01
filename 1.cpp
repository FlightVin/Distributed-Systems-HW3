#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>

struct Point
{
    double x, y;
};

struct DistancePoint
{
    double distance;
    Point point;
};

double calculateDistance(const Point &p1, const Point &p2)
{
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 2)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int N, M, K;
    std::vector<Point> P, Q;

    if (world_rank == 0)
    {
        std::ifstream input(argv[1]);
        if (!input)
        {
            std::cerr << "Error: Unable to open input file." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        input >> N >> M >> K;
        P.resize(N);
        Q.resize(M);

        for (int i = 0; i < N; i++)
        {
            input >> P[i].x >> P[i].y;
        }
        for (int i = 0; i < M; i++)
        {
            input >> Q[i].x >> Q[i].y;
        }
        input.close();
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = N / world_size;
    int remainder = N % world_size;
    int local_offset = world_rank * local_size + std::min(world_rank, remainder);
    if (world_rank < remainder)
        local_size++;

    std::vector<Point> local_P(local_size);

    MPI_Datatype MPI_POINT;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    std::vector<int> sendcounts(world_size), displs(world_size);
    for (int i = 0; i < world_size; ++i)
    {
        sendcounts[i] = N / world_size + (i < remainder ? 1 : 0);
        displs[i] = i * (N / world_size) + std::min(i, remainder);
    }

    MPI_Scatterv(P.data(), sendcounts.data(), displs.data(), MPI_POINT,
                 local_P.data(), local_size, MPI_POINT,
                 0, MPI_COMM_WORLD);

    for (int q = 0; q < M; q++)
    {
        Point query_point;
        if (world_rank == 0)
        {
            query_point = Q[q];
        }

        MPI_Bcast(&query_point, 1, MPI_POINT, 0, MPI_COMM_WORLD);

        std::vector<DistancePoint> local_distances(local_size);
        for (int i = 0; i < local_size; i++)
        {
            local_distances[i].distance = calculateDistance(query_point, local_P[i]);
            local_distances[i].point = local_P[i];
        }

        std::partial_sort(local_distances.begin(), local_distances.begin() + std::min(K, (int)local_distances.size()), local_distances.end(),
                          [](const DistancePoint &a, const DistancePoint &b)
                          { return a.distance < b.distance; });

        // Pad local_distances with dummy objects if it has fewer than K elements
        while (local_distances.size() < K)
        {
            local_distances.push_back({std::numeric_limits<double>::max(), {0.0, 0.0}});
        }

        if (world_rank == 0)
        {
            std::vector<DistancePoint> gathered_distances(K * world_size);

            std::vector<int> recvcounts(world_size), recvdispls(world_size);
            for (int i = 0; i < world_size; ++i)
            {
                recvcounts[i] = K * sizeof(DistancePoint); // Always receive K objects
                recvdispls[i] = i * K * sizeof(DistancePoint);
            }

            MPI_Gatherv(local_distances.data(), K * sizeof(DistancePoint), MPI_BYTE,
                        gathered_distances.data(), recvcounts.data(), recvdispls.data(), MPI_BYTE,
                        0, MPI_COMM_WORLD);

            std::partial_sort(gathered_distances.begin(), gathered_distances.begin() + K, gathered_distances.end(),
                              [](const DistancePoint &a, const DistancePoint &b)
                              { return a.distance < b.distance; });

            for (int i = 0; i < K; i++)
            {
                std::cout << std::fixed << std::setprecision(2)
                          << gathered_distances[i].point.x << " "
                          << gathered_distances[i].point.y << std::endl;
            }
        }
        else
        {
            MPI_Gatherv(local_distances.data(), K * sizeof(DistancePoint), MPI_BYTE,
                        nullptr, nullptr, nullptr, MPI_BYTE,
                        0, MPI_COMM_WORLD);
        }
    }

    MPI_Type_free(&MPI_POINT);
    MPI_Finalize();
    return 0;
}
