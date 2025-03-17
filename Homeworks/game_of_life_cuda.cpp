#include <iostream>
#include <cmath>
#include <vector>

#ifdef _MPI_
#include <mpi.h>
#endif

#ifdef _CUDA_
#include <cuda_runtime.h>
#endif

#include <chrono>  // Chrono for timing

using namespace std;
using namespace std::chrono;

constexpr int board_size = 10; // Adjustable global board size

int world_size, world_rank;
int neighbor_north, neighbor_south, neighbor_east, neighbor_west;
int neighbor_northeast, neighbor_northwest, neighbor_southeast, neighbor_southwest;

// CUDA kernel for Game of Life step
#ifdef _CUDA_
__global__ void gameOfLifeKernel(uint8_t* local_board, uint8_t* next_board, int width_with_ghosts, int local_width, int local_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x <= local_width && y > 0 && y <= local_height) {
        int idx = y * width_with_ghosts + x;
        printf("Thread (x=%d, y=%d) accessing idx=%d\n", x, y, idx);  // Debugging output
        int neighbors = local_board[idx - 1] + local_board[idx + 1] +
                        local_board[idx - width_with_ghosts] + local_board[idx + width_with_ghosts] +
                        local_board[idx - width_with_ghosts - 1] + local_board[idx - width_with_ghosts + 1] +
                        local_board[idx + width_with_ghosts - 1] + local_board[idx + width_with_ghosts + 1];
        next_board[idx] = (local_board[idx] && (neighbors == 2 || neighbors == 3)) ||
                          (!local_board[idx] && neighbors == 3);
    }
}
#endif

class GameOfLife {
private:
    uint8_t* local_board;   // Local board with ghost cells
    uint8_t* next_board;    // Next state of the local board
    uint8_t *d_local_board, *d_next_board; // CUDA device memory pointers
    int local_width, local_height;
    int width_with_ghosts, height_with_ghosts;

public:
    GameOfLife(int local_width, int local_height)
        : local_width(local_width), local_height(local_height),
          width_with_ghosts(local_width + 2), height_with_ghosts(local_height + 2) {
        local_board = new uint8_t[width_with_ghosts * height_with_ghosts]();
        next_board = new uint8_t[width_with_ghosts * height_with_ghosts]();
        
        // Allocate memory on GPU
        #ifdef _CUDA_
        cudaMalloc((void**)&d_local_board, width_with_ghosts * height_with_ghosts * sizeof(uint8_t));
        cudaMalloc((void**)&d_next_board, width_with_ghosts * height_with_ghosts * sizeof(uint8_t));
        #endif
    }

    ~GameOfLife() {
        delete[] local_board;
        delete[] next_board;
        
        #ifdef _CUDA_
        cudaFree(d_local_board);
        cudaFree(d_next_board);
        #endif
    }

    uint8_t* get_local_board() const {
        return local_board;
    }

    void print_local_board(const string& label) const {
        cout << "Rank " << world_rank << ": " << label << " Local Board:" << endl;
        for (int y = 1; y <= local_height; ++y) {
            for (int x = 1; x <= local_width; ++x) {
                cout << (int)local_board[y * width_with_ghosts + x] << " ";
            }
            cout << endl;
        }
        cout << "===========================\n";
    }

    void set_global(int x, int y) {
        int grid_size = sqrt(world_size);
        int block_x = x / local_width;
        int block_y = y / local_height;
        int owner_rank = block_y * grid_size + block_x;

        if (owner_rank == world_rank) {
            int local_x = x % local_width + 1;
            int local_y = y % local_height + 1;
            local_board[local_y * width_with_ghosts + local_x] = 1;
            cout << "Rank " << world_rank << ": Set global cell (" << x << ", " << y
                 << ") as local (" << local_x << ", " << local_y << ")" << endl;
        }
    }

    // Function to exchange boundaries between neighboring processes
    void exchange_boundaries() {
        #ifdef _MPI_
        MPI_Request requests[16];
        int req_idx = 0;

        // Debug output
        cout << "Rank " << world_rank << " is exchanging boundaries..." << endl;
        cout << "Neighbor North: " << neighbor_north << ", Neighbor South: " << neighbor_south << endl;

        // Row communication (North and South)
        if (neighbor_north != MPI_PROC_NULL) {
            cout << "Rank " << world_rank << ": Sending to neighbor_north " << neighbor_north << endl;
            MPI_Isend(&local_board[1 * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&local_board[0 * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }
        if (neighbor_south != MPI_PROC_NULL) {
            cout << "Rank " << world_rank << ": Sending to neighbor_south " << neighbor_south << endl;
            MPI_Isend(&local_board[local_height * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&local_board[(local_height + 1) * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }

        // Column communication (West and East)
        if (neighbor_west != MPI_PROC_NULL) {
            vector<uint8_t> left_col(local_height);
            for (int i = 0; i < local_height; ++i) {
                left_col[i] = local_board[(i + 1) * width_with_ghosts + 1];
            }
            cout << "Rank " << world_rank << ": Sending left column to neighbor_west " << neighbor_west << endl;
            MPI_Isend(left_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_west, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(left_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_west, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }

        if (neighbor_east != MPI_PROC_NULL) {
            vector<uint8_t> right_col(local_height);
            for (int i = 0; i < local_height; ++i) {
                right_col[i] = local_board[(i + 1) * width_with_ghosts + local_width];
            }
            cout << "Rank " << world_rank << ": Sending right column to neighbor_east " << neighbor_east << endl;
            MPI_Isend(right_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_east, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(right_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_east, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }

        // Diagonal communication (Northwest, Northeast, Southwest, Southeast)
        if (neighbor_northwest != MPI_PROC_NULL) {
            uint8_t nw = local_board[1 * width_with_ghosts + 1];  // Top-left corner (northwest)
            cout << "Rank " << world_rank << ": Sending nw to neighbor_northwest " << neighbor_northwest << endl;
            MPI_Isend(&nw, 1, MPI_UNSIGNED_CHAR, neighbor_northwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&nw, 1, MPI_UNSIGNED_CHAR, neighbor_northwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }
        if (neighbor_northeast != MPI_PROC_NULL) {
            uint8_t ne = local_board[1 * width_with_ghosts + local_width];  // Top-right corner (northeast)
            cout << "Rank " << world_rank << ": Sending ne to neighbor_northeast " << neighbor_northeast << endl;
            MPI_Isend(&ne, 1, MPI_UNSIGNED_CHAR, neighbor_northeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&ne, 1, MPI_UNSIGNED_CHAR, neighbor_northeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }
        if (neighbor_southwest != MPI_PROC_NULL) {
            uint8_t sw = local_board[local_height * width_with_ghosts + 1];  // Bottom-left corner (southwest)
            cout << "Rank " << world_rank << ": Sending sw to neighbor_southwest " << neighbor_southwest << endl;
            MPI_Isend(&sw, 1, MPI_UNSIGNED_CHAR, neighbor_southwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&sw, 1, MPI_UNSIGNED_CHAR, neighbor_southwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }
        if (neighbor_southeast != MPI_PROC_NULL) {
            uint8_t se = local_board[local_height * width_with_ghosts + local_width];  // Bottom-right corner (southeast)
            cout << "Rank " << world_rank << ": Sending se to neighbor_southeast " << neighbor_southeast << endl;
            MPI_Isend(&se, 1, MPI_UNSIGNED_CHAR, neighbor_southeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
            MPI_Irecv(&se, 1, MPI_UNSIGNED_CHAR, neighbor_southeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        }

        // Wait for all requests to complete
        MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        // Update the ghost corners after receiving data
        if (neighbor_northwest != MPI_PROC_NULL) {
            local_board[0 * width_with_ghosts + 0] = nw;  // Update top-left corner
        }
        if (neighbor_northeast != MPI_PROC_NULL) {
            local_board[0 * width_with_ghosts + (local_width + 1)] = ne;  // Update top-right corner
        }
        if (neighbor_southwest != MPI_PROC_NULL) {
            local_board[(local_height + 1) * width_with_ghosts + 0] = sw;  // Update bottom-left corner
        }
        if (neighbor_southeast != MPI_PROC_NULL) {
            local_board[(local_height + 1) * width_with_ghosts + (local_width + 1)] = se;  // Update bottom-right corner
        }

        // Update left and right ghost columns
        if (neighbor_west != MPI_PROC_NULL) {
            for (int i = 0; i < local_height; ++i) {
                local_board[(i + 1) * width_with_ghosts + 0] = left_col[i];  // Update left ghost column
            }
        }
        if (neighbor_east != MPI_PROC_NULL) {
            for (int i = 0; i < local_height; ++i) {
                local_board[(i + 1) * width_with_ghosts + (local_width + 1)] = right_col[i];  // Update right ghost column
            }
        }

        #endif
    }

    // Function to perform one step of the simulation
    void step() {
        exchange_boundaries();

        // Launch CUDA kernel
        #ifdef _CUDA_
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((local_width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (local_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        gameOfLifeKernel<<<numBlocks, threadsPerBlock>>>(d_local_board, d_next_board, width_with_ghosts, local_width, local_height);
        cudaDeviceSynchronize();
        #endif

        // Perform the local computation (could be optimized with CUDA in future)
        for (int y = 1; y <= local_height; ++y) {
            for (int x = 1; x <= local_width; ++x) {
                int idx = y * width_with_ghosts + x;
                int neighbors = local_board[idx - 1] + local_board[idx + 1] +
                                local_board[idx - width_with_ghosts] + local_board[idx + width_with_ghosts] +
                                local_board[idx - width_with_ghosts - 1] + local_board[idx - width_with_ghosts + 1] +
                                local_board[idx + width_with_ghosts - 1] + local_board[idx + width_with_ghosts + 1];
                next_board[idx] = (local_board[idx] && (neighbors == 2 || neighbors == 3)) ||
                                  (!local_board[idx] && neighbors == 3);
            }
        }

        swap(local_board, next_board);
    }
};

// Function to gather and print the global board
void gather_and_print_global_board(GameOfLife& game, int local_width, int local_height, int grid_size) {
    vector<uint8_t> global_board;
    if (world_rank == 0) {
        global_board.resize(board_size * board_size, 0);
    }

    vector<uint8_t> local_flat(local_width * local_height, 0);
    for (int y = 0; y < local_height; ++y) {
        for (int x = 0; x < local_width; ++x) {
            local_flat[y * local_width + x] = game.get_local_board()[(y + 1) * (local_width + 2) + (x + 1)];
        }
    }

    #ifdef _MPI_
    MPI_Gather(local_flat.data(), local_width * local_height, MPI_UNSIGNED_CHAR,
               global_board.data(), local_width * local_height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    #endif

    if (world_rank == 0) {
        cout << "Global Board State:" << endl;
        vector<vector<uint8_t>> global_matrix(board_size, vector<uint8_t>(board_size, 0));

        // Map gathered local boards into global board
        for (int rank = 0; rank < world_size; ++rank) {
            int block_row = rank / grid_size;
            int block_col = rank % grid_size;
            for (int y = 0; y < local_height; ++y) {
                for (int x = 0; x < local_width; ++x) {
                    int global_y = block_row * local_height + y;
                    int global_x = block_col * local_width + x;
                    global_matrix[global_y][global_x] = global_board[rank * local_width * local_height + y * local_width + x];
                }
            }
        }

        // Print the global board
        for (const auto& row : global_matrix) {
            for (const auto& cell : row) {
                cout << (int)cell << " ";
            }
            cout << endl;
        }
        cout << "===========================\n";
    }
}

int main(int argc, char** argv) {
    #ifdef _MPI_
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    #endif

    int grid_size = sqrt(world_size);
    int local_width = board_size / grid_size;
    int local_height = board_size / grid_size;

    GameOfLife game(local_width, local_height);

    if (world_rank == 0) {
        game.set_global(2, 2);
        game.set_global(3, 3);
        game.set_global(4, 1);
        game.set_global(4, 2);
        game.set_global(4, 3);
    }

    #ifdef _MPI_
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    // Timing start
    auto start_time = chrono::high_resolution_clock::now();

    for (int step = 0; step < 10; ++step) {
        game.step();
        #ifdef _MPI_
        MPI_Barrier(MPI_COMM_WORLD);
        if (step == 9) {
            gather_and_print_global_board(game, local_width, local_height, grid_size);
        }
        #else
        if (step == 9) {
            gather_and_print_global_board(game, local_width, local_height, grid_size);
        }
        #endif
    }

    // Timing end
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    if (world_rank == 0) {
        cout << "Execution time: " << duration.count() << " ms." << endl;
    }

    #ifdef _MPI_
    MPI_Finalize();
    #endif

    return 0;
}
