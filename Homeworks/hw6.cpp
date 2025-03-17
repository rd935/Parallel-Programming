//Ritwika Das - rd935
//Cited: some help from ChatGPT

//Compile: mpic++ -g -o game_of_life hw6.cpp
//Run: mpirun -np [NUM_PROCESSES] ./game_of_life

//This code is set to go through 10 steps with the Game of Life,
//however you can change it to do 1 step in the main function.
//You should only need to change the for statement.


#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

constexpr int board_size = 10; // Adjustable global board size

int world_size, world_rank;
int neighbor_north, neighbor_south, neighbor_east, neighbor_west;
int neighbor_northeast, neighbor_northwest, neighbor_southeast, neighbor_southwest;

class GameOfLife {
private:
    uint8_t* local_board;   // Local board with ghost cells
    uint8_t* next_board;    // Next state of the local board
    int local_width, local_height;
    int width_with_ghosts, height_with_ghosts;

public:
    GameOfLife(int local_width, int local_height)
        : local_width(local_width), local_height(local_height),
          width_with_ghosts(local_width + 2), height_with_ghosts(local_height + 2) {
        local_board = new uint8_t[width_with_ghosts * height_with_ghosts]();
        next_board = new uint8_t[width_with_ghosts * height_with_ghosts]();
        cout << "Rank " << world_rank << ": Initialized GameOfLife with local board size "
             << local_width << "x" << local_height << endl;
    }

    ~GameOfLife() {
        delete[] local_board;
        delete[] next_board;
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

    void exchange_boundaries() {
        MPI_Request requests[16];
        int req_idx = 0;

        // Row communication (North and South)
        MPI_Isend(&local_board[1 * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_board[0 * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_north, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Isend(&local_board[local_height * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&local_board[(local_height + 1) * width_with_ghosts + 1], local_width, MPI_UNSIGNED_CHAR, neighbor_south, 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Column communication (West and East)
        vector<uint8_t> left_col(local_height), right_col(local_height);
        for (int i = 0; i < local_height; ++i) {
            left_col[i] = local_board[(i + 1) * width_with_ghosts + 1];
            right_col[i] = local_board[(i + 1) * width_with_ghosts + local_width];
        }
        MPI_Isend(left_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_west, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(left_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_west, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Isend(right_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_east, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(right_col.data(), local_height, MPI_UNSIGNED_CHAR, neighbor_east, 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Diagonal communication (Northwest, Northeast, Southwest, Southeast)
        uint8_t nw = local_board[1 * width_with_ghosts + 1];  // Top-left corner (northwest)
        uint8_t ne = local_board[1 * width_with_ghosts + local_width];  // Top-right corner (northeast)
        uint8_t sw = local_board[local_height * width_with_ghosts + 1];  // Bottom-left corner (southwest)
        uint8_t se = local_board[local_height * width_with_ghosts + local_width];  // Bottom-right corner (southeast)

        MPI_Isend(&nw, 1, MPI_UNSIGNED_CHAR, neighbor_northwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&nw, 1, MPI_UNSIGNED_CHAR, neighbor_northwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Isend(&ne, 1, MPI_UNSIGNED_CHAR, neighbor_northeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&ne, 1, MPI_UNSIGNED_CHAR, neighbor_northeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Isend(&sw, 1, MPI_UNSIGNED_CHAR, neighbor_southwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&sw, 1, MPI_UNSIGNED_CHAR, neighbor_southwest, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Isend(&se, 1, MPI_UNSIGNED_CHAR, neighbor_southeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);
        MPI_Irecv(&se, 1, MPI_UNSIGNED_CHAR, neighbor_southeast, 0, MPI_COMM_WORLD, &requests[req_idx++]);

        // Wait for all requests to complete
        MPI_Waitall(req_idx, requests, MPI_STATUSES_IGNORE);

        // Update the ghost corners after receiving data
        local_board[0 * width_with_ghosts + 0] = nw;  // Update top-left corner
        local_board[0 * width_with_ghosts + (local_width + 1)] = ne;  // Update top-right corner
        local_board[(local_height + 1) * width_with_ghosts + 0] = sw;  // Update bottom-left corner
        local_board[(local_height + 1) * width_with_ghosts + (local_width + 1)] = se;  // Update bottom-right corner

        // Update left and right ghost columns
        for (int i = 0; i < local_height; ++i) {
            local_board[(i + 1) * width_with_ghosts + 0] = left_col[i];  // Update left ghost column
            local_board[(i + 1) * width_with_ghosts + (local_width + 1)] = right_col[i];  // Update right ghost column
        }
    }

    void step() {
        exchange_boundaries();
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

void initialize_neighbors(int rank, int grid_size) {
    int row = rank / grid_size;
    int col = rank % grid_size;
    int max_row = grid_size - 1;
    int max_col = grid_size - 1;

    neighbor_north = (row == 0) ? MPI_PROC_NULL : rank - grid_size;
    neighbor_south = (row == max_row) ? MPI_PROC_NULL : rank + grid_size;
    neighbor_west = (col == 0) ? MPI_PROC_NULL : rank - 1;
    neighbor_east = (col == max_col) ? MPI_PROC_NULL : rank + 1;

    neighbor_northwest = (row == 0 || col == 0) ? MPI_PROC_NULL : rank - grid_size - 1;
    neighbor_northeast = (row == 0 || col == max_col) ? MPI_PROC_NULL : rank - grid_size + 1;
    neighbor_southwest = (row == max_row || col == 0) ? MPI_PROC_NULL : rank + grid_size - 1;
    neighbor_southeast = (row == max_row || col == max_col) ? MPI_PROC_NULL : rank + grid_size + 1;
}

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

    MPI_Gather(local_flat.data(), local_width * local_height, MPI_UNSIGNED_CHAR,
               global_board.data(), local_width * local_height, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

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
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int grid_size = sqrt(world_size);
    int local_width = board_size / grid_size;
    int local_height = board_size / grid_size;

    initialize_neighbors(world_rank, grid_size);

    GameOfLife game(local_width, local_height);

    if (world_rank == 0) {
        game.set_global(2, 2);
        game.set_global(3, 3);
        game.set_global(4, 1);
        game.set_global(4, 2);
        game.set_global(4, 3);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    gather_and_print_global_board(game, local_width, local_height, grid_size);

    for (int step = 0; step < 10; ++step) {
        game.step();
        MPI_Barrier(MPI_COMM_WORLD);
        if (step == 9) { // Print only the final state
            gather_and_print_global_board(game, local_width, local_height, grid_size);
        }
    }

    MPI_Finalize();
    return 0;
}
