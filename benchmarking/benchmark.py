import time
import copy
from z3 import *

# utility functions
def valid(board, num, pos):
    # check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False
    # check col
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False
    # check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False
    return True

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)
    return None

# sudoku solvers
def solve_sudoku_dfs(board):
    """
    Solves Sudoku using simple recursive backtracking.
    Modifies board in-place. 
    Returns True if solved, False if impossible.
    """
    find = find_empty(board)
    if not find:
        return True
    row, col = find

    for i in range(1, 10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve_sudoku_dfs(board):
                return True

            board[row][col] = 0
    return False

def solve_sudoku_sat(board):
    """
    Solves Sudoku using the Z3 Theorem Prover (SAT Solver).
    This is supposedly faster than naive backtracking... but the loading time for Z3 might negate that.
    """
    # create 9x9 matrix of integer variables
    X = [[Int(f"x_{r}_{c}") for c in range(9)] for r in range(9)]
    s = Solver() #z3 solver instance

    # add constraints
    for r in range(9):
        for c in range(9):
            # cells must be 1-9
            s.add(X[r][c] >= 1, X[r][c] <= 9)
            
            # if the board has a fixed number, add that constraint
            if board[r][c] != 0:
                s.add(X[r][c] == board[r][c])

    # distinct constraints
    for r in range(9):
        s.add(Distinct(X[r])) # row uiqieness

    for c in range(9):
        s.add(Distinct([X[r][c] for r in range(9)])) # col uniqueness

    for i in range(3):
        for j in range(3):
            # box uniqueness
            box_cells = [
                X[r][c] 
                for r in range(i*3, (i+1)*3) 
                for c in range(j*3, (j+1)*3)
            ]
            s.add(Distinct(box_cells))

    if s.check() == sat:
        m = s.model()
        result = [[m.evaluate(X[r][c]).as_long() for c in range(9)] for r in range(9)]
        return result
    else:
        return None

# benchmarking utility function
def run_benchmark(puzzles, limit=None):
    """
    Runs both solvers on the provided list of puzzles.
    Returns a list of dictionaries containing raw timing results.
    """
    results = []
    
    subset = puzzles[:limit] if limit else puzzles
    
    print(f"Starting benchmark on {len(subset)} puzzles...")
    
    for idx, puzzle in enumerate(subset):
        board_naive = copy.deepcopy(puzzle)
        
        start_n = time.time()
        solve_sudoku_dfs(board_naive)
        end_n = time.time()
        
        start_z = time.time()
        solve_sudoku_sat(puzzle)
        end_z = time.time()
        
        results.append({
            "Puzzle_ID": idx,
            "Backtracking_Time": end_n - start_n,
            "Z3_Time": end_z - start_z
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(subset)} puzzles...")

    return results
