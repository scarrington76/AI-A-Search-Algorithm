#Scott Carrington

# This program implements A* for solving a sliding tile puzzle

import numpy as np
import queue


class PuzzleState():
    SOLVED_PUZZLE = np.arange(9).reshape((3, 3))  # Goal puzzle state
    move = 0  # Number of moves made to solve the puzzle

    def __init__(self, conf, g, predState):
        """ Constructor function to initialize the object """
        self.puzzle = conf  # Configuration of the state
        self.gcost = g  # Path cost
        self._compute_heuristic_cost()  # Set heuristic cost
        self.fcost = self.gcost + self.hcost
        self.pred = predState  # Predecesor state
        self.zeroloc = np.argwhere(self.puzzle == 0)[0]
        self.action_from_pred = None

    def __hash__(self):
        """ Maps the object to an integer, so it can be used in a set """
        return tuple(self.puzzle.ravel()).__hash__()

    def _compute_heuristic_cost(self):
        """ Updates the heuristic value (hcost) """
        # Map tiles to their final coordinates
        d = {0: (0, 0), 1: (0, 1), 2: (0, 2),
             3: (1, 0), 4: (1, 1), 5: (1, 2),
             6: (2, 0), 7: (2, 1), 8: (2, 2)}
        total_h = 0  # inital heuristic cost

        # add up Manhattan distances for each tile
        for r in range(3):
            for c in range(3):
                if self.puzzle[r][c] != 0:
                    total_h = total_h + abs(d[self.puzzle[r][c]][0] - r) + abs(d[self.puzzle[r][c]][1] - c)

        self.hcost = total_h

    def is_goal(self):
        """ Returns true if the current puzzle state is the goal state """
        return np.array_equal(PuzzleState.SOLVED_PUZZLE, self.puzzle)

    def __eq__(self, other):
        """ Returns true if current puzzle is in same configuration as other """
        return np.array_equal(self.puzzle, other.puzzle)

    def randomize(self):
        """ Randomizes the puzzle state """
        self.puzzle = np.random.permutation(np.arange(9)).reshape(3, 3)
        self._compute_heuristic_cost()
        self.fcost = self.gcost + self.hcost
        self.zeroloc = np.argwhere(self.puzzle == 0)[0]

    def __lt__(self, other):
        """ Returns true if current puzzle state has f-cost lower than other """
        return self.fcost < other.fcost

    def __str__(self):
        """
        Returns a string representation of puzzle state so the object
        can easily be displayed with print statements
        """
        return np.str(self.puzzle)

    def show_path(self):
        """ Shows the moves needed to go from initial state to the goal state """
        # Check for base case: no predecesor node
        if self.pred is not None:
            self.pred.show_path()

        # Print out move and state
        if PuzzleState.move == 0:
            print('START')
        else:
            print('Move', PuzzleState.move, 'ACTION:', self.action_from_pred)
        PuzzleState.move = PuzzleState.move + 1
        print(self)

    def can_move(self, direction):
        """ Returns true if move can be made in the specified direction """
        if direction == 'up':
            return self.zeroloc[0] > 0
        elif direction == 'down':
            return self.zeroloc[0] < 2
        elif direction == 'left':
            return self.zeroloc[1] > 0
        elif direction == 'right':
            return self.zeroloc[1] < 2
        else:
            raise ('wrong direction for checking move')

    def gen_next_state(self, direction):
        """
        Generates the successor puzzle state by swapping the zero with the
        tile located in the specified direction.
        Assumes the move can be made.
        """
        # Generate a copy of the current state
        s = PuzzleState(np.array(self.puzzle), self.gcost + 1, self)

        # Swap the zero with the tile located in the specified direction
        if direction == 'up':
            s.puzzle[s.zeroloc[0]][s.zeroloc[1]] = s.puzzle[s.zeroloc[0] - 1][s.zeroloc[1]]
            s.zeroloc[0] = s.zeroloc[0] - 1
        elif direction == 'down':
            s.puzzle[s.zeroloc[0]][s.zeroloc[1]] = s.puzzle[s.zeroloc[0] + 1][s.zeroloc[1]]
            s.zeroloc[0] = s.zeroloc[0] + 1
        elif direction == 'left':
            s.puzzle[s.zeroloc[0]][s.zeroloc[1]] = s.puzzle[s.zeroloc[0]][s.zeroloc[1] - 1]
            s.zeroloc[1] = s.zeroloc[1] - 1
        elif direction == 'right':
            s.puzzle[s.zeroloc[0]][s.zeroloc[1]] = s.puzzle[s.zeroloc[0]][s.zeroloc[1] + 1]
            s.zeroloc[1] = s.zeroloc[1] + 1
        else:
            raise ('wrong direction for next move')
        s.puzzle[s.zeroloc[0]][s.zeroloc[1]] = 0

        # Update the predecesor action and f-cost
        s.action_from_pred = direction
        s._compute_heuristic_cost()
        s.fcost = s.gcost + s.hcost

        return s


###########################
### Program begins here ###
###########################

# Display heading info
print('Artificial Intelligence')
print('MP1: A* for Sliding Puzzle')
print('SEMESTER: Fall 2019')
print('NAME: Scott Carrington')
print()

# Setup search data structures and load initial state onto frontier
frontier = queue.PriorityQueue()  # Keeps track of states that were reached, but not yet visited
a = np.loadtxt('mp1input.txt', dtype=np.int32)
start_state = PuzzleState(a, 0, None)
frontier.put(start_state)
closed_set = set()  # Keeps track of states that were already visited (explored)
num_states = 0  # Keeps track of the number of states visited during the search process

# Start of A* algorithm
while not frontier.empty():
    # Choose state at front of priority queue
    next_state = frontier.get()
    num_states = num_states + 1

    # If goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break

    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)

    # Expand state (up to 4 moves possible)
    possible_moves = ['up', 'down', 'left', 'right']
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:
                frontier.put(neighbor)
            # If it's already in the frontier, it's gauranteed to have lower cost, so no need to update

print('\nNumber of states visited =', num_states)