import numpy as np
import piece

class Environment:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    # coord Agent (WK)
    currentStateW = ()
    # Objective coord
    currentStateB = ()
    # coord obstacle
    currentObs = ()
    # Standard penalty per movement
    reward = -1
    # Final reward
    treasure = 100
    # Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)):

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            # Initialization with dtype=object to store both numbers and pieces
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs = currentObs

            # Setting special positions in the environment
            # Placing the reward at position B (BK is reward)
            self.board[self.currentStateB] = self.treasure
            
            # Placing the King at position W (WK is the agent)
            self.board[self.currentStateW] = piece.King(True) 

            # Placing obstacle
            self.board[self.currentObs] = self.wall_penalization

            self.initState = 1

    def init2(self, rows=3, cols=4,currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)):     

        if self.initState is None:
            self.rows = rows
            self.cols = cols

            # Initialization with dtype=object to store both numbers and pieces
            self.board = np.empty((rows, cols), dtype=object)

            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs   = currentObs

            # Filling the board with Manhattan distance-based penalties
            goal_r, goal_c = self.currentStateB
            for r in range(rows):
                for c in range(cols):
                    dist = abs(r - goal_r) + abs(c - goal_c)
                    if dist == 0:
                        self.board[r, c] = self.treasure   
                    else:
                        self.board[r, c] = -dist           # -1, -2, -3, -4, -5, ...

            # Placing obstacle at its position
            self.board[self.currentObs] = self.wall_penalization

            # Placing the King at position W
            self.board[self.currentStateW] = piece.King(True)

            self.initState = 1


    def get_environment(self):
        return self.board 

    def get_state(self):
        """
        Returns actual position of the agent (WK).
        """
        return self.currentStateW

    def print_board(self):
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = "| "
            for c in range(self.cols):
                cell_value = self.board[r, c]
                
                if (r, c) == self.currentStateW:
                    symbol = "K " 
                elif (r, c) == self.currentStateB: 
                    symbol = "100" 
                elif cell_value == -1:
                    symbol = "- "
                else:
                    symbol = "? " 
                
                row_display += f"{symbol} | " if symbol != "100" else f"{symbol}| "
            print(row_display)
            print("-" * (self.cols * 4 + 1))
        

    def is_finish(self):
        if self.currentStateB == self.currentStateW[:2]:
            return True
        return False
    
    def move_piece(self, action):
        """
        Moves the piece to the action received from the agent
        """
        current_r, current_c = self.currentStateW
        
        # Delta definitions for each action
        # (delta_row, delta_column)
        deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }
        
        # if action is not valid, do nothing
        dr, dc = deltas.get(action, (0, 0))
        
        # Calculate new position
        new_r = current_r + dr
        new_c = current_c + dc
        new_pos = (new_r, new_c)
        
        # Validation and reward determination
        
        reward = self.reward # Standard movement penalty
        done = False         # Episode completion flag

        # Boundary check
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Si surt del tauler: Penalització forta i NO es mou
            return self.currentStateW, self.wall_penalization, done

        # Goal check
        if new_pos == self.currentStateB:
            reward = self.treasure
            done = True 

        # Obstacle check
        if (self.currentObs[0] == new_r and self.currentObs[1] == new_c):
            # Si troba obstacle: Penalització forta i NO es mou
            return self.currentStateW, self.wall_penalization, done

        # If the new position is NOT the goal/obstacle/wall,

        king_obj = self.board[self.currentStateW]
        
        # Once moved, update the board:
        self.board[self.currentStateW] = -1 
        
        self.currentStateW = new_pos
        
        # Moving the King object to the new position
        self.board[new_pos] = king_obj
        
        return self.currentStateW, reward, done
    
    def reset_environment(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1), mode='default'):
        """
        Reset the environment. mode='default' keeps the original -1 fill.
        mode='init2' will initialize the board using the Manhattan-shaped values from `init2()`.
        """
        if mode == 'init2':
            prev = self.initState
            self.initState = None
            self.init2(rows=rows, cols=cols, currentStateW=currentStateW, currentStateB=currentStateB, currentObs=currentObs)
            self.initState = 1 if prev is None else prev
            return

        # Default behaviour:
        self.rows = rows
        self.cols = cols


        self.board = np.full((rows, cols), -1, dtype=object)

        self.currentStateW = currentStateW
        self.currentStateB = currentStateB
        self.currentObs = currentObs

        self.board[self.currentStateB] = self.treasure

        self.board[self.currentStateW] = piece.King(True)

        self.board[self.currentObs] = self.wall_penalization
