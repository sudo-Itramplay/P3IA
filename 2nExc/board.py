import numpy as np
import piece as piece 

# Chess board dimensions
BOARD_SIZE = 8
BOLD  = "\033[1m"
RESET = "\033[0m"

class Board:
    initState = None
    
    rows = 0
    cols = 0
        
    board = []
    # coord Agent (Rei Blanc)
    currentStateW = ()
    # coord Recompensa final (Objectiu)
    currentStateB = ()
    # Recompensa per moviment (pas)
    reward = -1
    # Bonificació final
    treasure = 100
    # Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=8, cols=8, currentStateWK=(7,3), currentStateBK=(0,4), currentStateWR=(7,0)):

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = (currentStateWK, currentStateWR)
            self.currentStateB = currentStateBK

            self.board[self.currentStateW[0]] = piece.King(True)
            self.board[self.currentStateW[1]] = piece.Rook(True)
            self.board[self.currentStateB] = piece.King(False)

            self.initState = 1

    def get_environment(self):
        return self.board

    def get_state(self):
        """Returns Actual White positions."""
        return self.currentStateW

    def getCurrentState(self):
        """
        Returns a list of pieces with their positions and IDs.
        """
        pieces = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r, c]
                

                if cell is not None and hasattr(cell, 'name'):
                    piece_id = None
                    
                    if cell.name == 'K': 
                        piece_id = 6 if cell.color else 12
                    elif cell.name == 'R': 
                        piece_id = 2 if cell.color else 8
                    
                    if piece_id is not None:
                        pieces.append([r, c, piece_id])
                        
        return pieces

    def print_board(self):
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = ""
            for c in range(self.cols):
                cell_value = self.board[r, c]
                symbol_display = ""
                if cell_value != -1 and hasattr(cell_value, '__str__'):

                    symbol = str(cell_value)
                    #This will print white pieces
                    if hasattr(cell_value, 'color') and cell_value.color:
                        symbol_display = f"{BOLD}{symbol}{RESET} " 
                    else:
                        #This will print black pieces
                        symbol_display = f"{RESET}{symbol}{RESET} "

                else:
                    #if the value of the cell is -1 we print -
                    symbol_display = "- " 
                
                # Each cell starts with a separator: | Symbol
                row_display += f"| {symbol_display}"
            
            row_display += " |" # Final separator
            print(row_display)
            print("-" * (self.cols * 4 + 1))

    def is_finish(self):
        """
        Check if the white king has reached the black king's position.
        """
        if self.currentStateB == self.currentStateW[:2]:
            return True
        return False
    
    def movePiece(self, start, to):
        # Creates dictionaries for quick lookup
        
        start_dict = {p[2]: p for p in start}
        to_dict = {p[2]: p for p in to}

        # Find the moved piece by comparing start and to states
        for piece_id, piece_start in start_dict.items():
            if piece_id in to_dict:
                piece_to = to_dict[piece_id]

                # If the position has changed, update the board
                if piece_start[0:2] != piece_to[0:2]:
                    start_pos = (piece_start[0], piece_start[1])
                    to_pos = (piece_to[0], piece_to[1])
                    
                    piece_obj = self.board[start_pos[0]][start_pos[1]]
                    
                    if piece_obj != -1:
                        self.board[to_pos[0]][to_pos[1]] = piece_obj
                        self.board[start_pos[0]][start_pos[1]] = -1
                    return 
    
    def reset_environment(self, rows=8, cols=8, currentStateWK=(7,3), currentStateBK=(0,4), currentStateWR=(7,0)):

        self.rows = rows
        self.cols = cols
        

        self.board = np.full((rows, cols), -1, dtype=object)
        
        self.currentStateW = (currentStateWK, currentStateWR)
        self.currentStateB = currentStateBK


        self.board[self.currentStateW[0]] = piece.King(True)
        self.board[self.currentStateW[1]] = piece.Rook(True)
        self.board[self.currentStateB] = piece.King(False)

        self.initState = 1

    # --------------------------------------------------------------------------------------------------
    # Chess helper utilities
    # --------------------------------------------------------------------------------------------------
    def getPieceState(self, state, piece_id):
        """Returns the position and ID of a specific piece from a combined state."""
        for p in state:
            if p[2] == piece_id:
                return p
        return None

    def getWhiteState(self, current_state):
        """Gets the white pieces (id 1..6) from a combined state."""
        return [p for p in current_state if p[2] <= 6]

    def getBlackState(self, current_state):
        """Gets the black pieces (id 7..) from a combined state."""
        return [p for p in current_state if p[2] > 6]




    # ---------------------------- Move generation primitives ----------------------------
    def _occupied_maps(self, my_pieces, rival_pieces):
        """Maps of occupied squares for own and rival pieces."""
        own = {(r, c): pid for r, c, pid in my_pieces}
        rivals = {(r, c): pid for r, c, pid in rival_pieces}
        return own, rivals

    def _king_moves(self, r, c):
        """Generates moves for a king."""
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                yield nr, nc

    def _rook_moves(self, r, c, own_map, rival_map):
        """Generates moves for a rook."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if (nr, nc) in own_map:
                    break
                yield nr, nc
                if (nr, nc) in rival_map:
                    break
                nr += dr
                nc += dc

    def _generate_moves_for_piece(self, piece_tuple, my_pieces, rival_pieces):
        """Generates pseudo-legal moves for a given piece."""
        r, c, pid = piece_tuple
        own_map, rival_map = self._occupied_maps(my_pieces, rival_pieces)
        moves = []

        if pid % 6 == 0 or pid % 6 == 6:  
            for nr, nc in self._king_moves(r, c):
                if (nr, nc) not in own_map:
                    moves.append([nr, nc, pid])

        elif pid % 6 == 2:  
            for nr, nc in self._rook_moves(r, c, own_map, rival_map):
                if (nr, nc) not in own_map:
                    moves.append([nr, nc, pid])

        return moves

    def getListNextStatesW(self, my_state, rival_state=None):
        """Generates pseudo-legal successors for white and returns updated piece lists."""
        rival_state = rival_state or []
        next_states = []
        for idx, p in enumerate(my_state):
            candidate_positions = self._generate_moves_for_piece(p, my_state, rival_state)
            for new_pos in candidate_positions:
                new_state = my_state.copy()
                new_state[idx] = new_pos
                next_states.append(new_state)
        return next_states

    def getListNextStatesB(self, my_state, rival_state=None):
        """Generates pseudo-legal successors for black and returns updated piece lists."""
        rival_state = rival_state or []
        next_states = []
        for idx, p in enumerate(my_state):
            candidate_positions = self._generate_moves_for_piece(p, my_state, rival_state)
            for new_pos in candidate_positions:
                new_state = list(my_state)
                new_state[idx] = new_pos
                next_states.append(new_state)
        return next_states

    def getNextPositions(self, state, rival_state=None):
        """Returns possible next positions for the player to move."""
        if state is None:
            return None
        if state[2] > 6:
            next_states = self.getListNextStatesB([state], rival_state)
        else:
            next_states = self.getListNextStatesW([state], rival_state)
        return [s[0][0:2] for s in next_states]

    # ---------------------------- Legality helpers ----------------------------
    def verify_single_piece_moved(self, state_before, state_after):
        """Verifies that exactly one piece has moved between two states."""
        moved_pieces = 0
        for piece_before in state_before:
            if piece_before not in state_after:
                moved_pieces += 1
        return moved_pieces == 1

    def getMovement(self, state, next_state):
        """Identifies which piece moved between two states."""
        piece_state = None
        piece_next_state = None
        for piece in state:
            if piece not in next_state:
                moved_piece = piece[2]
                piece_next = self.getPieceState(next_state, moved_piece)
                if piece_next is not None:
                    piece_state = piece
                    piece_next_state = piece_next
                    break
        return [piece_state, piece_next_state]

    def _is_square_attacked_by_rook(self, square, rooks, occupancy):
        """Checks if a square (row,col) is attacked by any rival rook (rook attack)."""
        sr, sc = square
        for rr, rc, _ in rooks:
            if rr == sr:
                step = 1 if rc < sc else -1
                blocked = any(((rr, cc) in occupancy) for cc in range(rc + step, sc, step))
                if not blocked:
                    return True
            if rc == sc:
                step = 1 if rr < sr else -1
                blocked = any(((rrr, rc) in occupancy) for rrr in range(rr + step, sr, step))
                if not blocked:
                    return True
        return False

    def _is_square_attacked_by_king(self, square, kings):
        """Checks if a square (row,col) is attacked by any rival king (king attack)."""
        sr, sc = square
        for kr, kc, _ in kings:
            if max(abs(sr - kr), abs(sc - kc)) == 1:
                return True
        return False

    def isWatchedBk(self, current_state):
        """Tells if the black king is in check (square attacked by white king/rook)."""
        bk = self.getPieceState(current_state, 12)
        if bk is None:
            return False
        white_state = self.getWhiteState(current_state)
        wk = self.getPieceState(current_state, 6)
        wr = self.getPieceState(current_state, 2)
        white_rooks = [wr] if wr else []
        white_kings = [wk] if wk else []
        occupancy = {(r, c) for r, c, _ in current_state}
        return self._is_square_attacked_by_king(bk[0:2], white_kings) or self._is_square_attacked_by_rook(bk[0:2], white_rooks, occupancy)

    def isWatchedWk(self, current_state):
        """Indica si el rei blanc està en escac (casella atacada per rei/torre negra)."""
        wk = self.getPieceState(current_state, 6)
        if wk is None:
            return False
        black_state = self.getBlackState(current_state)
        bk = self.getPieceState(current_state, 12)
        br = self.getPieceState(current_state, 8)
        black_rooks = [br] if br else []
        black_kings = [bk] if bk else []
        occupancy = {(r, c) for r, c, _ in current_state}
        return self._is_square_attacked_by_king(wk[0:2], black_kings) or self._is_square_attacked_by_rook(wk[0:2], black_rooks, occupancy)

    def is_legal_transition(self, current_player_state, rival_state, moves, color):
        """Construeix l'estat següent i valida moviment únic i rei propi fora d'escac."""
        if not self.verify_single_piece_moved(current_player_state, moves):
            return False, None

        move_info = self.getMovement(current_player_state, moves)
        if move_info[0] is None or move_info[1] is None:
            return False, None

        new_pos_coords = move_info[1][0:2]
        new_rival_state = [p for p in rival_state if p[0:2] != new_pos_coords]

        next_node = moves + new_rival_state

        if color and self.isWatchedWk(next_node):
            return False, None
        if (not color) and self.isWatchedBk(next_node):
            return False, None

        return True, next_node

    def get_all_next_states(self, current_state, color):
        """Returns all possible legal next states for the given color."""
        if color:  # white
            current_player_pieces = self.getWhiteState(current_state)
            rival_pieces = self.getBlackState(current_state)
            next_moves_raw = self.getListNextStatesW(current_player_pieces, rival_pieces)
        else:
            current_player_pieces = self.getBlackState(current_state)
            rival_pieces = self.getWhiteState(current_state)
            next_moves_raw = self.getListNextStatesB(current_player_pieces, rival_pieces)

        legal_next_states = []
        for next_move_pieces in next_moves_raw:
            is_legal, next_node = self.is_legal_transition(current_player_pieces, rival_pieces, next_move_pieces, color)
            if is_legal:
                legal_next_states.append(next_node)
        return legal_next_states
