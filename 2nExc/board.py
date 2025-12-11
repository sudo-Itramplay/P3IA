import numpy as np
import piece as piece 

# Chess board dimensions
BOARD_SIZE = 8

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
        """Inicialitza un tauler d'escacs 8x8 amb Rei Blanc a (7,3) i Rei Negre a (0,4)."""

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            # Inicialitzem amb dtype=object per poder guardar números i peces
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = (currentStateWK, currentStateWR)
            self.currentStateB = currentStateBK


            #Posem posició dels blancs
            # Posem el Rei Blanc a la posició W
            self.board[self.currentStateW[0]] = piece.King(True)
            # Posem el Rook Blanc a la posició W1
            self.board[self.currentStateW[1]] = piece.Rook(True)
            # Posem el Rei Negre (Objectiu) a la posició B
            self.board[self.currentStateB] = piece.King(False)

            self.initState = 1


    """
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <TODO> -----------------------------------------------------------------
    FET ESTEM ELIMINANT PERO HO COMENTEM ENLLCO DELIMINARLO TAL QUAL
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    
    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates
    


    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState
    
    -----------------------------------------------------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    ------------------------------------ <\TODO> -----------------------------------------------------------------
    ###########################################################################################################
    ###########################################################################################################
    -----------------------------------------------------------------------------------------------------------
    """

    def get_environment(self):
        """Retorna la matriu del tauler actual."""
        return self.board

    def get_state(self):
        """Retorna la posició actual del rei blanc (agent)."""
        return self.currentStateW

    def getCurrentState(self):
        """Retorna l'estat complet: llista de totes les peces blanques i negres actuals al tauler."""
        pieces = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.board[r, c]
                
                # Verifiquem si la casella conté una peça vàlida
                if cell is not None and hasattr(cell, 'name'):
                    piece_id = None
                    
                    # Lògica d'assignació d'IDs segons Tipus i Color
                    # IDs estàndard: 2=Torre Blanca, 6=Rei Blanc, 8=Torre Negra, 12=Rei Negre
                    if cell.name == 'K': # Rei
                        piece_id = 6 if cell.color else 12
                    elif cell.name == 'R': # Torre
                        piece_id = 2 if cell.color else 8
                    
                    # Només afegim si hem identificat la peça
                    if piece_id is not None:
                        pieces.append([r, c, piece_id])
                        
        return pieces

    def print_board(self):
        """Mostra el tauler de joc de forma visual per consola."""
        # Set consistent width based on | S | format (3 visible characters per cell, plus start/end |)
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = ""
            for c in range(self.cols):
                cell_value = self.board[r, c]
                symbol_display = ""
                if cell_value != -1 and hasattr(cell_value, '__str__'):
                    #We use te propeerty of the peices to now the symbol, K R and its proper color
                    symbol = str(cell_value)
                    #This will print white pieces
                    if hasattr(cell_value, 'color') and cell_value.color:
                        symbol_display = f"{symbol} " 
                    else:
                        #This will print black pieces, we print 100, but could print symbol as the cell calue K in black
                        symbol_display = 100
                else:
                    #if the value of the cell is -1 we print -
                    symbol_display = "- " 
                
                # Each cell starts with a separator: | Symbol
                row_display += f"| {symbol_display}"
            
            row_display += " |" # Final separator
            print(row_display)
            print("-" * (self.cols * 4 + 1))

    def is_finish(self):
        """Comprova si el rei blanc ha arribat a la casella objectiu."""
        if self.currentStateB == self.currentStateW[:2]:
            return True
        return False
    
    def movePiece(self, start, to):
        # Crear diccionaris per trobar peces ràpidament per ID o posició
        # start i to són llistes de tuples: (fila, col, id_peça)
        
        start_dict = {p[2]: p for p in start} # Clau: ID peça, Valor: Tupla completa
        to_dict = {p[2]: p for p in to}

        # Iterar per trobar quina peça ha canviat de coordenades
        for piece_id, piece_start in start_dict.items():
            if piece_id in to_dict:
                piece_to = to_dict[piece_id]
                
                # Si les coordenades (índex 0 i 1) són diferents, aquesta és la peça moguda
                if piece_start[0:2] != piece_to[0:2]:
                    start_pos = (piece_start[0], piece_start[1])
                    to_pos = (piece_to[0], piece_to[1])
                    
                    # Recuperar l'objecte peça del tauler
                    piece_obj = self.board[start_pos[0]][start_pos[1]]
                    
                    # Realitzar el moviment físic a la matriu
                    if piece_obj != -1:
                        self.board[to_pos[0]][to_pos[1]] = piece_obj
                        self.board[start_pos[0]][start_pos[1]] = -1
                    return # Assumim que només es mou una peça per torn
    
    def reset_environment(self, rows=8, cols=8, currentStateWK=(7,3), currentStateBK=(0,4), currentStateWR=(7,0)):
        """Inicialitza un tauler d'escacs 8x8 amb Rei Blanc a (7,3) i Rei Negre a (0,4)."""

        #TODO I've comented this condition
        #if self.initState is None:
        self.rows = rows
        self.cols = cols
        
        # Inicialitzem amb dtype=object per poder guardar números i peces
        self.board = np.full((rows, cols), -1, dtype=object)
        
        self.currentStateW = (currentStateWK, currentStateWR)
        self.currentStateB = currentStateBK


        #Posem posició dels blancs
        # Posem el Rei Blanc a la posició W
        self.board[self.currentStateW[0]] = piece.King(True)
        # Posem el Rook Blanc a la posició W1
        self.board[self.currentStateW[1]] = piece.Rook(True)
        # Posem el Rei Negre (Objectiu) a la posició B
        self.board[self.currentStateB] = piece.King(False)

        self.initState = 1

    # --------------------------------------------------------------------------------------------------
    # Chess helper utilities (ported from aichess.py)
    # --------------------------------------------------------------------------------------------------
    def getPieceState(self, state, piece_id):
        """Retorna la tupla de la peça amb id donat dins l'estat o ``None`` si no hi és."""
        for p in state:
            if p[2] == piece_id:
                return p
        return None

    def getWhiteState(self, current_state):
        """Extreu les peces blanques (id 1..6) d'un estat combinat."""
        return [p for p in current_state if p[2] <= 6]

    def getBlackState(self, current_state):
        """Extreu les peces negres (id 7..12) d'un estat combinat."""
        return [p for p in current_state if p[2] > 6]

    # ---------------------------- Move generation primitives ----------------------------
    def _occupied_maps(self, my_pieces, rival_pieces):
        """Construeix mapes d'ocupació per casella (fila,col) de peces pròpies i rivals."""
        own = {(r, c): pid for r, c, pid in my_pieces}
        rivals = {(r, c): pid for r, c, pid in rival_pieces}
        return own, rivals

    def _king_moves(self, r, c):
        """Genera caselles adjacents legals per a un rei (casella = parell (fila,col))."""
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                yield nr, nc

    def _rook_moves(self, r, c, own_map, rival_map):
        """Genera caselles en línia recta per a una torre fins topar o capturar."""
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
        """Calcula moviments pseudo-legals per a un rei o una torre donat l'estat actual."""
        r, c, pid = piece_tuple
        own_map, rival_map = self._occupied_maps(my_pieces, rival_pieces)
        moves = []

        if pid % 6 == 0 or pid % 6 == 6:  # Kings (6, 12)
            for nr, nc in self._king_moves(r, c):
                if (nr, nc) not in own_map:
                    moves.append([nr, nc, pid])

        elif pid % 6 == 2:  # Rooks (2, 8)
            for nr, nc in self._rook_moves(r, c, own_map, rival_map):
                if (nr, nc) not in own_map:
                    moves.append([nr, nc, pid])

        # De moment només suportem finals amb rei i torre; ampliar segons calgui
        return moves

    def getListNextStatesW(self, my_state, rival_state=None):
        """Genera successors pseudo-legals per a les blanques i retorna llistes de peces actualitzades."""
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
        """Genera successors pseudo-legals per a les negres i retorna llistes de peces actualitzades."""
        rival_state = rival_state or []
        next_states = []
        for idx, p in enumerate(my_state):
            candidate_positions = self._generate_moves_for_piece(p, my_state, rival_state)
            for new_pos in candidate_positions:
                new_state = my_state.copy()
                new_state[idx] = new_pos
                next_states.append(new_state)
        return next_states

    def getNextPositions(self, state, rival_state=None):
        """Retorna la llista de caselles accessibles (fila,col) per a una peça concreta."""
        if state is None:
            return None
        if state[2] > 6:
            next_states = self.getListNextStatesB([state], rival_state)
        else:
            next_states = self.getListNextStatesW([state], rival_state)
        return [s[0][0:2] for s in next_states]

    # ---------------------------- Legality helpers ----------------------------
    def verify_single_piece_moved(self, state_before, state_after):
        """Verifica que només una peça del jugador hagi canviat de casella entre dos estats."""
        moved_pieces = 0
        for piece_before in state_before:
            if piece_before not in state_after:
                moved_pieces += 1
        return moved_pieces == 1

    def getMovement(self, state, next_state):
        """Identifica la peça que s'ha mogut entre dos estats (origen i destí)."""
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
        """Comprova si una casella (fila,col) està atacada per alguna torre segons ocupació."""
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
        """Comprova si una casella (fila,col) és adjacient a un rei rival (atac de rei)."""
        sr, sc = square
        for kr, kc, _ in kings:
            if max(abs(sr - kr), abs(sc - kc)) == 1:
                return True
        return False

    def isWatchedBk(self, currentState):
        # 1. Get the Black King's position and ensure it is a Numpy array
        # This solves the "tuple implies no subtraction" error and enables math operations
        bkPosition = np.array(self.getPieceState(currentState, 12)[0:2])
        
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # --- WHITE KING (WK) LOGIC ---
        if wkState is not None:
            # Extract WK position (assuming the first two elements are x, y)
            wkPosition = np.array(wkState[0:2])
            
            # Calculate distance between kings (Chebyshev distance)
            # max(|x1 - x2|, |y1 - y2|)
            dist = np.max(np.abs(wkPosition - bkPosition))

            # Check if the White King is within distance 2 of the Black King
            if dist <= 2:
                return True

        # --- WHITE ROOK (WR) LOGIC ---
        if wrState is not None:
            # Extract WR position
            wrPosition = np.array(wrState[0:2])
            
            # Check if they share the same row or column
            # wrPosition[0] refers to the row/x, wrPosition[1] refers to the col/y
            if wrPosition[0] == bkPosition[0] or wrPosition[1] == bkPosition[1]:
                return True

        return False

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
        """Retorna totes les transicions legals pel torn actual (color True=blanc)."""
        if color:  # white
            current_player_pieces = self.getWhiteState(current_state)
            rival_pieces = self.getBlackState(current_state)
            next_moves_raw = self.getListNextStatesW(current_player_pieces, rival_pieces)
        else:
            #current_player_pieces = self.getBlackState(current_state)
            rival_pieces = self.getWhiteState(current_state)
            next_moves_raw = self.getListNextStatesB(current_player_pieces, rival_pieces)

        legal_next_states = []
        for next_move_pieces in next_moves_raw:
            is_legal, next_node = self.is_legal_transition(current_player_pieces, rival_pieces, next_move_pieces, color)
            if is_legal:
                legal_next_states.append(next_node)
        return legal_next_states
