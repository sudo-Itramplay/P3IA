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
    # coord obstacle
    currentObs = ()
    # Recompensa per moviment (pas)
    reward = -1
    # Bonificació final
    treasure = 100
    # Wall penalization
    wall_penalization = -100
    
    def __init__(self, rows=8, cols=8, currentStateW=(7,3), currentStateB=(0,4), currentObs=None):
        """Inicialitza un tauler d'escacs 8x8 amb Rei Blanc a (7,3) i Rei Negre a (0,4)."""

        if self.initState is None:
            self.rows = rows
            self.cols = cols
            
            # Inicialitzem amb dtype=object per poder guardar números i peces
            self.board = np.full((rows, cols), -1, dtype=object)
            
            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs = currentObs

            # Posem el Rei Blanc a la posició W
            self.board[self.currentStateW] = piece.King(True)
            
            # Posem el Rei Negre (Objectiu) a la posició B
            self.board[self.currentStateB] = piece.King(False)

            # Posem obstacle si es proporciona
            if self.currentObs is not None:
                self.board[self.currentObs] = self.wall_penalization

            self.initState = 1

    def init2(self, rows=3, cols=4,currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1)):     
        """Inicialitza amb recompensa manhattan negativa cap al tresor i obstacle."""

        if self.initState is None:
            self.rows = rows
            self.cols = cols

            # Tauler amb números (recompenses) + peces
            self.board = np.empty((rows, cols), dtype=object)

            self.currentStateW = currentStateW
            self.currentStateB = currentStateB
            self.currentObs   = currentObs

            # Omplim amb recompensa = -distància Manhattan fins al tresor
            goal_r, goal_c = self.currentStateB
            for r in range(rows):
                for c in range(cols):
                    dist = abs(r - goal_r) + abs(c - goal_c)
                    if dist == 0:
                        self.board[r, c] = self.treasure
                    else:
                        self.board[r, c] = -dist

            # Posem obstacle (casella grisa)
            self.board[self.currentObs] = self.wall_penalization

            # Posem el Rei a la posició W
            self.board[self.currentStateW] = piece.King(True)

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
                # Si la casella té una peça (King object), registrem-la
                if isinstance(cell, object) and hasattr(cell, 'name'):
                    # Obtenim l'id de la peça (6=White King, 12=Black King, etc.)
                    piece_id = 6 if (hasattr(cell, 'color') and cell.color) else 12
                    pieces.append([r, c, piece_id])
        return pieces

    def print_board(self):
        """Mostra el tauler de joc de forma visual per consola."""
        print("-" * (self.cols * 4 + 1))
        for r in range(self.rows):
            row_display = "| "
            for c in range(self.cols):
                cell_value = self.board[r, c]
                
                if (r, c) == self.currentStateW:
                    symbol = "K " 
                elif (r, c) == self.currentStateB: #Millor comprovar coord que valor
                    symbol = "100" 
                elif cell_value == -1:
                    symbol = "- "
                else:
                    symbol = "? " 
                
                row_display += f"{symbol} | " if symbol != "100" else f"{symbol}| "
            print(row_display)
            print("-" * (self.cols * 4 + 1))
        

    def is_finish(self):
        """Comprova si el rei blanc ha arribat a la casella objectiu."""
        if self.currentStateB == self.currentStateW[:2]:
            return True
        return False
    
    def move_piece(self, action):
        """Mou el rei blanc segons l'acció (0 amunt, 1 avall, 2 dreta, 3 esquerra)."""
        current_r, current_c = self.currentStateW
        
        # 1. Definició de Deltas (Canvi de coordenades segons l'acció)
        # Format: (delta_fila, delta_columna)
        deltas = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, 1),   # Right
            3: (0, -1)   # Left
        }
        
        # Recuperem el desplaçament. Si l'acció no existeix, no ens movem (0,0)
        dr, dc = deltas.get(action, (0, 0))
        
        # Calculem la proposta de nova posició
        new_r = current_r + dr
        new_c = current_c + dc
        new_pos = (new_r, new_c)
        
        # --- 2. Validacions i Lògica de Moviment ---
        
        reward = self.reward # Cost per defecte (-1)
        done = False         # Per defecte no hem acabat

        # A) Validació de límits del tauler (Murs)
        if not (0 <= new_r < self.rows and 0 <= new_c < self.cols):
            # Si surt del tauler: Penalització forta i NO es mou
            return self.currentStateW, self.wall_penalization, done

        # B) Comprovem si hem arribat a l'objectiu
        if new_pos == self.currentStateB:
            reward = self.treasure
            done = True # Episodi acabat

        # C) Validació de Obstacle
        if (self.currentObs[0] == new_r and self.currentObs[1] == new_c):
            # Si troba obstacle: Penalització forta i NO es mou
            return self.currentStateW, self.wall_penalization, done

        # --- 3. Actualització del Tauler (Física del moviment) ---

        # Recuperem l'objecte Rei
        king_obj = self.board[self.currentStateW]
        
        # Buidem la casella antiga
        self.board[self.currentStateW] = -1 
        
        # Actualitzem coordenades internes
        self.currentStateW = new_pos
        
        # Posem el Rei a la nova casella (visualització)
        # Nota: Si és l'objectiu, tècnicament el 'mengem', però visualment posem el rei igualment
        self.board[new_pos] = king_obj
        
        return self.currentStateW, reward, done
    
    def reset_environment(self, rows=3, cols=4, currentStateW=(2,0), currentStateB=(0,3), currentObs=(1,1), mode='default'):
        """Reinicia l'entorn; `mode='init2'` aplica recompenses Manhattan, `default` omple amb -1."""
        # If caller requested the init2 shaped initialization, reuse init2()
        if mode == 'init2':
            # init2 only runs when self.initState is None, so temporarily clear it
            prev = self.initState
            self.initState = None
            self.init2(rows=rows, cols=cols, currentStateW=currentStateW, currentStateB=currentStateB, currentObs=currentObs)
            # restore initState flag
            self.initState = 1 if prev is None else prev
            return

        # Default behaviour: simple -1 fill and place special cells
        self.rows = rows
        self.cols = cols

        # 1. Inicialitzem amb dtype=object per poder guardar números I peces
        self.board = np.full((rows, cols), -1, dtype=object)

        self.currentStateW = currentStateW
        self.currentStateB = currentStateB
        self.currentObs = currentObs

        # 2. Posem el valor 100 a la posició B
        self.board[self.currentStateB] = self.treasure

        # 3. Posem el King a la posició W
        self.board[self.currentStateW] = piece.King(True)

        # 4. Posem obstacle
        self.board[self.currentObs] = self.wall_penalization

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

    def isWatchedBk(self, current_state):
        """Indica si el rei negre està en escac (casella atacada per rei/torre blanca)."""
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
        """Retorna totes les transicions legals pel torn actual (color True=blanc)."""
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
