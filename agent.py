import heapq
import copy
import pickle

# Check if point is within board size
def size_check(point):
    if 0 <= point[0] < BOARD_SIZE and 0 <= point[1] < BOARD_SIZE:
        return True
    return False


def get_heuristic_value_jump(next_jump, cur):
    dmin = min(abs(next_jump[0] - cur[0]), abs(next_jump[1] - cur[1]))
    dmax = max(abs(next_jump[0] - cur[0]), abs(next_jump[1] - cur[1]))
    return int(1.4 * dmin + (dmax - dmin))


BOARD_SIZE = 16
INITIAL_NUMBER = 9

n_value = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

black_final_locations = [[11, 14], [11, 15], [12, 13], [12, 14], [12, 15], [13, 12], [13, 13], [13, 14],
                         [13, 15], [14, 11], [14, 12], [14, 13], [14, 14], [14, 15], [15, 11], [15, 12], [15, 13],
                         [15, 14], [15, 15]]
white_final_locations = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1],
                         [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [4, 0], [4, 1]]

initial_black_moves = [[[0, 3], [2, 5]], [[1, 4], [3, 6]], [[0, 2], [4, 6]], [[0, 0], [2, 4]], [[1, 1], [5, 7]],
                       [[2, 3], [3, 4]], [[0, 1], [6, 7]], [[3, 0], [7, 6]], [[0, 4], [1, 5]]]

initial_white_moves = [[[15, 12], [13, 10]], [[14, 11], [12, 9]], [[15, 13], [11, 9]], [[15, 15], [13, 11]],
                       [[14, 14], [10, 8]], [[13, 12], [12, 11]], [[15, 14], [9, 8]], [[12, 15], [8, 9]],
                       [[15, 11], [14, 10]]]

diagonal_locations = [[2, 4], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [5, 3], [5, 4], [5, 5],
                      [5, 6], [5, 7], [5, 8], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [7, 5], [7, 6], [7, 7],
                      [7, 8], [7, 9], [7, 10], [8, 6], [8, 7], [8, 8], [8, 9], [8, 10], [8, 11], [9, 7], [9, 8], [9, 9],
                      [9, 10], [9, 11], [9, 12], [10, 8], [10, 9], [10, 10], [10, 11], [10, 12], [10, 13], [11, 9],
                      [11, 10], [11, 11], [11, 12], [11, 13]]

black_initial_diagonal = [[1, 1], [1, 2], [2, 1], [2, 3], [3, 2]]
white_initial_diagonal = [[12, 13], [13, 13], [13, 14], [14, 13], [14, 14]]

black_final_pos = [15, 15]
white_final_pos = [0, 0]

h_val = [[6], [12], [20], [20]]
h_weight = 0
max_moves_count = 40
min_moves_count = 20


class HalmaGamePlayer:

    def __init__(self, all_config, selected_moves, my_goal_dict):
        self.player = all_config.player
        self.opp_player = all_config.opp_player
        self.board = all_config.board
        self.temp_board = 0
        self.selected_moves = selected_moves
        self.my_goal_dict = my_goal_dict

    def generate_output_path(self, cur_pos, next_pos):

        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if neighbor == next_pos:
                return [
                    'E ' + str(cur_pos[1]) + ',' + str(cur_pos[0]) + ' ' + str(next_pos[1]) + ',' + str(next_pos[0])]

        visited = {}
        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if size_check(neighbor) and self.board[neighbor[0]][neighbor[1]] != 0:
                next_jump = [neighbor[0] + n_value[idx][0], neighbor[1] + n_value[idx][1]]
                if size_check(next_jump) and self.board[next_jump[0]][next_jump[1]] == 0:
                    visited[tuple(next_jump)] = [cur_pos[0], cur_pos[1]]
                    visited[tuple([cur_pos[0], cur_pos[1]])] = None
                    if next_jump == next_pos:
                        break
                    open_queue = [[get_heuristic_value_jump(next_jump, cur_pos), next_jump, cur_pos]]
                    self.temp_board = copy.deepcopy(self.board)
                    while open_queue:
                        h, cur, old = heapq.heappop(open_queue)
                        if cur == next_pos:
                            break

                        self.temp_board[old[0]][old[1]] = 0
                        self.temp_board[cur[0]][cur[1]] = self.player

                        for idx2 in range(len(n_value)):
                            temp_cur = [cur[0] + n_value[idx2][0], cur[1] + n_value[idx2][1]]
                            next_jump = [temp_cur[0] + n_value[idx2][0], temp_cur[1] + n_value[idx2][1]]
                            if size_check(temp_cur) and self.temp_board[temp_cur[0]][
                                temp_cur[1]] != 0 and size_check(next_jump) and (
                                    self.temp_board[next_jump[0]][next_jump[1]] == 0) and (
                                    tuple(next_jump) not in visited):
                                heuristic = get_heuristic_value_jump(next_jump, cur)
                                heapq.heappush(open_queue, [heuristic, next_jump, cur])
                                visited[tuple(next_jump)] = cur

        if not visited:
            return ''
        temp_var = next_pos
        output_path = []
        while temp_var is not None:
            t1 = visited[tuple(temp_var)]
            if t1 is not None:
                output_path.append(
                    'J ' + str(t1[1]) + ',' + str(t1[0]) + ' ' + str(temp_var[1]) + ',' + str(temp_var[0]))
            temp_var = t1
        return output_path

    def validate_next_move(self, cur_pos, next_pos):

        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if neighbor == next_pos:
                return True

        visited = {}
        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if size_check(neighbor) and self.board[neighbor[0]][neighbor[1]] != 0:
                next_jump = [neighbor[0] + n_value[idx][0], neighbor[1] + n_value[idx][1]]
                if size_check(next_jump) and self.board[next_jump[0]][next_jump[1]] == 0:
                    visited[tuple(next_jump)] = [cur_pos[0], cur_pos[1]]
                    visited[tuple([cur_pos[0], cur_pos[1]])] = None
                    if next_jump == next_pos:
                        return True
                    open_queue = [[get_heuristic_value_jump(next_jump, cur_pos), next_jump, cur_pos]]
                    self.temp_board = copy.deepcopy(self.board)
                    while open_queue:
                        h, cur, old = heapq.heappop(open_queue)
                        if cur == next_pos:
                            return True

                        self.temp_board[old[0]][old[1]] = 0
                        self.temp_board[cur[0]][cur[1]] = self.player

                        for idx2 in range(len(n_value)):
                            temp_cur = [cur[0] + n_value[idx2][0], cur[1] + n_value[idx2][1]]
                            next_jump = [temp_cur[0] + n_value[idx2][0], temp_cur[1] + n_value[idx2][1]]
                            if size_check(temp_cur) and self.temp_board[temp_cur[0]][
                                temp_cur[1]] != 0 and size_check(next_jump) and (
                                    self.temp_board[next_jump[0]][next_jump[1]] == 0) and (
                                    tuple(next_jump) not in visited):
                                heuristic = get_heuristic_value_jump(next_jump, cur)
                                heapq.heappush(open_queue, [heuristic, next_jump, cur])
                                visited[tuple(next_jump)] = cur

        return False

    # Check if all pawns are on opposite side, maintain list of pawn locations on each side to verify
    def is_terminal_state(self):
        my_count, opp_count = 0, 0
        if self.player == 1:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == 1 and ([i, j] in black_final_locations):
                        my_count += 1
                    if self.board[i][j] == 2 and ([i, j] in white_final_locations):
                        opp_count += 1

        elif self.player == 2:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == 1 and ([i, j] in black_final_locations):
                        opp_count += 1
                    if self.board[i][j] == 2 and ([i, j] in white_final_locations):
                        my_count += 1

        if my_count == 19 or opp_count == 19:
            return True
        return False

    def set_my_goal(self):
        goals = {}
        if self.player == 1:
            initial = black_final_locations
            final = white_final_locations
            final_pos = [0, 0]
        else:
            initial = white_final_locations
            final = black_final_locations
            final_pos = [15, 15]

        final_flag = False
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.player and ([i, j] in initial):
                    final_flag = True
                    break
        if final_flag:
            pawn_count = 0
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == self.player and pawn_count < 19:
                        goals[(i, j)] = final[pawn_count]
                        pawn_count += 1
        else:
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == self.player:
                        goals[(i, j)] = final_pos

        return goals

    # Number of my pawns in goal and opponent pawns in goal
    def calculate_utility_value(self):
        my_count, opp_count = 0, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.player:
                    my_count += 1
                elif self.board[i][j] == self.opp_player:
                    opp_count += 1
        if my_count == 19:
            return 200000
        elif opp_count == 19:
            return -200000
        else:
            return 0

    def get_final_heuristic_value(self):
        r_sum, dmin, dmax = 0, 0, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.player:
                    if (i, j) in self.my_goal_dict:
                        final_goal = self.my_goal_dict[(i, j)]
                    elif self.player == 1:
                        final_goal = white_final_pos
                    else:
                        final_goal = black_final_pos

                    if self.player == 1:
                        if [i, j] in black_final_locations:
                            r_sum += 90
                            if h_weight == 2:
                                r_sum += 20

                        dmin = min((i - final_goal[0]), (j - final_goal[1]))
                        dmax = max((i - final_goal[0]), (j - final_goal[1]))

                        if i >= 7:
                            r_sum += 20
                        if [i, j] in black_initial_diagonal:
                            r_sum += 10

                    elif self.player == 2:
                        if [i, j] in white_final_locations:
                            r_sum += 90
                            if h_weight == 2:
                                r_sum += 20

                        dmin = min((final_goal[0] - i), (final_goal[1] - j))
                        dmax = max((final_goal[0] - i), (final_goal[1] - j))
                        if i <= 7:
                            r_sum += 20
                        if [i, j] in white_initial_diagonal:
                            r_sum += 10

                    r_sum += 15 * int(1.4 * dmin + (dmax - dmin))

                    if [i, j] in diagonal_locations:
                        r_sum += 40

        return r_sum

    def get_heuristic_value(self, next_pos, cur, pid):
        if pid:
            temp_turn = self.player
        else:
            temp_turn = self.opp_player

        if tuple(cur) in self.my_goal_dict:
            final_goal = self.my_goal_dict[tuple(cur)]
        elif temp_turn == 1:
            final_goal = white_final_pos
        else:
            final_goal = black_final_pos

        # Distance from final location
        dmin = min(abs(next_pos[0] - final_goal[0]), abs(next_pos[1] - final_goal[1]))
        dmax = max(abs(next_pos[0] - final_goal[0]), abs(next_pos[1] - final_goal[1]))

        h_sum = int(1.4 * dmin + (dmax - dmin))

        if ((cur[0], cur[1]) in self.selected_moves) and (self.selected_moves[(cur[0], cur[1])] == next_pos):
            h_sum -= 30

        # Displacement between moves
        if temp_turn == 1:
            dmin = min((next_pos[0] - cur[0]), (next_pos[1] - cur[1]))
            dmax = max((next_pos[0] - cur[0]), (next_pos[1] - cur[1]))
        else:
            dmin = min((cur[0] - next_pos[0]), (cur[1] - next_pos[1]))
            dmax = max((cur[0] - next_pos[0]), (cur[1] - next_pos[1]))

        h_sum += int(1.4 * dmin + (dmax - dmin))

        # Other criteria for Black pawn
        if temp_turn == 1:

            if next_pos in black_initial_diagonal:
                h_sum += 5

            if h_weight == 1 and cur[0] < 7 and (next_pos[1] - cur[1] > 0) and (next_pos[0] - cur[0]) > 0:
                h_sum += 30

            # Moving down and right and location in diagonal location
            if (next_pos[0] - cur[0]) > 0 and (next_pos[1] - cur[1] > 0) and next_pos in diagonal_locations:
                h_sum += 70

            # Crossing the centre
            if cur[0] < 7 and next_pos[0] >= 7:
                h_sum += 80
                # moving towards right
                if next_pos[1] > cur[1]:
                    h_sum += 10

            # Moving towards centre
            elif cur[0] < 7 and (7 - next_pos[0] < 7 - cur[0]):
                h_sum += 30

            elif next_pos[0] < 7 and cur[0] >= 7:
                h_sum -= 70

            if cur[0] < 7 and (next_pos[0] < 7) and (7 - next_pos[0] < 7 - cur[0]) and (next_pos[1] - cur[1] > 0):
                h_sum += 20

            # movement within final positions
            if (h_weight >= 1) and (cur in black_final_locations) and next_pos in black_final_locations:
                h_sum -= 40

            elif (cur in black_final_locations) and next_pos in black_final_locations:
                h_sum -= 30

            if cur in black_final_locations and next_pos not in black_final_locations:
                h_sum -= 200

            # add extra value if pawns are moved out of initial position
            if cur in white_final_locations and next_pos not in white_final_locations:
                h_sum += 500

            if (cur in white_final_locations) and (next_pos in white_final_locations) and (
                    (next_pos[0] - cur[0]) >= 0) and ((next_pos[1] - cur[1]) >= 0):
                h_sum += 200

            # Entering final positions
            if (cur not in black_final_locations) and (next_pos in black_final_locations):
                h_sum += 75


        # Other criteria for white pawn
        elif temp_turn == 2:

            if h_weight == 1 and (cur[0] < 7) and (cur[1] - next_pos[1] > 0) and (cur[0] - next_pos[0] > 0):
                h_sum += 30

            if next_pos in white_initial_diagonal:
                h_sum += 5

            if (cur[0] - next_pos[0] > 0) and (cur[1] - next_pos[1] > 0) and next_pos in diagonal_locations:
                h_sum += 70

            if cur[0] > 7 and next_pos[0] <= 7:
                h_sum += 80
                # moving towards left
                if next_pos[1] < cur[1]:
                    h_sum += 10

            elif cur[0] > 7 and (next_pos[0] - 7 < cur[0] - 7):
                h_sum += 30

            elif next_pos[0] > 7 and cur[0] <= 7:
                h_sum -= 70

            if (cur[0] > 7) and (next_pos[0] < 7) and (next_pos[0] - 7 < cur[0] - 7) and (cur[1] - next_pos[1] > 0):
                h_sum += 20

            # movement within final positions
            if (h_weight >= 1) and (cur in white_final_locations) and next_pos in white_final_locations:
                h_sum -= 40

            elif (cur in white_final_locations) and next_pos in white_final_locations:
                h_sum -= 30

            if cur in white_final_locations and next_pos not in white_final_locations:
                h_sum -= 200

            # add extra value if pawns are moved out of initial position
            if cur in black_final_locations and next_pos not in black_final_locations:
                h_sum += 500

            if cur in black_final_locations and next_pos in black_final_locations and (cur[0] - next_pos[0] >= 0) and (
                    cur[1] - next_pos[1] >= 0):
                h_sum += 200

            # Entering final positions
            if (cur not in white_final_locations) and (next_pos in white_final_locations):
                h_sum += 75


        return h_sum

    # Generate all single and jump moves
    def generate_all_moves(self, pid):
        if pid:
            temp_p = self.player
        else:
            temp_p = self.opp_player

        if temp_p == 1:
            initial = white_final_locations
            final = black_final_locations

        else:
            initial = black_final_locations
            final = white_final_locations

        all_jump_moves = []
        all_adj_moves = []
        initial_moves = []
        initial_camp_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == temp_p:

                    # Generate all adjacent moves
                    for idx in range(len(n_value)):
                        neighbor = [i + n_value[idx][0], j + n_value[idx][1]]
                        if size_check(neighbor) and (self.board[neighbor[0]][neighbor[1]] == 0) and (
                                not (([i, j] not in initial) and (neighbor in initial) or (([i, j] in final) and (neighbor not in final)))):
                            move_heuristic = self.get_heuristic_value(neighbor, [i, j], pid)

                            if ([i, j] in initial) and (neighbor in initial) and (
                                    (self.player == 1 and (neighbor[0] - i) > 0 and (neighbor[1] - j >= 0)) or (
                                    self.player == 2 and (i - neighbor[0]) > 0 and (j - neighbor[1] >= 0))):
                                initial_camp_moves.append([move_heuristic, [i, j], neighbor])
                            elif [i, j] in initial and (neighbor not in initial):
                                initial_moves.append([move_heuristic, [i, j], neighbor])
                            elif [i, j] not in initial:
                                all_adj_moves.append([move_heuristic, [i, j], neighbor])

                    # Generate all jump moves
                    for idx in range(len(n_value)):
                        neighbor = [i + n_value[idx][0], j + n_value[idx][1]]
                        if size_check(neighbor) and self.board[neighbor[0]][neighbor[1]] != 0:
                            next_jump = [neighbor[0] + n_value[idx][0], neighbor[1] + n_value[idx][1]]
                            if size_check(next_jump) and self.board[next_jump[0]][next_jump[1]] == 0:
                                visited = {tuple(next_jump): [i, j], tuple([i, j]): None}
                                open_queue = [[get_heuristic_value_jump(next_jump, [i, j]), next_jump, [i, j]]]
                                self.temp_board = copy.deepcopy(self.board)
                                while open_queue:
                                    h, cur, old = heapq.heappop(open_queue)
                                    if not ((([i, j] not in initial) and (cur in initial)) or (([i, j] in final) and (cur not in final))):
                                        move_heuristic = self.get_heuristic_value(cur, [i, j], pid)
                                        if ([i, j] in initial) and (cur in initial) and (
                                                (self.player == 1 and (cur[0] - i) > 0 and (cur[1] - j > 0)) or (
                                                self.player == 2 and (i - cur[0]) > 0 and (j - cur[1] > 0))):
                                            initial_camp_moves.append([move_heuristic, [i, j],
                                                                       cur])
                                        elif [i, j] in initial and (cur not in initial):
                                            initial_moves.append([move_heuristic, [i, j], cur])
                                        elif [i, j] not in initial:
                                            all_jump_moves.append([move_heuristic, [i, j], cur])

                                    self.temp_board[old[0]][old[1]] = 0
                                    self.temp_board[cur[0]][cur[1]] = temp_p

                                    for idx2 in range(len(n_value)):
                                        temp_cur = [cur[0] + n_value[idx2][0], cur[1] + n_value[idx2][1]]
                                        next_jump = [temp_cur[0] + n_value[idx2][0], temp_cur[1] + n_value[idx2][1]]
                                        if size_check(temp_cur) and self.temp_board[temp_cur[0]][
                                            temp_cur[1]] != 0 and size_check(next_jump) and (
                                                self.temp_board[next_jump[0]][next_jump[1]] == 0) and (
                                                tuple(next_jump) not in visited):
                                            heuristic = get_heuristic_value_jump(next_jump, cur)
                                            heapq.heappush(open_queue, [heuristic, next_jump, cur])
                                            visited[tuple(next_jump)] = cur

        if len(initial_moves) > 0:
            return sorted(initial_moves, key=lambda t: t[0], reverse=True)
        elif len(initial_camp_moves) > 0:
            return sorted(initial_camp_moves, key=lambda t: t[0], reverse=True)

        return sorted(all_jump_moves + all_adj_moves, key=lambda t: t[0], reverse=True)

    def set_move(self, pos, turn):
        if turn:
            self.board[pos[2][0]][pos[2][1]] = self.player
        else:
            self.board[pos[2][0]][pos[2][1]] = self.opp_player
        self.board[pos[1][0]][pos[1][1]] = 0

    def unset_move(self, pos, turn):
        if turn:
            self.board[pos[1][0]][pos[1][1]] = self.player
        else:
            self.board[pos[1][0]][pos[1][1]] = self.opp_player
        self.board[pos[2][0]][pos[2][1]] = 0


class HalmaPlayerAgent:
    def __init__(self, all_config):
        self.all_config = all_config
        self.selected_moves = {}
        self.my_goal_dict = {}
        self.current_player = all_config.player
        try:
            with open('playdata.txt', 'rb') as fp:
                self.selected_moves = pickle.load(fp)
                self.my_goal_dict = pickle.load(fp)
            fp.close()
        except:
            self.selected_moves = {}
            self.my_goal_dict = {}

        if all_config.player == 1:
            loc = black_final_locations
        else:
            loc = white_final_locations

        # Change heuristics during game
        final_count = 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if all_config.board[i][j] == all_config.player and ([i, j] in loc):
                    final_count += 1
        global h_weight
        if final_count > 16:
            h_weight = 3
        elif final_count > 10:
            h_weight = 2
        elif final_count > 5:
            h_weight = 1
        else:
            h_weight = 0

    # Select best move
    def best_move(self):
        board_config = HalmaGamePlayer(self.all_config, self.selected_moves, self.my_goal_dict)

        if len(self.selected_moves) < INITIAL_NUMBER:
            if self.current_player == 1 and len(initial_black_moves) >= len(self.selected_moves):
                # check if move is valid
                temp_best_move = initial_black_moves.pop(len(self.selected_moves))
                if board_config.validate_next_move(temp_best_move[0], temp_best_move[1]):
                    next_best_move = temp_best_move
                else:
                    next_best_move = self.SearchandPrune(board_config)

            elif self.current_player == 2 and len(initial_white_moves) >= len(self.selected_moves):
                # check if move is valid
                temp_best_move = initial_white_moves.pop(len(self.selected_moves))
                if board_config.validate_next_move(temp_best_move[0], temp_best_move[1]):
                    next_best_move = temp_best_move
                else:
                    next_best_move = self.SearchandPrune(board_config)
            else:
                next_best_move = self.SearchandPrune(board_config)

        else:
            next_best_move = self.SearchandPrune(board_config)

        if next_best_move:
            out_list = board_config.generate_output_path(next_best_move[0], next_best_move[1])
            with open('output.txt', 'w+') as fp:
                for item in out_list[::-1]:
                    fp.write(item + "\n")
            fp.close()

            self.all_config.board[next_best_move[0][0]][next_best_move[0][1]] = 0
            self.all_config.board[next_best_move[1][0]][next_best_move[1][1]] = self.current_player
            self.my_goal_dict = board_config.set_my_goal()
            self.selected_moves[tuple(next_best_move[1])] = next_best_move[0]
            with open('playdata.txt', 'wb') as fp:
                pickle.dump(self.selected_moves, fp, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.my_goal_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

    def SearchandPrune(self, board_config):
        self.next_move = []
        self.depth_limit = 3
        self.current_depth = 0
        self.max_depth = 0

        value = self.max_ab(board_config, -float('inf'), float('inf'), self.depth_limit)

        return self.next_move

    def min_ab(self, board_config, alpha, beta, depth_limit):
        if board_config.is_terminal_state():
            return board_config.calculate_utility_value()
        if depth_limit == 0:
            return board_config.get_final_heuristic_value()

        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

        v = float('inf')
        for move in board_config.generate_all_moves(False)[:min_moves_count]:
            board_config.set_move(move, False)
            node_val = self.max_ab(board_config, alpha, beta, depth_limit - 1)
            if node_val < v:
                v = node_val
            board_config.unset_move(move, False)

            if v <= alpha:
                self.current_depth -= 1
                return v
            beta = min(beta, v)

        self.current_depth -= 1
        return v

    def max_ab(self, board_config, alpha, beta, depth_limit):
        if board_config.is_terminal_state():
            return board_config.calculate_utility_value()
        if depth_limit == 0:
            return board_config.get_final_heuristic_value()

        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

        v = -float('inf')
        for move in board_config.generate_all_moves(True)[:max_moves_count]:
            board_config.set_move(move, True)
            node_val = self.min_ab(board_config, alpha, beta, depth_limit - 1)
            if node_val > v:
                v = node_val
                if depth_limit == self.depth_limit:
                    self.next_move = move[1:]
            board_config.unset_move(move, True)

            if v >= beta:
                self.current_depth -= 1
                return v
            alpha = max(alpha, v)

        self.current_depth -= 1
        return v


# class to store game type, color, remaining time and board configuration
# board configuration is of type list[list]
class BoardGame:
    def __init__(self, gtype, color, time, board):
        self.gtype = gtype
        if color == "BLACK":
            self.player = 1
            self.opp_player = 2
        else:
            self.player = 2
            self.opp_player = 1
        self.time = time
        self.board = board
        # Temp board to set pawn positions during jumps
        self.temp_board = 0
        self.my_player = HalmaPlayerAgent(self)

        if self.gtype == "SINGLE":
            self.make_single_move()
        else:
            self.my_player.best_move()

    def generate_output_path(self, cur_pos, next_pos):
        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if neighbor == next_pos:
                return [
                    'E ' + str(cur_pos[1]) + ',' + str(cur_pos[0]) + ' ' + str(next_pos[1]) + ',' + str(next_pos[0])]

        visited = {}

        for idx in range(len(n_value)):
            neighbor = [cur_pos[0] + n_value[idx][0], cur_pos[1] + n_value[idx][1]]
            if size_check(neighbor) and self.board[neighbor[0]][neighbor[1]] != 0:
                next_jump = [neighbor[0] + n_value[idx][0], neighbor[1] + n_value[idx][1]]
                if size_check(next_jump) and self.board[next_jump[0]][next_jump[1]] == 0:
                    visited[tuple(next_jump)] = [cur_pos[0], cur_pos[1]]
                    visited[tuple([cur_pos[0], cur_pos[1]])] = None
                    if next_jump == next_pos:
                        break
                    open_queue = [[get_heuristic_value_jump(next_jump, cur_pos), next_jump, cur_pos]]
                    self.temp_board = copy.deepcopy(self.board)
                    while open_queue:
                        h, cur, old = heapq.heappop(open_queue)
                        if cur == next_pos:
                            break

                        self.temp_board[old[0]][old[1]] = 0
                        self.temp_board[cur[0]][cur[1]] = self.player

                        for idx2 in range(len(n_value)):
                            temp_cur = [cur[0] + n_value[idx2][0], cur[1] + n_value[idx2][1]]
                            next_jump = [temp_cur[0] + n_value[idx2][0], temp_cur[1] + n_value[idx2][1]]
                            if size_check(temp_cur) and self.temp_board[temp_cur[0]][
                                temp_cur[1]] != 0 and size_check(next_jump) and (
                                    self.temp_board[next_jump[0]][next_jump[1]] == 0) and (
                                    tuple(next_jump) not in visited):
                                heuristic = get_heuristic_value_jump(next_jump, cur)
                                heapq.heappush(open_queue, [heuristic, next_jump, cur])
                                visited[tuple(next_jump)] = cur

        temp_var = next_pos
        output_path = []
        while temp_var is not None:
            t1 = visited[tuple(temp_var)]
            if t1 is not None:
                output_path.append(
                    'J ' + str(t1[1]) + ',' + str(t1[0]) + ' ' + str(temp_var[1]) + ',' + str(temp_var[0]))
            temp_var = t1
        return output_path

    def get_heuristic_value_for_single_move(self, next_pos, cur):

        temp = self.player

        # Distance from final location
        dmin, dmax = 0, 0
        if temp == 2:
            dmin = min(abs(next_pos[0] - black_final_pos[0]), abs(next_pos[1] - black_final_pos[1]))
            dmax = max(abs(next_pos[0] - black_final_pos[0]), abs(next_pos[1] - black_final_pos[1]))

        if temp == 1:
            dmin = min(abs(next_pos[0] - white_final_pos[0]), abs(next_pos[1] - white_final_pos[1]))
            dmax = max(abs(next_pos[0] - white_final_pos[0]), abs(next_pos[1] - white_final_pos[1]))

        h_sum = int(1.4 * dmin + (dmax - dmin))

        # Displacement between moves
        if temp == 1:
            dmin = min((next_pos[0] - cur[0]), (next_pos[1] - cur[1]))
            dmax = max((next_pos[0] - cur[0]), (next_pos[1] - cur[1]))
        else:
            dmin = min((cur[0] - next_pos[0]), (cur[1] - next_pos[1]))
            dmax = max((cur[0] - next_pos[0]), (cur[1] - next_pos[1]))

        h_sum += int(1.4 * dmin + (dmax - dmin))

        # Other criteria for Black pawn
        if temp == 1:
            # Moving down and right and location in diagonal location
            if next_pos[0] - cur[0] > 0 and (next_pos[1] - cur[0] > 0) and next_pos in diagonal_locations:
                h_sum += 70

            # Crossing the centre
            if cur[0] < 7 and next_pos[0] >= 7:
                h_sum += 70
                # moving towards right
                if next_pos[1] > cur[1]:
                    h_sum += 10

            # Moving towards centre
            elif cur[0] < 7 and (7 - next_pos[0] < 7 - cur[0]):
                h_sum += 30

            elif next_pos[0] < 7 and cur[0] >= 7:
                h_sum -= 70

            if cur[0] < 7 and (7 - next_pos[0] < 7 - cur[0]) and (next_pos[1] - cur[1] > 0):
                h_sum += 20

            # movement within final positions
            if cur in black_final_locations and next_pos in black_final_locations:
                h_sum -= 20

            if cur in black_final_locations and next_pos not in black_final_locations:
                h_sum -= 200

            # add extra value if pawns are moved out of initial position
            if cur in white_final_locations and next_pos not in white_final_locations:
                h_sum += 500

            if (cur in white_final_locations) and (next_pos in white_final_locations) and (
                    (next_pos[0] - cur[0]) >= 0) and ((next_pos[1] - cur[1]) >= 0):
                h_sum += 200

            # Entering final positions
            if (cur not in black_final_locations) and (next_pos in black_final_locations):
                h_sum += 60

        # Other criteria for white pawn
        elif temp == 2:
            if cur[0] - next_pos[0] > 0 and (cur[0] - next_pos[1] > 0) and next_pos in diagonal_locations:
                h_sum += 70

            if cur[0] > 7 and next_pos[0] <= 7:
                h_sum += 70
                # moving towards left
                if next_pos[1] < cur[1]:
                    h_sum += 10

            elif cur[0] > 7 and (next_pos[0] - 7 < cur[0] - 7):
                h_sum += 30

            elif next_pos[0] > 7 and cur[0] <= 7:
                h_sum -= 70

            if cur[0] > 7 and (next_pos[0] - 7 < cur[0] - 7) and (cur[1] - next_pos[1] > 0):
                h_sum += 20

            # movement within final positions
            if cur in white_final_locations and next_pos in white_final_locations:
                h_sum -= 20

            if cur in white_final_locations and next_pos not in white_final_locations:
                h_sum -= 200

            # add extra value if pawns are moved out of initial position
            if cur in black_final_locations and next_pos not in black_final_locations:
                h_sum += 500

            if cur in black_final_locations and next_pos in black_final_locations and (cur[0] - next_pos[0] >= 0) and (
                    cur[1] - next_pos[1] >= 0):
                h_sum += 200

                # Entering final positions
            if (cur not in white_final_locations) and (next_pos in white_final_locations):
                h_sum += 60

        return h_sum

    # If player is given color, check if neighbor is empty. If empty generate adjacent move. Else generate jump move.
    # If both return None, there is no valid move.
    def make_single_move(self):

        if self.player == 1:
            initial = white_final_locations
            final = black_final_locations

        else:
            initial = black_final_locations
            final = white_final_locations

        all_adj_moves = []
        all_jump_moves = []
        initial_camp_moves = []
        initial_moves = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.player:

                    # Generate adjacent move
                    for idx in range(len(n_value)):
                        neighbor = [i + n_value[idx][0], j + n_value[idx][1]]
                        if size_check(neighbor) and (self.board[neighbor[0]][neighbor[1]] == 0) and (
                                not ((([i, j] not in initial) and (neighbor in initial)) or (([i, j] in final) and (neighbor not in final)))):
                            if ([i, j] in initial) and (neighbor in initial) and (
                                    (self.player == 1 and (neighbor[0] - i) > 0 and (neighbor[1] - j >= 0)) or (
                                    self.player == 2 and (i - neighbor[0]) > 0 and (j - neighbor[1] >= 0))):
                                initial_camp_moves.append(
                                    [self.get_heuristic_value_for_single_move(neighbor, [i, j]), [i, j], neighbor])
                            elif [i, j] in initial and (neighbor not in initial):
                                initial_moves.append(
                                    [self.get_heuristic_value_for_single_move(neighbor, [i, j]), [i, j], neighbor])
                            elif [i, j] not in initial:
                                all_adj_moves.append(
                                    [self.get_heuristic_value_for_single_move(neighbor, [i, j]), [i, j], neighbor])

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == self.player:

                    # Generate all jump moves
                    for idx in range(len(n_value)):
                        neighbor = [i + n_value[idx][0], j + n_value[idx][1]]
                        if size_check(neighbor) and self.board[neighbor[0]][neighbor[1]] != 0:
                            next_jump = [neighbor[0] + n_value[idx][0], neighbor[1] + n_value[idx][1]]
                            if size_check(next_jump) and self.board[next_jump[0]][next_jump[1]] == 0:
                                visited = {tuple(next_jump): [i, j], tuple([i, j]): None}
                                open_queue = [[get_heuristic_value_jump(next_jump, [i, j]), next_jump, [i, j]]]
                                self.temp_board = copy.deepcopy(self.board)
                                while open_queue:
                                    h, cur, old = heapq.heappop(open_queue)
                                    if not ((([i, j] not in initial) and (cur in initial)) or (
                                            ([i, j] in final) and (cur not in final))):
                                        move_heuristic = self.get_heuristic_value_for_single_move(cur, [i, j])
                                        if ([i, j] in initial) and (cur in initial) and (
                                                (self.player == 1 and (cur[0] - i) > 0 and (cur[1] - j > 0)) or (
                                                self.player == 2 and (i - cur[0]) > 0 and (j - cur[1] > 0))):
                                            initial_camp_moves.append([move_heuristic, [i, j],
                                                                       cur])
                                        elif [i, j] in initial and (cur not in initial):
                                            initial_moves.append([move_heuristic, [i, j], cur])
                                        elif [i, j] not in initial:
                                            all_jump_moves.append([move_heuristic, [i, j], cur])

                                    self.temp_board[old[0]][old[1]] = 0
                                    self.temp_board[cur[0]][cur[1]] = self.player

                                    for idx2 in range(len(n_value)):
                                        temp_cur = [cur[0] + n_value[idx2][0], cur[1] + n_value[idx2][1]]
                                        next_jump = [temp_cur[0] + n_value[idx2][0], temp_cur[1] + n_value[idx2][1]]
                                        if size_check(temp_cur) and self.temp_board[temp_cur[0]][
                                            temp_cur[1]] != 0 and size_check(next_jump) and (
                                                self.temp_board[next_jump[0]][next_jump[1]] == 0) and (
                                                tuple(next_jump) not in visited):
                                            heuristic = get_heuristic_value_jump(next_jump, cur)
                                            heapq.heappush(open_queue, [heuristic, next_jump, cur])
                                            visited[tuple(next_jump)] = cur

        if len(initial_moves) > 0:
            moves = sorted(initial_moves, key=lambda t: t[0], reverse=True)

        elif len(initial_camp_moves) > 0:
            moves = sorted(initial_camp_moves, key=lambda t: t[0], reverse=True)

        else:
            moves = sorted(all_adj_moves + all_jump_moves, key=lambda t: t[0], reverse=True)
        out_list = []
        if moves:
            out_list = self.generate_output_path(moves[0][1], moves[0][2])

        with open('output.txt', 'w+') as fp:
            for item in out_list[::-1]:
                fp.write(item + "\n")
        fp.close()


if __name__ == '__main__':
    with open('input.txt', 'r') as fp:
        program_input = fp.readlines()
        config = []
        # 1 - Black, 2 - White, 0 - Empty
        for i in range(3, len(program_input)):
            temp = []
            for j in range(len(program_input[i].strip())):
                if program_input[i][j] == 'B':
                    temp.append(1)
                elif program_input[i][j] == 'W':
                    temp.append(2)
                elif program_input[i][j] == '.':
                    temp.append(0)
            config.append(temp)
    fp.close()

    game_obj = BoardGame(program_input[0].strip(), program_input[1].strip(), float(program_input[2]), config)
