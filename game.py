import random

SIZE = 4
ROWS = SIZE
COLS = SIZE
EMPTY = 0

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Game:
    def __init__(self, tiles: [[int]] = None):
        self.tiles = tiles if tiles else [[EMPTY for row in range(ROWS)] for col in range(
            COLS)]
        self.game_over = False
        self.score = 0
        if not tiles:
            self._spawn()
            self._spawn()

    def move(self, direction: int):
        save_state = self._copy_tiles()
        if direction == UP:
            self._move_up()
        elif direction == DOWN:
            self._move_down()
        elif direction == LEFT:
            self._move_left()
        elif direction == RIGHT:
            self._move_right()
        if save_state != self.tiles:
            self._spawn()
            if self._check_game_over():
                self._end_game()

    def check_available(self, direction: int) -> bool:
        if direction == UP:
            return self._check_up()
        elif direction == DOWN:
            return self._check_down()
        elif direction == LEFT:
            return self._check_left()
        elif direction == RIGHT:
            return self._check_right()

    def flattened_board(self) -> [int]:
        return [row for col in self.tiles for row in col]

    def print_board(self):
        print(f" {'-'*9}")
        for row in range(ROWS):
            print("|", end="")
            for col in range(COLS):
                print(f" {self.tiles[col][row]}", end="")
            print(" |")
        print(f" {'-'*9}")

    def _check_up(self) -> bool:
        for col in range(COLS):
            for row in range(1, ROWS):
                if self.tiles[col][row] and self.tiles[col][row-1] in (EMPTY, self.tiles[col][row]):
                    return True
        return False

    def _check_down(self) -> bool:
        for col in range(COLS):
            for row in range(ROWS - 1):
                if self.tiles[col][row] and self.tiles[col][row+1] in (EMPTY, self.tiles[col][row]):
                    return True
        return False

    def _check_left(self) -> bool:
        for row in range(ROWS):
            for col in range(1, COLS):
                if self.tiles[col][row] and self.tiles[col-1][row] in (EMPTY, self.tiles[col][row]):
                    return True
        return False

    def _check_right(self) -> bool:
        for row in range(ROWS):
            for col in range(COLS - 1):
                if self.tiles[col][row] and self.tiles[col+1][row] in (EMPTY, self.tiles[col][row]):
                    return True
        return False

    def _copy_tiles(self) -> [[int]]:
        return [[self.tiles[col][row] for row in range(ROWS)] for col in range(COLS)]

    def _move_up(self):
        for col in range(COLS):
            for row in range(ROWS):
                if self.tiles[col][row]:
                    self._move_piece(col, row, 0, -1)

    def _move_down(self):
        for col in range(COLS):
            for row in reversed(range(ROWS)):
                if self.tiles[col][row]:
                    self._move_piece(col, row, 0, 1)

    def _move_left(self):
        for row in range(ROWS):
            for col in range(COLS):
                if self.tiles[col][row]:
                    self._move_piece(col, row, -1, 0)

    def _move_right(self):
        for row in range(ROWS):
            for col in reversed(range(COLS)):
                if self.tiles[col][row]:
                    self._move_piece(col, row, 1, 0)

    def _move_piece(self, col: int, row: int, dx: int, dy: int):
        original_col, original_row = col, row
        while col+dx >= 0 and col+dx < COLS and row+dy >= 0 and row+dy < ROWS and self.tiles[col+dx][row+dy] == EMPTY:
            col += dx
            row += dy
        if col+dx >= 0 and col+dx < COLS and row+dy >= 0 and row+dy < ROWS and self.tiles[col+dx][row+dy] == self.tiles[original_col][original_row]:
            self.tiles[original_col][original_row] = EMPTY
            self.tiles[col+dx][row+dy] *= 2
            self.score += self.tiles[col+dx][row+dy]
        else:
            self.tiles[original_col][original_row], self.tiles[col][row] = EMPTY, self.tiles[original_col][original_row]

    def _spawn(self):
        col, row = self._get_random_empty_tile()
        val = 2 if random.random() < 0.9 else 4
        self.score += val
        self.tiles[col][row] = val

    def _check_game_over(self):
        if self._fullboard():
            for col in range(COLS):
                if any([self.tiles[col][row] == self.tiles[col][row+1] for row in range(ROWS-1)]):
                    return False
            for row in range(ROWS):
                if any([self.tiles[col][row] == self.tiles[col+1][row] for col in range(COLS - 1)]):
                    return False
            return True
        return False

    def _fullboard(self) -> bool:
        return all([all([self.tiles[col][row] != EMPTY for row in range(ROWS)]) for col in range(COLS)])

    def _get_random_empty_tile(self) -> (int, int):
        checked = []
        while True:
            col, row = random.randrange(COLS), random.randrange(ROWS)
            if (col, row) not in checked and self.tiles[col][row] == EMPTY:
                return col, row
            else:
                checked.append((col, row))

    def _end_game(self):
        self.game_over = True
        self._calculate_final_score()

    def _calculate_final_score(self):
        self._final_score = sum(
            [sum([self.tiles[col][row] for row in range(ROWS)]) for col in range(COLS)])
