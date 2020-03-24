import random

SIZE = 4
ROWS = SIZE
COLS = SIZE
EMPTY = 0

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class Board:
    def __init__(self):
        self._tiles = [[EMPTY for row in range(ROWS)] for col in range(COLS)]
        self._dead = False
        self._raw_score = 0
        self._final_score = 0
        self._spawn()
        self._spawn()

    def move(self, direction: int) -> None:
        save_state = self._copy_tiles()
        if direction == UP:
            self._move_up()
        elif direction == DOWN:
            self._move_down()
        elif direction == LEFT:
            self._move_left()
        elif direction == RIGHT:
            self._move_right()
        if save_state != self._tiles:
            self._spawn()
            if self._check_game_over():
                self._end_game()

    def get_tile(self, col: int, row: int) -> int:
        return self._tiles[col][row]

    def dead(self) -> bool:
        return self._dead

    def raw_score(self) -> int:
        return self._raw_score

    def final_score(self) -> int:
        return self._final_score

    def _copy_tiles(self) -> [[int]]:
        return [[self._tiles[col][row] for row in range(ROWS)] for col in range(COLS)]

    def _move_up(self) -> None:
        for col in range(COLS):
            self._set_col(col, self._merge(self._get_col(col)))

    def _move_down(self) -> None:
        for col in range(COLS):
            self._set_col(col, self._merge(self._get_col(col)[::-1])[::-1])

    def _move_left(self) -> None:
        for row in range(ROWS):
            self._set_row(row, self._merge(self._get_row(row)))

    def _move_right(self) -> None:
        for row in range(ROWS):
            self._set_row(row, self._merge(self._get_row(row)[::-1])[::-1])

    def _get_row(self, row: int) -> [int]:
        return [self._tiles[col][row] for col in range(COLS)]

    def _get_col(self, col: int) -> [int]:
        return [self._tiles[col][row] for row in range(ROWS)]

    def _set_row(self, row: int, nums: [int]) -> None:
        for col in range(COLS):
            self._tiles[col][row] = nums[col]

    def _set_col(self, col: int, nums: [int]) -> None:
        for row in range(ROWS):
            self._tiles[col][row] = nums[row]

    def _merge(self, nums: [int]) -> [int]:
        nums = [num for num in nums if num != EMPTY]
        new_nums = []
        i = 0
        while i < len(nums):
            if i < len(nums) - 1 and nums[i] == nums[i+1]:
                new_nums.append(nums[i] * 2)
                self._raw_score += nums[i] * 2
                i += 1
            else:
                new_nums.append(nums[i])
            i += 1
        return new_nums + [EMPTY] * (SIZE - len(new_nums))

    def _spawn(self) -> None:
        col, row = self._get_random_empty_tile()
        self._tiles[col][row] = 2 if random.random() < 0.9 else 4

    def _check_game_over(self) -> None:
        if self._full_board():
            for col in range(COLS):
                if any([self._tiles[col][row] == self._tiles[col][row+1] for row in range(ROWS-1)]):
                    return False
            for row in range(ROWS):
                if any([self._tiles[col][row] == self._tiles[col+1][row] for col in range(COLS - 1)]):
                    return False
            return True
        return False

    def _full_board(self) -> bool:
        return all([all([self._tiles[col][row] != EMPTY for row in range(ROWS)]) for col in range(COLS)])

    def _get_random_empty_tile(self) -> (int, int):
        checked = []
        while True:
            col, row = random.randrange(COLS), random.randrange(ROWS)
            if (col, row) not in checked and self._tiles[col][row] == EMPTY:
                return col, row
            else:
                checked.append((col, row))

    def _end_game(self) -> None:
        self._dead = True
        self._calculate_final_score()

    def _calculate_final_score(self) -> None:
        self._final_score = sum(
            [sum([self._tiles[col][row] for row in range(ROWS)]) for col in range(COLS)])


class Game:
    def __init__(self):
        self._board = Board()

    def move(self, direction: int) -> None:
        self._board.move(direction)

    def dead(self) -> bool:
        return self._board.dead()

    def raw_score(self) -> int:
        return self._board.raw_score()

    def final_score(self) -> int:
        return self._board.final_score()

    def display_board(self) -> None:
        print(f" {'-'*9}")
        for row in range(ROWS):
            print("|", end="")
            for col in range(COLS):
                print(f" {self._board.get_tile(col, row)}", end="")
            print(" |")
        print(f" {'-'*9}")


if __name__ == "__main__":
    games = []
    for _ in range(1000):
        games.append(Game())
    directions = [UP, DOWN, LEFT, RIGHT]
    while not all([game.dead() for game in games]):
        for game in games:
            game.move(random.choice(directions))

    for game in sorted(games, key=lambda g: g.fitness()):
        game.display_board()
        print(f"FITNESS: {game.raw_score()}")
