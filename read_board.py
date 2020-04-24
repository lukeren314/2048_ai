import pyautogui
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

background_color = (187, 173, 160)
pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR\tesseract.exe'


def read_board() -> [[int]]:
    screen_image = pyautogui.screenshot()
    left, top, right, bottom = _get_background_box(screen_image)
    num_board = []
    if right-left == bottom-top:
        cropped_board = screen_image.crop((left, top, right, bottom))
        board = _process_board_image(cropped_board)

        board_w, board_h = board.size
        tile_w, tile_h = int(board_w/4), int(board_h/4)
        for col in range(4):
            num_board.append([])
            for row in range(4):
                num_int = _get_int(board, col, row, tile_w, tile_h)
                num_board[col].append(num_int)
    return num_board


def _get_background_box(screen_image: Image) -> ('left', 'top', 'right', 'bottom'):
    width, height = screen_image.size
    pixels = screen_image.convert('RGB').load()

    min_x, min_y, max_x, max_y = width, height, 0, 0
    for x in range(width):
        for y in range(height):
            if pixels[x, y] == background_color:
                min_x = min(x, min_x)
                min_y = min(y, min_y)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
    return min_x, min_y, max_x, max_y
    # rect_x, rect_y = min_x, min_y
    # rect_w, rect_h = max_x-min_x, max_y-min_y
    # print(f'x: {rect_x} y: {rect_y} w: {rect_w} h: {rect_h}')


UPPER_BOUND = 240
LOWER_BOOUND = 25


def _process_board_image(cropped_board: Image) -> Image:
    old_w, old_h = cropped_board.size
    board = cropped_board.resize(
        (old_w*2, old_h*2), Image.ANTIALIAS).convert('RGB')
    # board = board.point(lambda x: 0 if x > 245 else x)
    # board.show()
    data = board.getdata()
    new_data = []
    for item in data:
        if item[0] > UPPER_BOUND and item[1] > UPPER_BOUND and item[2] > UPPER_BOUND:
            new_data.append((0, 0, 0))
        else:
            new_data.append(item)
    board.putdata(new_data)
    # board.show()
    enhancer = ImageEnhance.Contrast(board)
    board = enhancer.enhance(2)
    board = board.convert('L')
    board = board.point(lambda x: 0 if x < LOWER_BOOUND else 255)
    board = board.filter(ImageFilter.SMOOTH_MORE)
    # board.show()
    return board


def _get_int(board: Image, col: int, row: int, tile_w: int, tile_h: int) -> int:
    tile = board.crop(
        (tile_w*col, tile_h*row, tile_w*(col+1), tile_h*(row+1)))
    num_str = pytesseract.image_to_string(
        tile, config=r'--oem 3 --psm 6 outputbase digits')
    num_str = ''.join([c for c in num_str if c.isdigit()])
    num_int = int(num_str) if num_str else 0
    return num_int
