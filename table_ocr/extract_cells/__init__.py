import cv2
import os

def extract_cell_images_from_table(image):
    # 이미지 블러 처리로 노이즈 제거
    BLUR_KERNEL_SIZE = (17, 17)
    STD_DEV_X_DIRECTION = 0
    STD_DEV_Y_DIRECTION = 0
    blurred = cv2.GaussianBlur(image, BLUR_KERNEL_SIZE, STD_DEV_X_DIRECTION, STD_DEV_Y_DIRECTION)
    
    # 이미지 이진화
    MAX_COLOR_VAL = 255
    BLOCK_SIZE = 15
    SUBTRACT_FROM_MEAN = -2
    img_bin = cv2.adaptiveThreshold(
        ~blurred,
        MAX_COLOR_VAL,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        BLOCK_SIZE,
        SUBTRACT_FROM_MEAN,
    )
    
    # 수직선과 수평선 감지
    vertical = horizontal = img_bin.copy()
    SCALE = 5
    image_width, image_height = horizontal.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image_width / SCALE), 1))
    horizontally_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(image_height / SCALE)))
    vertically_opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel)
    
    # 선 굵기 강화
    horizontally_dilated = cv2.dilate(horizontally_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    vertically_dilated = cv2.dilate(vertically_opened, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    
    # 수직선과 수평선 합치기
    mask = horizontally_dilated + vertically_dilated
    
    # 테이블 셀 윤곽선 찾기
    contours, heirarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )
    
    # 윤곽선을 직사각형으로 근사화
    perimeter_lengths = [cv2.arcLength(c, True) for c in contours]
    epsilons = [0.05 * p for p in perimeter_lengths]
    approx_polys = [cv2.approxPolyDP(c, e, True) for c, e in zip(contours, epsilons)]
    
    # 직사각형 형태의 윤곽선만 필터링
    approx_rects = [p for p in approx_polys if len(p) == 4]
    bounding_rects = [cv2.boundingRect(a) for a in approx_polys]
    
    # 너무 작은 직사각형 필터링
    MIN_RECT_WIDTH = 40
    MIN_RECT_HEIGHT = 10
    bounding_rects = [
        r for r in bounding_rects if MIN_RECT_WIDTH < r[2] and MIN_RECT_HEIGHT < r[3]
    ]
    
    # 가장 큰 직사각형(전체 테이블) 제거
    largest_rect = max(bounding_rects, key=lambda r: r[2] * r[3])
    bounding_rects = [b for b in bounding_rects if b is not largest_rect]
    
      # 세로로 병합된 셀 감지
    def detect_vertically_merged_cells(cells):
        merged_cells = []
        for i, cell in enumerate(cells):
            for j, other_cell in enumerate(cells[i+1:], start=i+1):
                if abs(cell[0] - other_cell[0]) < 5 and abs(cell[2] - other_cell[2]) < 5:
                    if cell[1] < other_cell[1] and cell[1] + cell[3] > other_cell[1]:
                        merged_cells.append((i, j))
        return merged_cells

    # 셀들을 행과 열로 그룹화
    def group_cells_into_rows_and_columns(cells):
        sorted_cells = sorted(cells, key=lambda c: (c[1], c[0]))  # y좌표로 정렬 후 x좌표로 정렬
        rows = []
        current_row = []
        last_y = -1
        for cell in sorted_cells:
            if abs(cell[1] - last_y) > 10:  # 새로운 행 시작
                if current_row:
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [cell]
                last_y = cell[1]
            else:
                current_row.append(cell)
        if current_row:
            rows.append(sorted(current_row, key=lambda c: c[0]))
        return rows

    cells = [c for c in bounding_rects]
    merged_cells = detect_vertically_merged_cells(cells)
    rows = group_cells_into_rows_and_columns(cells)

    # 병합된 셀 처리 및 이미지 추출
    cell_images_rows = []
    for i, row in enumerate(rows):
        cell_images_row = []
        for j, (x, y, w, h) in enumerate(row):
            cell_image = image[y:y+h, x:x+w]
            is_merged = any((i, k) in merged_cells for k in range(i+1, len(rows)))
            if is_merged:
                # 병합된 셀의 경우 다음 행의 셀과 병합
                next_cell = next((cell for cell in rows[i+1] if abs(cell[0] - x) < 5), None)
                if next_cell:
                    _, ny, _, nh = next_cell
                    cell_image = image[y:ny+nh, x:x+w]
            cell_images_row.append({"image": cell_image, "merged": is_merged})
        cell_images_rows.append(cell_images_row)

    return cell_images_rows

def main(f):
    directory, filename = os.path.split(f)
    table = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    rows = extract_cell_images_from_table(table)
    cell_img_dir = os.path.join(directory, "cells")
    os.makedirs(cell_img_dir, exist_ok=True)
    paths = []

    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_filename = "{:03d}-{:03d}{}.png".format(i, j, "_merged" if cell["merged"] else "")
            path = os.path.join(cell_img_dir, cell_filename)
            cv2.imwrite(path, cell["image"])
            paths.append({"path": path, "merged": cell["merged"]})

    return paths