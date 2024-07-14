import cv2
import os
from find_table_rows_and_cols import find_table_rows_and_cols 

def main(files):
    results = []
    for f in files:
        directory, filename = os.path.split(f)
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        tables = find_tables(image)
        files = []
        filename_sans_extension = os.path.splitext(filename)[0]
        if tables:
            os.makedirs(os.path.join(directory, filename_sans_extension), exist_ok=True)
        for i, table in enumerate(tables):
            rows, cols = find_table_rows_and_cols(table)
            # 행과 열을 출력합니다.
            print('행:')
            for row in rows:
                print(row)
            print('열:')
            for col in cols:
                print(col)

            # 행과 열을 이미지에 그립니다.
            for row in rows:
                cv2.line(table, (0, row[0]), (table.shape[1], row[1]), (0, 0, 255), 2)
            for col in cols:
                cv2.line(table, (col[0], 0), (col[1], table.shape[0]), (0, 0, 255), 2)

            # 결과 이미지를 표시합니다.
            cv2.imshow('Table with Rows and Columns', table)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            table_filename = "table-{:03d}.png".format(i)
            table_filepath = os.path.join(
                directory, filename_sans_extension, table_filename
            )
            files.append(table_filepath)
            cv2.imwrite(table_filepath, table)
        if tables:
            results.append((f, files))
    # Results is [[<input image>, [<images of detected tables>]]]
    return results

# 테스트 코드
files = ['C:\AI\_9.jpg'] # 테스트할 이미지 파일 목록
results = main(files)