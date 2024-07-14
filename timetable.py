import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import json
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OCRProcessor:
    def __init__(self):
        logging.info("OCR 프로세서 초기화 중...")
        self.processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
        self.model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
        logging.info("OCR 프로세서 초기화 완료")

    def recognize_text(self, image):
        """
        이미지에서 텍스트를 인식합니다.
        :param image: PIL Image 객체
        :return: 인식된 텍스트
        """
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()

def split_image_into_lines(image, min_line_height=20):
    """
    이미지를 텍스트 줄 단위로 분할합니다.
    :param image: NumPy 배열 형태의 이미지
    :param min_line_height: 최소 줄 높이
    :return: 분할된 줄 이미지 리스트
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 수평 프로젝션 프로필 계산
    h_proj = np.sum(thresh, axis=1)
    
    lines = []
    start = 0
    for i in range(1, len(h_proj)):
        if h_proj[i] == 0 and h_proj[i-1] > 0 and i - start > min_line_height:
            lines.append(image[start:i])
            start = i
    if start < len(h_proj) - 1:
        lines.append(image[start:])
    
    return lines

def process_cell_image(ocr, image_path):
    """
    셀 이미지를 처리하고 OCR을 수행합니다.
    :param ocr: OCRProcessor 인스턴스
    :param image_path: 이미지 파일 경로
    :return: 인식된 텍스트
    """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"이미지를 불러올 수 없습니다: {image_path}")
        return ""

    lines = split_image_into_lines(image)
    recognized_texts = []
    for line in lines:
        pil_line = Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB))
        text = ocr.recognize_text(pil_line)
        recognized_texts.append(text)
    
    return " ".join(recognized_texts)

def get_cell_dimensions(directory):
    """
    모든 셀의 크기를 분석하여 병합된 셀을 감지합니다.
    :param directory: 이미지 파일이 있는 디렉토리 경로
    :return: 셀 크기 정보 딕셔너리
    """
    cell_dimensions = {}
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            row, col = map(int, filename.split('.')[0].split('-'))
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is not None:
                height, width = image.shape[:2]
                cell_dimensions[(row, col)] = (width, height)
    return cell_dimensions

def analyze_merged_cells(cell_dimensions):
    """
    셀 크기 정보를 바탕으로 병합된 셀을 분석합니다.
    :param cell_dimensions: 셀 크기 정보 딕셔너리
    :return: 병합 정보가 포함된 셀 정보 딕셔너리
    """
    merged_cells = {}
    base_width = min(width for width, _ in cell_dimensions.values())
    base_height = min(height for _, height in cell_dimensions.values())

    for (row, col), (width, height) in cell_dimensions.items():
        colspan = max(1, round(width / base_width))
        rowspan = max(1, round(height / base_height))
        merged_cells[(row, col)] = {
            "colspan": colspan,
            "rowspan": rowspan
        }
    
    return merged_cells

def process_timetable_cells(directory):
    """
    디렉토리 내의 모든 셀 이미지를 처리합니다.
    :param directory: 이미지 파일이 있는 디렉토리 경로
    :return: 처리된 시간표 데이터
    """
    ocr = OCRProcessor()
    timetable_data = {}
    cell_dimensions = get_cell_dimensions(directory)
    merged_cells = analyze_merged_cells(cell_dimensions)

    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    logging.info(f"{len(image_files)}개의 이미지 파일을 처리합니다.")

    for filename in tqdm(image_files, desc="이미지 처리 중"):
        row, col = map(int, filename.split('.')[0].split('-'))
        image_path = os.path.join(directory, filename)
        
        try:
            text = process_cell_image(ocr, image_path)
            if row not in timetable_data:
                timetable_data[row] = {}
            timetable_data[row][col] = {
                "text": text,
                "colspan": merged_cells.get((row, col), {}).get("colspan", 1),
                "rowspan": merged_cells.get((row, col), {}).get("rowspan", 1)
            }
            logging.debug(f"파일 {filename} 처리 완료: '{text}'")
        except Exception as e:
            logging.error(f"파일 {filename} 처리 중 오류 발생: {str(e)}")

    return timetable_data

def save_to_json(data, output_path):
    """
    처리된 데이터를 JSON 파일로 저장합니다.
    :param data: 저장할 데이터
    :param output_path: 저장할 파일 경로
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logging.info(f"결과가 {output_path}에 저장되었습니다.")

def main(directory, output_path):
    """
    메인 실행 함수
    :param directory: 이미지 파일이 있는 디렉토리 경로
    :param output_path: 결과를 저장할 JSON 파일 경로
    """
    logging.info("시간표 셀 처리 시작")
    timetable_data = process_timetable_cells(directory)
    save_to_json(timetable_data, output_path)
    logging.info("처리 완료")

if __name__ == "__main__":
    directory = "C:\AI\_8\cells"  # 이미지 파일이 있는 디렉토리 경로
    output_path = "timetable_ocr_results.json"  # 결과를 저장할 JSON 파일 경로
    main(directory, output_path)