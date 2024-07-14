import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
import json
import logging
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TableDetector:
    def __init__(self):
        logging.info("TableDetector 초기화 중...")
        self.processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        logging.info("TableDetector 초기화 완료")

    def detect_table(self, image: np.ndarray) -> List[Dict[str, any]]:
        logging.info("표 감지 시작")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
        
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detected_objects.append({
                "label": self.model.config.id2label[label.item()],
                "score": score.item(),
                "box": box
            })
        
        logging.info(f"{len(detected_objects)}개의 객체 감지됨")
        return detected_objects

    def visualize_detected_objects(self, image: np.ndarray, objects: List[Dict[str, any]], output_path: str):
        plt.figure(figsize=(20, 20))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for obj in objects:
            box = obj['box']
            label = obj['label']
            plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                              fill=False, edgecolor='red', linewidth=2))
            plt.text(box[0], box[1], f"{label}: {obj['score']:.2f}", 
                     color='red', fontsize=12, backgroundcolor='white')
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"감지된 객체 시각화 결과 저장됨: {output_path}")

class OCRProcessor:
    def __init__(self):
        logging.info("OCRProcessor 초기화 중...")
        self.processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
        self.model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
        logging.info("OCRProcessor 초기화 완료")

    def recognize_text(self, cell_image: np.ndarray) -> str:
        pil_image = Image.fromarray(cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB))
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

class TimetableStructurer:
    @staticmethod
    def structure_timetable(cells: List[Dict[str, any]]) -> Dict[str, any]:
        logging.info("시간표 구조화 시작")
        structured_data = {"cells": cells}
        logging.info(f"{len(structured_data['cells'])}개의 셀 구조화 완료")
        return structured_data

class TimetableRecognizer:
    def __init__(self):
        logging.info("TimetableRecognizer 초기화 중...")
        self.table_detector = TableDetector()
        self.ocr = OCRProcessor()
        self.structurer = TimetableStructurer()
        logging.info("TimetableRecognizer 초기화 완료")

    def process_timetable(self, image_path: str) -> Dict[str, any]:
        logging.info(f"시간표 처리 시작: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        
        detected_objects = self.table_detector.detect_table(image)
        self.table_detector.visualize_detected_objects(image, detected_objects, "detected_objects.png")
        
        cells = []
        for i, obj in enumerate(detected_objects):
            if obj["label"] == "table cell":
                x1, y1, x2, y2 = map(int, obj["box"])
                cell_image = image[y1:y2, x1:x2]
                text = self.ocr.recognize_text(cell_image)
                cells.append({
                    "id": i,
                    "text": text,
                    "box": obj["box"],
                    "label": obj["label"]
                })
                # 각 셀 이미지 저장
                cv2.imwrite(f"cell_{i}.png", cell_image)
                logging.info(f"셀 {i}: 텍스트 '{text}' 인식됨")
        
        structured_data = self.structurer.structure_timetable(cells)
        logging.info("시간표 처리 완료")
        return structured_data

    def save_results(self, data: Dict[str, any], output_path: str):
        logging.info(f"결과 저장 중: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info("결과 저장 완료")

def main(image_path: str, output_path: str):
    try:
        recognizer = TimetableRecognizer()
        timetable_data = recognizer.process_timetable(image_path)
        recognizer.save_results(timetable_data, output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        if saved_data and saved_data['cells']:
            logging.info(f"시간표 데이터가 {output_path}에 성공적으로 저장되었습니다.")
            logging.info(f"인식된 셀 수: {len(saved_data['cells'])}")
            for cell in saved_data['cells']:
                logging.info(f"셀 ID: {cell['id']}, 텍스트: {cell['text']}")
        else:
            logging.error("저장된 데이터가 비어 있거나 셀이 없습니다.")
    
    except Exception as e:
        logging.error(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    image_path = "C:\AI\_8.png"
    output_path = "timetable_data.json"
    main(image_path, output_path)