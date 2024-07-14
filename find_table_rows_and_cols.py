import torch
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from io import BytesIO

# 모델 로드
table_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/table-transformer-structure-recognition")
table_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
ocr_processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
ocr_model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")

def load_image(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        return Image.open(BytesIO(response.content))
    else:
        return Image.open(image_path)

def detect_table_structure(image):
    inputs = table_feature_extractor(images=image, return_tensors="pt")
    outputs = table_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = table_feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
    return results

def extract_cells(results, image_size):
    cells = []
    rows = []
    columns = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i) for i in box.tolist()]
        class_name = table_model.config.id2label[label.item()]
        if class_name == "table cell":
            cells.append(box)
        elif class_name == "table row":
            rows.append(box)
        elif class_name == "table column":
            columns.append(box)
    
    # 행과 열 정렬
    rows.sort(key=lambda x: x[1])  # y 좌표로 정렬
    columns.sort(key=lambda x: x[0])  # x 좌표로 정렬
    
    return cells, rows, columns

def recognize_text(image, box):
    cell_image = image.crop(box)
    pixel_values = ocr_processor(cell_image, return_tensors="pt").pixel_values
    generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def process_timetable(image_path):
    image = load_image(image_path)
    results = detect_table_structure(image)
    cells, rows, columns = extract_cells(results, image.size)
    
    timetable_data = []
    for cell in cells:
        text = recognize_text(image, cell)
        row_index = next(i for i, row in enumerate(rows) if row[1] <= cell[1] <= row[3])
        col_index = next(i for i, col in enumerate(columns) if col[0] <= cell[0] <= col[2])
        timetable_data.append({"row": row_index, "col": col_index, "text": text})
    
    return timetable_data

def structure_timetable(timetable_data):
    structured_data = {"요일": [], "시간": []}
    for cell in timetable_data:
        if cell["row"] == 0 and cell["col"] > 0:
            structured_data["요일"].append(cell["text"])
        elif cell["col"] == 0 and cell["row"] > 0:
            structured_data["시간"].append(cell["text"])
        elif cell["row"] > 0 and cell["col"] > 0:
            day = structured_data["요일"][cell["col"] - 1]
            time = structured_data["시간"][cell["row"] - 1]
            if day not in structured_data:
                structured_data[day] = []
            structured_data[day].append({"시간": time, "내용": cell["text"]})
    
    return structured_data

def main(image_path):
    timetable_data = process_timetable(image_path)
    structured_timetable = structure_timetable(timetable_data)
    
    print("시간표 정보:")
    for day, classes in structured_timetable.items():
        if day not in ["요일", "시간"]:
            print(f"\n{day}:")
            for class_info in classes:
                print(f"  {class_info['시간']}: {class_info['내용']}")

if __name__ == "__main__":
    image_path = "C:\AI\_9.jpg"  # 시간표 이미지 경로를 지정하세요
    main(image_path)