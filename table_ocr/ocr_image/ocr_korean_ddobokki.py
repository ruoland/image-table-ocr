from transformers import pipeline
import cv2
import numpy as np
from PIL import Image
import os

pipe = pipeline("image-to-text", model="ddobokki/ko-trocr")

def ocr_korean(img):
    txt = pipe(img)
    return txt[0]['generated_text']

def segment_text_lines(image_path, output_dir):
    # 이미지 읽기
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 수평 프로젝션 프로필 계산
    proj = np.sum(denoised, 1)
    
    # 텍스트 라인 찾기
    th = 0
    H, W = img.shape[:2]
    uppers = [y for y in range(H-1) if proj[y] <= th and proj[y+1] > th]
    lowers = [y for y in range(H-1) if proj[y] > th and proj[y+1] <= th]
    
    # 각 텍스트 라인 추출 및 저장
    line_images = []
    for i, (upper, lower) in enumerate(zip(uppers, lowers)):
        if lower - upper > 5:  # 너무 얇은 라인 무시
            line_img = img[upper:lower, :]
            line_images.append(line_img)
            
            # 라인 이미지 저장
            cv2.imwrite(os.path.join(output_dir, f'line_{i+1}.jpg'), line_img)
    
    return line_images

def ocr_korean_multi_line(image_path, output_dir):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    line_images = segment_text_lines(image_path, output_dir)
    results = []
    
    for i, line_img in enumerate(line_images):
        # OpenCV 이미지를 PIL Image로 변환
        line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(line_img_rgb)
        
        # OCR 수행
        text = ocr_korean(pil_img)
        results.append(text)
    
    return results

# 사용 예:
image_path = 'C:\AI\_9.jpg'
output_dir = 'c:\AI\_9\dirctory'
recognized_lines = ocr_korean_multi_line(image_path, output_dir)
for i, line in enumerate(recognized_lines, 1):
    print(f"Line {i}: {line}")
print(f"분할된 라인 이미지가 {output_dir}에 저장되었습니다.")