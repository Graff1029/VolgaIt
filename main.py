import cv2
from ultralytics import YOLO
import easyocr
import csv
from datetime import timedelta
import os

def main():
    print("Загружаем модели")

    #загружаем наши модели
    detection_model = YOLO('runs/detect/train8/weights/best.pt')
    ocr_reader = easyocr.Reader(['en'])

    #открытие видеофайла
    video_path = "rest_video.mp4"
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл {video_path} не найден")
        print("Положи видео в папку с проектом")
        return

    cap = cv2.VideoCapture(video_path)

    results_file = open('detected_plates.csv', 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(results_file)
    csv_writer.writerow(['time', 'plate_num'])

    #обработка видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_detection_time = -1
    frame_count = 0
    print("Начинаем обрабтку видео")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #время видео
        current_time_sec = frame_count / fps
        time_formatted = str(timedelta(seconds=current_time_sec))[2:10].replace('.',':')

        #поиск номерныхх знаков на карте
        detections = detection_model(frame, verbose = False)

        #обаботка каждой найденной области
        for box in detections[0].boxes:
            if box.conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_roi = frame[y1:y2, x1:x2]
                if plate_roi.shape[0] < 20 or plate_roi.shape[1] <60:
                    continue

                #улучшение изображения чтобы распознать
                try:
                    ocr_result = ocr_reader.readtext(plate_roi_enhanced, detail=0)
                    if ocr_result:
                        plate_text = ''.join(ocr_result).upper().replace(' ','')
                        plate_text = ''.join(c for c in plate_text if c.isalnum())#только буквы и цыфры
                        if last_detection_time == -1 or (current_time_sec - last_detection_time) >= 0.2:#условие не чаще 1 раза в 200 мс
                            if len(plate_text) >= 5:
                                print(f"Время: {time_formatted}, Номер: {plate_text}")
                                csv_writer.writerow([time_formatted, plate_text])
                                last_detection_time = current_time_sec
                except Exception as e:
                    continue
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")#прогресс каждые 100 кадров
    cap.release()
    results_file.close()
    print("Обработка видео завершена")
    print("Результаты сохранены в файл: detectes_plates.csv")
if __name__ == "__main__":
    main()
                            
