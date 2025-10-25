
print("== Скрипт запущен ==")


from ultralytics import YOLO
 #предобученная модель YOLO
print("Загружаем модель YOLO")
model = YOLO('yolo11n.pt')
#Обучение на данных
print('Начинаем обучеие')
model.train(
    data = 'data.yaml',
    epochs = 50,
    imgsz = 640,
    device = 'cpu'
)
print("Обучение завершено")