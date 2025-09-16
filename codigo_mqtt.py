import cv2
from ultralytics import YOLO
import paho.mqtt.client as mqtt

# Modelo
model_path = "best.pt"
model = YOLO(model_path)

# Mapeamento das fases para ângulos
fase_to_angle = {
    "fase_1": 30,
    "fase_2": 60,
    "fase_3": 90
}

# Config MQTT
BROKER = "192.168.15.8"  # IP do broker
PORT = 1883
TOPIC = "hidroponia/servo"

# Cliente MQTT
client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# Imagem
image_path = 'images/classe1.jpg'

# Predição
results = model.predict(source=image_path, conf=0.25, iou=0.7, device="cpu")

detected_angles = []

for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name in fase_to_angle:
            detected_angles.append(fase_to_angle[class_name])

if detected_angles:
    avg_angle = int(sum(detected_angles) / len(detected_angles))
    print(f"Média das fases detectadas: {avg_angle}°")

    # Publica no tópico
    client.publish(TOPIC, avg_angle)
    print(f"Ângulo enviado via MQTT → {avg_angle}°")
else:
    print("Nenhuma fase detectada.")
