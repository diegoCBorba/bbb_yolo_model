import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
import itertools
import tflite_runtime.interpreter as tflite

# Caminho do modelo TFLite
MODEL_PATH = "best_saved_model/best_float32.tflite"

# Inicializar o interpreter TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# Lista de imagens para simulação
image_paths = [
    'images/classe1.jpg',
    'images/classe2.jpg',
    'images/classe3.jpg'
]

def preprocess(image_path, input_shape):
    """Redimensiona e normaliza a imagem"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")
    h, w = input_shape[1], input_shape[2]  # geralmente [1, 640, 640, 3]
    img = cv2.resize(image, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img, image.shape[:2]

def postprocess(outputs, original_shape, conf_threshold=0.25, iou_threshold=0.7):
    """Extrai classes e aplica NMS simples"""
    predictions = outputs[0]  # Assumindo [N, 85] -> x,y,w,h,conf,...classes
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_probs = predictions[:, 5:]
    class_ids = np.argmax(class_probs, axis=1)
    scores = scores * np.max(class_probs, axis=1)
    
    # Filtrar por confiança
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # NMS simples
    keep = []
    if len(boxes) == 0:
        return []

    x1 = boxes[:,0] - boxes[:,2]/2
    y1 = boxes[:,1] - boxes[:,3]/2
    x2 = boxes[:,0] + boxes[:,2]/2
    y2 = boxes[:,1] + boxes[:,3]/2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w*h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    detected_angles = [fase_to_angle[f"fase_{class_ids[i]+1}"] for i in keep if f"fase_{class_ids[i]+1}" in fase_to_angle]
    return detected_angles

# Loop principal
for image_path in itertools.cycle(image_paths):
    input_shape = input_details[0]['shape']
    img, original_shape = preprocess(image_path, input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(o['index']) for o in output_details]
    
    detected_angles = postprocess(outputs, original_shape, conf_threshold=0.25, iou_threshold=0.7)
    
    if detected_angles:
        avg_angle = int(sum(detected_angles)/len(detected_angles))
        print(f"[{image_path}] Média das fases detectadas: {avg_angle}°")
        client.publish(TOPIC, str(avg_angle))
        print(f"Ângulo enviado via MQTT → {avg_angle}°")
    else:
        print(f"[{image_path}] Nenhuma fase detectada.")
    
    time.sleep(60)  # Delay entre imagens
