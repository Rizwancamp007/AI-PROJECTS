import cv2
import numpy as np
import matplotlib.pyplot as plt

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

output_layers = net.getUnconnectedOutLayersNames()

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread("image1.jpg")
if image is None:
    raise FileNotFoundError("Image not found. Check filename/path.")

height, width, _ = image.shape

blob = cv2.dnn.blobFromImage(
    image, 0.00392, (416, 416), (0, 0, 0),
    swapRB=True, crop=False
)

net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    conf = f"{confidences[i]:.2f}"

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image, f"{label} {conf}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (0, 255, 0), 2
    )

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

cv2.imwrite("output_image.jpg", image)
