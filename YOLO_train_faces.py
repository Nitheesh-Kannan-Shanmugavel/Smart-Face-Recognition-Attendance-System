import os
import cv2
import numpy as np
from ultralytics import YOLO
import albumentations as A


student_folder = "students"
output_dataset_folder = "yolo_dataset"
train_folder = os.path.join(output_dataset_folder, "train")
os.makedirs(train_folder, exist_ok=True)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Error loading face cascade classifier")
    exit(1)

def convert_to_yolo_format(box, image_width, image_height, class_id):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height

    return f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}"


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image


def validate_and_correct_bbox(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height


    if x_max <= x_min:
        x_max = x_min + 1  
    if y_max <= y_min:
        y_max = y_min + 1  


    x_min = max(0, min(x_min, img_width - 1))
    y_min = max(0, min(y_min, img_height - 1))
    x_max = max(x_min + 1, min(x_max, img_width))
    y_max = max(y_min + 1, min(y_max, img_height))

    return [x_min, y_min, x_max, y_max]  


def augment_image(image, boxes, img_width, img_height):
    valid_boxes = [validate_and_correct_bbox(box, img_width, img_height) for box in boxes]

    augmentations = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(blur_limit=15, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.CLAHE(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ]

    transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))
    augmented_images = [image]
    augmented_boxes = [valid_boxes]
    
    for _ in range(4):
        augmented = transform(image=image, bboxes=valid_boxes)
        aug_image = augmented['image']
        aug_boxes = augmented['bboxes']
        augmented_images.append(aug_image)
        augmented_boxes.append(aug_boxes)
    
    return augmented_images, augmented_boxes

student_names = sorted(os.listdir(student_folder))
class_id_mapping = {name: idx for idx, name in enumerate(student_names)}

for student_name in student_names:
    student_path = os.path.join(student_folder, student_name)
    if not os.path.isdir(student_path):
        continue
    
    class_id = class_id_mapping[student_name]
    
    for image_name in os.listdir(student_path):
        image_path = os.path.join(student_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            image = cv2.resize(image, (640, 480))
            img_height, img_width, _ = image.shape
            
            enhanced_image = enhance_image(image)
            
            gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 0:
                print(f"No faces detected in {image_path}")
                continue
            
            augmented_images, augmented_boxes = augment_image(enhanced_image, faces, img_width, img_height)
            
            for i, aug_image in enumerate(augmented_images):
                base_name = f"{student_name}_{os.path.splitext(image_name)[0]}_aug_{i}"
                output_image_path = os.path.join(train_folder, f"{base_name}.jpg")
                output_label_path = os.path.join(train_folder, f"{base_name}.txt")
                
                cv2.imwrite(output_image_path, aug_image)
                with open(output_label_path, "w") as f:
                    for (x_min, y_min, x_max, y_max) in augmented_boxes[i]:
                        yolo_box = convert_to_yolo_format((x_min, y_min, x_max, y_max), img_width, img_height, class_id)
                        f.write(yolo_box + "\n")
                
                print(f"Processed {student_name}/{image_name} (augmentation {i})")
        
        except PermissionError:
            print(f"Permission denied for file: {image_path}")


dataset_config = f"""
train: {os.path.abspath(train_folder)}
val: {os.path.abspath(train_folder)}  

nc: {len(student_names)}
names: {list(student_names)}
"""

with open(os.path.join(output_dataset_folder, "custom_dataset.yaml"), "w") as f:
    f.write(dataset_config)

model = YOLO("yolov8m.pt")
results = model.train(
    data=os.path.join(output_dataset_folder, "custom_dataset.yaml"),
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolo_student_recognition",
    augment=True
)

print("Training complete!")
