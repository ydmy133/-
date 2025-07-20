import albumentations as A
import cv2
import os


def augment_image(image_path, label_path, output_dir, num_augments=5):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # 50%概率水平翻转
        A.RandomBrightnessContrast(p=0.2),  # 20%概率调整亮度和对比度
        A.RandomGamma(p=0.2),  # 20%概率调整Gamma值
        A.RandomRotate90(p=0.5),  # 50%概率随机旋转90/180/270度
        A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.5)  # 50%概率添加Cutout遮挡
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    image = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)

    for i in range(num_augments):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_class_labels = augmented['class_labels']

        output_image_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg")
        output_label_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.txt")

        cv2.imwrite(output_image_path, augmented_image)

        with open(output_label_path, 'w') as f:
            for j in range(len(augmented_bboxes)):
                bbox = augmented_bboxes[j]
                class_id = augmented_class_labels[j]
                f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")


# 增强训练集
train_image_dir = 'climbing_detection_dataset/images/train'
train_label_dir = 'climbing_detection_dataset/labels/train'
output_dir = 'climbing_detection_dataset/augmented/train'
os.makedirs(output_dir, exist_ok=True)
for image_file in os.listdir(train_image_dir):
    image_path = os.path.join(train_image_dir, image_file)
    label_path = os.path.join(train_label_dir, image_file.replace('.jpg', '.txt'))
    augment_image(image_path, label_path, output_dir, num_augments=5)

# 增强验证集
val_image_dir = 'climbing_detection_dataset/images/val'
val_label_dir = 'climbing_detection_dataset/labels/val'
output_dir = 'climbing_detection_dataset/augmented/val'
os.makedirs(output_dir, exist_ok=True)
for image_file in os.listdir(val_image_dir):
    image_path = os.path.join(val_image_dir, image_file)
    label_path = os.path.join(val_label_dir, image_file.replace('.jpg', '.txt'))
    augment_image(image_path, label_path, output_dir, num_augments=5)