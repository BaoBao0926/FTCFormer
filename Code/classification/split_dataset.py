
import datasets
import random
import os
import shutil



original_dataset_dir = '/home/cheng/muyi/TCFormer/classification/datasets/WHU-RS19/WHU-RS19/RSDataset'  # orginal dataset
output_dataset_dir = "/home/cheng/muyi/TCFormer/classification/datasets/WHU-RS19/WHU-RS19"  # target dataset
train_ratio = 0.8  # ration of training and testing
random_seed = 42  # fixed random seed

# set random seed
random.seed(random_seed)

# create dir
train_dir = os.path.join(output_dataset_dir, "train")
val_dir = os.path.join(output_dataset_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)


for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(images)  # random shuffle
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    # create class dir
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # copy image into training dir
    for img in train_images:
        shutil.copy2(os.path.join(class_dir, img), os.path.join(train_class_dir, img))

    # copy images into testing dir
    for img in val_images:
        shutil.copy2(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

    print(f"finish class {class_name} spliting: training set {len(train_images)} number, test set {len(val_images)} number")

print("finish!")
