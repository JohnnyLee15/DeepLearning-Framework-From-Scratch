import os
from PIL import Image
import csv
import numpy as np

def jpeg_to_greyscale(image_dir):
    data_rows = []

    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                img_data = []
                img_path = os.path.join(class_dir, file)

                if "virus" in img_path.lower():
                    img_data.append("virus")
                elif "bacteria" in img_path.lower():
                    img_data.append("bacteria")
                else:
                    img_data.append("normal")

                with Image.open(img_path).convert('L') as img:
                    img = img.resize((128, 128))
                    img = np.array(img).flatten()
                    img_data += img.tolist()
                    data_rows.append(img_data)

    return data_rows

def write_to_csv(data_rows, output_file):
    with open(output_file, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(data_rows)


def jpeg_to_csv(image_dir, out_path):
    data_greyscale = jpeg_to_greyscale(image_dir)
    write_to_csv(data_greyscale, out_path)


data_dir = "DataFiles/chest_xray/"
test_dir = os.path.join(data_dir, "test")
train_dir = os.path.join(data_dir, "train")

out_dir = os.path.join(data_dir, "csv_format")
out_test = os.path.join(out_dir, "test.csv")
out_train = os.path.join(out_dir, "train.csv")

jpeg_to_csv(test_dir, out_test)
jpeg_to_csv(train_dir, out_train)

