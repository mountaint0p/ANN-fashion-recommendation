"""
This script reads the Category and Attribute Prediction Benchmark from the DeepFashion dataset and splits the data into train/val/test groups and saves the img_path, bbox vector, category vector,
attribute vector for each image in all the 3 groups. It processes only 1/10 of the images.
"""
import os
import numpy as np
import pandas as pd


class create_DeepFashion:

    def __init__(self, dataset_path):

        # The constants
        img_folder_name = "dataset/img"
        eval_folder_name = "dataset/Eval"
        anno_folder_name = "dataset/Anno"
        list_eval_partition_file = "list_eval_partition.txt"
        list_attr_img_file = "list_attr_img.txt"
        list_category_img_file = "list_category_img.txt"
        list_category_cloth_file = "list_category_cloth.txt"
        list_bbox_file = "list_bbox.txt"
        # The data structures
        self.train = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])
        self.val = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])
        self.test = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])

        # Construct the paths
        self.path = dataset_path
        self.img_dir = os.path.join(self.path, img_folder_name)
        self.eval_dir = os.path.join(self.path, eval_folder_name)
        self.anno_dir = os.path.join(self.path, anno_folder_name)

        self.list_eval_partition = os.path.join(self.eval_dir, list_eval_partition_file)
        self.list_attr_img = os.path.join(self.anno_dir, list_attr_img_file)
        self.list_category_img = os.path.join(self.anno_dir, list_category_img_file)
        self.list_category_cloth = os.path.join(self.anno_dir, list_category_cloth_file)
        self.list_bbox = os.path.join(self.anno_dir, list_bbox_file)

    def read_imgs_and_split(self):
        # Declaring the names of the CSVs where the split data would be stored
        train_file = "train.csv"
        val_file = "val.csv"
        test_file = "test.csv"

        # Read in the category index to category name mapping from the DeepFashion dataset
        category_to_name = {}

        with open(self.list_category_cloth) as f:
            count = int(f.readline().strip())  # Read the first line
            _ = f.readline().strip()  # Read and throw away the header

            i = 0
            for line in f:
                words = line.split()
                category_to_name[i] = str(words[0])
                i = i + 1

        assert count == 50

        # Read in the image to category mapping from the DeepFashion dataset
        image_to_category = {}
        with open(self.list_category_img) as f:
            imgs_count = int(f.readline().strip())  # Read the first line
            _ = f.readline().strip()  # Read and throw away the header

            count = 0
            # Read each line and split the words and store the data
            for line in f:
                count += 1
                if count % 5 != 1:
                    continue
                words = line.split()
                image_to_category[words[0].strip()] = int(words[1].strip())

                # Print progress after every 500 iterations
                if count % 500 == 1:
                    print(f"Processed {count} lines in image_to_category mapping.")

        # Read in the image to bbox mapping
        image_to_bbox = {}
        with open(self.list_bbox) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # Read and throw away the header

            count = 0
            # Read each line and split the words and store the data
            for line in f:
                count += 1
                if count % 5 != 1:
                    continue
                words = line.split()
                data = (words[1], words[2], words[3], words[4])
                image_to_bbox[words[0]] = data

                # Print progress after every 500 iterations
                if count % 500 == 1:
                    print(f"Processed {count} lines in image_to_bbox mapping.")

        # Read in the images
        with open(self.list_eval_partition) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # Read and throw away the header

            count = 0
            for line in f:
                count += 1
                if count % 5 != 1:
                    continue
                words = line.split()
                img = words[0].strip()
                category_idx = image_to_category[img]
                category = str(category_to_name[category_idx - 1])
                bbox = np.asarray(image_to_bbox[img], dtype=np.int16)

                # Divide and save the data into train/val/test dataframes
                if words[1].strip() == "train":
                    self.train = pd.concat([self.train, pd.DataFrame([{"img_path": img, "bbox": bbox, "category": category}])], ignore_index=True)
                if words[1].strip() == "val":
                    self.val = pd.concat([self.val, pd.DataFrame([{"img_path": img, "bbox": bbox, "category": category}])], ignore_index=True)
                if words[1].strip() == "test":
                    self.test = pd.concat([self.test, pd.DataFrame([{"img_path": img, "bbox": bbox, "category": category}])], ignore_index=True)

                # Print progress after every 500 iterations
                if count % 500 == 1:
                    print(f"Processed {count} lines in image processing.")

            print("Training images", int(self.train.shape[0]))
            print("Validation images", int(self.val.shape[0]))
            print("Test images", int(self.test.shape[0]))

        # Store the data structures
        self.train.to_csv(self.path + "/split-data/train_new.csv", index=False)
        self.val.to_csv(self.path + "/split-data/val_new.csv", index=False)
        self.test.to_csv(self.path + "/split-data/test_new.csv", index=False)
        print("Storage done")


if __name__ == "__main__":
    current_path = os.getcwd()
    df = create_DeepFashion(current_path)
    df.read_imgs_and_split()
