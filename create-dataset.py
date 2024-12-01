"""
This script reads the Category and Attribute Prediction Benchmark from the DeepFashion dataset and splits the data into train/val/test groups and saves the img_path, bbox vector, category vector,
attribute vector for each image in all the 3 groups. 
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

    def read_imgs_and_split(self, X):
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
                words = line.split()
                image_to_category[words[0].strip()] = int(words[1].strip())

                if count % 50000 == 0:
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
                words = line.split()
                data = (words[1], words[2], words[3], words[4])
                image_to_bbox[words[0]] = data

                # Print progress after every 50000 iterations
                if count % 50000 == 0:
                    print(f"Processed {count} lines in image_to_bbox mapping.")

        # Initialize the list to collect all data
        data_list = []

        # Read in the images
        with open(self.list_eval_partition) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # Read and throw away the header

            count = 0
            for line in f:
                count += 1
                words = line.split()
                img = words[0].strip()
                category_idx = image_to_category[img]
                category = str(category_to_name[category_idx - 1])
                bbox = np.asarray(image_to_bbox[img], dtype=np.int16)
                partition = words[1].strip()

                data = {"img_path": img, "bbox": bbox, "category": category, "partition": partition}
                data_list.append(data)

                # Print progress after every 50000 iterations
                if count % 50000 == 0:
                    print(f"Processed {count} lines in image processing.")

            print("Total images", len(data_list))

        # Create dataframe from data_list
        df_all = pd.DataFrame(data_list)

        # Limit the number of images per category per partition
        def sample_group(group):
            if len(group) > X:
                return group.sample(n=X, random_state=42)
            else:
                return group

        df_sampled = df_all.groupby(['partition', 'category'], group_keys=False).apply(sample_group)

        # Split into train, val, test
        self.train = df_sampled[df_sampled['partition'] == 'train'].drop('partition', axis=1)
        self.val = df_sampled[df_sampled['partition'] == 'val'].drop('partition', axis=1)
        self.test = df_sampled[df_sampled['partition'] == 'test'].drop('partition', axis=1)

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
    X = 200  # Set the maximum number of elements per category
    df.read_imgs_and_split(X)
