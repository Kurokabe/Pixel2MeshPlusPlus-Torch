import os
import pickle
import random
from glob import glob

import cv2
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core import Dataset
from azureml.fsspec import AzureMachineLearningFileSystem
from PIL import Image
from torch.utils.data.dataset import Dataset


class P2MPPDataset(Dataset):
    def __init__(self, file_list_df, data_root):
        self.file_list_df = file_list_df
        self.data_root = data_root

        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        self.p2mppdata = ml_client.data.get("P2MPP", version="1")

    def __len__(self):
        return len(self.file_list_df)

    def __getitem__(self, idx):
        current_item = self.file_list_df.iloc[idx]
        object_name = current_item["object_name"]
        dataset_type = current_item["dataset_type"]
        fs_p2mppdata = AzureMachineLearningFileSystem(self.p2mppdata.path)

        pkl_path = os.path.join(
            self.data_root, dataset_type, "data", f"{object_name}.dat"
        )

        with fs_p2mppdata.open(pkl_path, "rb") as f:
            pkl = pickle.load(f, encoding="bytes")

        label = pkl

        # load image file
        img_root = os.path.join(self.data_root, dataset_type, "rendering")
        img_path = os.path.join(img_root, object_name)
        metadata_path = os.path.join(img_path, "rendering_metadata.txt")
        with fs_p2mppdata.open(metadata_path, "r") as f:
            camera_meta_data = np.loadtxt(f)
        # if self.mesh_root is not None:
        #     mesh = np.loadtxt(
        #         os.path.join(
        #             self.mesh_root, category + "_" + item_id + "_00_predict.xyz"
        #         )
        #     )
        # else:
        mesh = None

        imgs = np.zeros((3, 224, 224, 3))
        poses = np.zeros((3, 5))

        
        # image_files = fs_p2mppdata.glob(os.path.join(img_path, "*.png"))
        image_files = [os.path.join(img_path, f"{i:03d}.png") for i in range(24)]
        selected_image_files = random.sample(image_files, 3)

        for idx, file_name in enumerate(
            selected_image_files
        ):  # TODO take 3 random images
            view = int(os.path.splitext(os.path.basename(file_name))[0])
            with fs_p2mppdata.open(file_name, "rb") as f:
                img = Image.open(f)
                img = np.array(img)
            img[np.where(img[:, :, 3] == 0)] = 255
            img = cv2.resize(img, (224, 224))
            img_inp = img.astype("float32") / 255.0
            imgs[idx] = img_inp[:, :, :3]
            poses[idx] = camera_meta_data[view]

        label = label[1] if len(label) == 2 else label
        points = label[:, :3]
        normals = label[:, 3:]

        imgs = np.transpose(imgs, (0, 3, 1, 2))

        return {
            "images": imgs.astype(np.float32),
            "images_orig": imgs.astype(np.float32),
            "points": points.astype(np.float32),
            "normals": normals.astype(np.float32),
            "poses": poses.astype(np.float32),
            "filename": object_name,
        }
