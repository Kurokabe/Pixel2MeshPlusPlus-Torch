import numpy as np
import cv2
import os
import pickle
from torch.utils.data.dataset import Dataset


class ShapeNet(Dataset):
    def __init__(self, file_list, data_root, image_root):
        self.file_list = file_list
        self.data_root = data_root
        self.image_root = image_root

        self.pkl_list = []
        with open(file_list, "r") as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                self.pkl_list.append(line)

    def __len__(self):
        return len(self.pkl_list)

    def __getitem__(self, idx):
        pkl_item = self.pkl_list[idx]
        pkl_path = os.path.join(self.data_root, pkl_item)
        pkl = pickle.load(open(pkl_path, "rb"), encoding="bytes")

        label = pkl

        # load image file
        img_root = self.image_root
        ids = pkl_item.split("_")
        category = ids[-3]
        item_id = ids[-2]
        img_path = os.path.join(img_root, category, item_id, "rendering")
        camera_meta_data = np.loadtxt(os.path.join(img_path, "rendering_metadata.txt"))
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
        for idx, view in enumerate([0, 6, 7]):
            img = cv2.imread(
                os.path.join(img_path, str(view).zfill(2) + ".png"),
                cv2.IMREAD_UNCHANGED,
            )
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
            "filename": pkl_item,
        }
        return imgs, label, poses, pkl_item, mesh
