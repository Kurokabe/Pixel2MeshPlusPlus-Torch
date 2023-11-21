import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        data_list,
        data_root,
        test_size: float,
        seed: int,
        batch_size: int,
        num_workers: int,
        num_points: int,
    ):
        super().__init__()
        self.data_list = data_list
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_points = num_points

        if name == "p2mpp":
            from p2mpp.data.p2mpp_dataset import P2MPPDataset
        elif name == "p2mpp_azure":
            from p2mpp.data.p2mpp_dataset_azure import P2MPPDataset
        else:
            raise ValueError(f"Unknown dataset name:  {name}")

        data_list_df = pd.read_csv(data_list)
        # data_list_df = data_list_df[data_list_df["dataset_type"] == "ShapeNet"]
        train_file_list_df, test_file_list_df = train_test_split(
            data_list_df, test_size=test_size, random_state=seed
        )
        # train_file_list_df = data_list_df[data_list_df["dataset_type"] == "ShapeNet"]
        # test_file_list_df = data_list_df[data_list_df["dataset_type"] == "ShapeNet"]

        self.train_dataset = P2MPPDataset(train_file_list_df, data_root)
        self.test_dataset = P2MPPDataset(test_file_list_df, data_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.shapenet_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.shapenet_collate,
        )

    def shapenet_collate(self, batch):
        if len(batch) > 1:
            # num_points = max(b["points"].shape[0] for b in batch)

            points_orig, normals_orig = [], []

            for t in batch:
                pts, normal = t["points"], t["normals"]
                length = pts.shape[0]
                choices = np.resize(np.random.permutation(length), self.num_points)
                t["points"], t["normals"] = pts[choices], normal[choices]
                points_orig.append(torch.from_numpy(pts))
                normals_orig.append(torch.from_numpy(normal))
            ret = default_collate(batch)
            ret["points_orig"] = points_orig
            ret["normals_orig"] = normals_orig
            return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret
