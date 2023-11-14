import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
from p2mpp.data.shapenet import ShapeNet
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file_list,
        test_file_list,
        data_root,
        image_root,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.train_file_list = train_file_list
        self.test_file_list = test_file_list
        self.data_root = data_root
        self.image_root = image_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # TODO clean data_root
        self.train_dataset = ShapeNet(train_file_list, data_root + "/train", image_root)
        self.test_dataset = ShapeNet(test_file_list, data_root + "/test", image_root)

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
            num_points = max(b["points"].shape[0] for b in batch)

            points_orig, normals_orig = [], []

            for t in batch:
                pts, normal = t["points"], t["normals"]
                length = pts.shape[0]
                choices = np.resize(np.random.permutation(length), num_points)
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
