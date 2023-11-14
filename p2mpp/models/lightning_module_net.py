import torch
from typing import Literal, Tuple
import pytorch_lightning as pl
from p2mpp.configs import NetworkConfig, LossConfig, OptimConfig
from p2mpp.models.p2m import P2MModel
from p2mpp.models.losses.p2m import P2MLoss
from p2mpp.models.mesh.ellipsoid import Ellipsoid


class LightningModuleNet(pl.LightningModule):
    def __init__(
        self,
        network_config: NetworkConfig,
        loss_config: LossConfig,
        optim_config: OptimConfig,
    ):
        super().__init__()
        self.model = P2MModel(
            hidden_dim=network_config.hidden_dim,
            coord_dim=network_config.coord_dim,
            last_hidden_dim=network_config.last_hidden_dim,
            gconv_activation=network_config.gconv_activation,
            backbone=network_config.backbone,
            align_with_tensorflow=network_config.align_with_tensorflow,
            base_mesh_config=network_config.base_mesh_config,
            z_threshold=network_config.z_threshold,
            camera_f=network_config.camera_f,
            camera_c=network_config.camera_c,
        )
        self.ellipsoid = Ellipsoid(network_config.base_mesh_config.mesh_pose)

        self.criterion = P2MLoss(loss_config=loss_config, ellipsoid=self.ellipsoid)

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=optim_config.lr,
            betas=(optim_config.adam_beta1, optim_config.adam_beta2),
            weight_decay=optim_config.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        poses = batch["poses"]

        pred = self.model(images, poses)
        loss, loss_summary = self.criterion(pred, batch)

        # Prefix all keys from loss_summary with 'train/'
        loss_summary = {f"train/{k}": v for k, v in loss_summary.items()}

        self.log_dict(
            loss_summary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        poses = batch["poses"]

        pred = self.model(images, poses)
        loss, loss_summary = self.criterion(pred, batch)

        # Prefix all keys from loss_summary with 'val/'
        loss_summary = {f"val/{k}": v for k, v in loss_summary.items()}

        self.log_dict(
            loss_summary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def configure_optimizers(self):
        return self.optimizer
