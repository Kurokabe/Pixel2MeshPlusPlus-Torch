from typing import Literal, Tuple

import pytorch_lightning as pl
import torch
from loguru import logger

from p2mpp.configs import LossConfig, NetworkConfig, OptimConfig
from p2mpp.models.losses.p2m import P2MLoss
from p2mpp.models.mesh.ellipsoid import Ellipsoid
from p2mpp.models.p2m import P2MModel
from p2mpp.utils.vis.renderer import MeshRenderer


class LightningModuleNet(pl.LightningModule):
    def __init__(
        self,
        network_config: NetworkConfig,
        loss_config: LossConfig,
        optim_config: OptimConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
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
        log_level = "DEBUG"
        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

        logger.remove(0)
        logger.add(
            "file.log",
            level=log_level,
            format=log_format,
            colorize=False,
            backtrace=True,
            diagnose=True,
        )

        self.ellipsoid = Ellipsoid(network_config.base_mesh_config.mesh_pose)

        self.criterion = P2MLoss(loss_config=loss_config, ellipsoid=self.ellipsoid)

        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=optim_config.lr,
            betas=(optim_config.adam_beta1, optim_config.adam_beta2),
            weight_decay=optim_config.weight_decay,
        )

        self.renderer = MeshRenderer(
            camera_f=network_config.camera_f,
            camera_c=network_config.camera_c,
            mesh_pos=network_config.base_mesh_config.mesh_pose,
        )

        self.train_sample_outputs = []
        self.validation_sample_outputs = []
        self.num_samples_to_visualize = 16

    def on_train_epoch_start(self):
        self.train_sample_outputs = []

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        poses = batch["poses"]

        logger.info(f"Processing batch {batch['filename']}")

        pred = self.model(images, poses)
        loss, loss_summary = self.criterion(pred, batch)

        # Prefix all keys from loss_summary with 'train/'
        loss_summary = {f"train/{k}": v for k, v in loss_summary.items()}

        self.log_dict(
            loss_summary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if len(self.train_sample_outputs) < self.num_samples_to_visualize:
            self.train_sample_outputs.append((batch, pred))

        return loss

    def on_train_epoch_end(self):
        input_batch, pred = self.train_sample_outputs[0]
        render_mesh = self.renderer.p2m_batch_visualize(
            input_batch, pred, self.ellipsoid.faces
        )
        self.logger.experiment.add_image(
            "train/render_mesh", render_mesh, self.current_epoch
        )

        self.logger.experiment.add_mesh(
            "train/mesh_gt", input_batch["points"][:6], global_step=self.current_epoch
        )
        self.logger.experiment.add_mesh(
            "train/mesh_pred", pred["pred_coord"][2][:6], global_step=self.current_epoch
        )

        output_reconst = self.renderer.visualize_reconstruction_images(
            input_images=input_batch["images"].cpu(),
            reconstructed_images=pred["reconst"].detach().cpu(),
        )
        self.logger.experiment.add_image(
            "train/output_reconst", output_reconst, self.current_epoch
        )

    def on_validation_epoch_start(self):
        self.validation_sample_outputs = []

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

        if len(self.validation_sample_outputs) < self.num_samples_to_visualize:
            self.validation_sample_outputs.append((batch, pred))

        return loss

    def on_validation_epoch_end(self):
        input_batch, pred = self.validation_sample_outputs[0]
        # render_mesh = self.renderer.p2m_batch_visualize(
        #     input_batch, pred, self.ellipsoid.faces
        # )
        # self.logger.experiment.add_image(
        #     "val/render_mesh", render_mesh, self.current_epoch
        # )

        self.logger.experiment.add_mesh(
            "val/mesh_gt", input_batch["points"][:6], global_step=self.current_epoch
        )
        self.logger.experiment.add_mesh(
            "val/mesh_pred", pred["pred_coord"][2][:6], global_step=self.current_epoch
        )

        output_reconst = self.renderer.visualize_reconstruction_images(
            input_images=input_batch["images"].cpu(),
            reconstructed_images=pred["reconst"].detach().cpu(),
        )
        self.logger.experiment.add_image(
            "val/output_reconst", output_reconst, self.current_epoch
        )

    def configure_optimizers(self):
        return self.optimizer
