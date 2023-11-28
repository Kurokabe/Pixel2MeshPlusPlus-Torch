from typing import Literal, Tuple

import pytorch_lightning as pl
import torch
from loguru import logger

from p2mpp.configs import LossConfig, NetworkConfig, OptimConfig, P2MPPConfig
from p2mpp.configs.config import BaseMeshConfig
from p2mpp.models.lightning_module_net import LightningModuleNet
from p2mpp.models.losses.p2m import P2MLoss
from p2mpp.models.losses.p2mpp import P2MPPLoss
from p2mpp.models.mesh.ellipsoid import Ellipsoid
from p2mpp.models.mesh.icosahedron import Icosahedron
from p2mpp.models.p2m import P2MModel
from p2mpp.models.p2mpp import P2MPPModel
from p2mpp.utils.vis.renderer import MeshRenderer


class LightningModuleP2MPP(pl.LightningModule):
    def __init__(
        self,
        p2m_ckpt_path: str,
        p2m_config: NetworkConfig,
        p2mpp_config: P2MPPConfig,
        p2mpp_loss_config: LossConfig,
        optim_config: OptimConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.p2m_model = P2MModel(
            hidden_dim=p2m_config.hidden_dim,
            coord_dim=p2m_config.coord_dim,
            last_hidden_dim=p2m_config.last_hidden_dim,
            gconv_activation=p2m_config.gconv_activation,
            backbone=p2m_config.backbone,
            align_with_tensorflow=p2m_config.align_with_tensorflow,
            base_mesh_config=p2m_config.base_mesh_config,
            z_threshold=p2m_config.z_threshold,
            camera_f=p2m_config.camera_f,
            camera_c=p2m_config.camera_c,
        )
        self.p2m_model.eval()

        self.restore_p2m_ckpt(p2m_ckpt_path)

        hypothesis_shape = self.create_hypothesis_shape(p2mpp_config.hypothesis_shape)
        self.p2mpp_model = P2MPPModel(
            align_with_tensorflow=p2mpp_config.align_with_tensorflow,
            base_mesh_config=p2mpp_config.base_mesh_config,
            z_threshold=p2mpp_config.z_threshold,
            camera_f=p2mpp_config.camera_f,
            camera_c=p2mpp_config.camera_c,
            backbone=p2mpp_config.backbone,
            hypothesis_shape=hypothesis_shape,
            nn_encoder_ckpt_path=p2m_ckpt_path,
        )
        self.num_iterations = p2mpp_config.num_iterations

        basemesh_config = p2mpp_config.base_mesh_config
        self.ellipsoid = Ellipsoid(basemesh_config.mesh_pose)

        self.criterion = P2MPPLoss(loss_config=p2mpp_loss_config, ellipsoid=self.ellipsoid)

        self.optimizer = torch.optim.Adam(
            params=list(self.p2mpp_model.parameters()),
            lr=optim_config.lr,
            betas=(optim_config.adam_beta1, optim_config.adam_beta2),
            weight_decay=optim_config.weight_decay,
        )

        self.train_sample_outputs = []
        self.validation_sample_outputs = []
        self.num_samples_to_visualize = 4

    def create_hypothesis_shape(self, name: str):
        if name == "icosahedron":
            return Icosahedron()
        else:
            raise ValueError(f"Unknown hypothesis shape: {name}")

    def forward(self, images, poses):
        # with torch.no_grad():
        #     coarse_pred = self.p2m_model(images.detach().clone(), poses)
        batch_size = images.size(0)
        coarse_pred = self.p2m_model.init_pts.data.unsqueeze(0).expand(batch_size, -1, -1)

        # coarse_vertices = coarse_pred["pred_coord"][2]
        coarse_vertices = coarse_pred.cuda()
        input_vertices = coarse_vertices
        for i in range(self.num_iterations):
            fine_pred = self.p2mpp_model(input_vertices, images.clone(), poses)
            input_vertices = fine_pred["pred_coord"]

        # fine_pred = self.p2mpp_model(coarse_vertices, images, poses)
        return fine_pred, coarse_pred

    def on_train_epoch_start(self):
        self.train_sample_outputs = []

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        poses = batch["poses"]

        fine_pred, coarse_pred = self.forward(images, poses)

        loss, loss_summary = self.criterion(fine_pred, batch)

        # Prefix all keys from loss_summary with 'train/'
        loss_summary = {f"train/{k}": v for k, v in loss_summary.items()}

        self.log_dict(
            loss_summary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if len(self.train_sample_outputs) < self.num_samples_to_visualize:
            self.train_sample_outputs.append((batch, fine_pred, coarse_pred))

        return loss

    def on_train_epoch_end(self):
        input_batch, fine_pred, coarse_pred = self.train_sample_outputs[0]
        faces = self.ellipsoid.faces[0].unsqueeze(0).expand(self.num_samples_to_visualize, -1, -1)

        self.logger.experiment.add_mesh(
            "train/mesh_gt", input_batch["points"][:self.num_samples_to_visualize], global_step=self.current_epoch
        )
        self.logger.experiment.add_mesh(
            "train/mesh_coarse_pred",
            # coarse_pred["pred_coord"][2][:self.num_samples_to_visualize],
            coarse_pred[:self.num_samples_to_visualize],
            faces=faces,
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_mesh(
            "train/mesh_fine_pred",
            fine_pred["pred_coord"][:self.num_samples_to_visualize],
            faces=faces,
            global_step=self.current_epoch,
        )

    def on_validation_epoch_start(self):
        self.validation_sample_outputs = []

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        poses = batch["poses"]

        fine_pred, coarse_pred = self.forward(images, poses)

        loss, loss_summary = self.criterion(fine_pred, batch)

        # Prefix all keys from loss_summary with 'train/'
        loss_summary = {f"val/{k}": v for k, v in loss_summary.items()}

        self.log_dict(
            loss_summary, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        if len(self.validation_sample_outputs) < self.num_samples_to_visualize:
            self.validation_sample_outputs.append((batch, fine_pred, coarse_pred))

        return loss

    def on_validation_epoch_end(self):
        input_batch, fine_pred, coarse_pred = self.validation_sample_outputs[0]
        
        faces = self.ellipsoid.faces[0].unsqueeze(0).expand(self.num_samples_to_visualize, -1, -1)

        self.logger.experiment.add_mesh(
            "val/mesh_gt", input_batch["points"][:self.num_samples_to_visualize], 
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_mesh(
            "val/mesh_coarse_pred",
            # coarse_pred["pred_coord"][2][:self.num_samples_to_visualize],
            coarse_pred[:self.num_samples_to_visualize],
            faces=faces,
            global_step=self.current_epoch,
        )
        self.logger.experiment.add_mesh(
            "val/mesh_fine_pred",
            fine_pred["pred_coord"][:self.num_samples_to_visualize],
            faces=faces,
            global_step=self.current_epoch,
        )

    def restore_p2m_ckpt(self, p2m_ckpt_path: str):
        p2m_ckpt = torch.load(p2m_ckpt_path)
        state_dict = {
            key.replace("model.", ""): value
            for key, value in p2m_ckpt["state_dict"].items()
            if key.startswith("model.")
        }
        self.p2m_model.load_state_dict(state_dict)

    def configure_optimizers(self):
        return self.optimizer
