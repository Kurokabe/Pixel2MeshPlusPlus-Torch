from jsonargparse import lazy_instance
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger

from p2mpp.data.datamodule import DataModule
from p2mpp.models.lightning_module_p2mpp import LightningModuleP2MPP


def cli_main():
    cli = LightningCLI(
        LightningModuleP2MPP,
        DataModule,
        trainer_defaults={
            "logger": lazy_instance(
                TensorBoardLogger, save_dir="./lightning_logs", name="p2mpp"
            )
        },
        # trainer_defaults={"callbacks": [checkpoint_callback]},
    )

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()
