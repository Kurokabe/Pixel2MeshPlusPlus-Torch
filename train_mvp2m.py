from pytorch_lightning.cli import LightningCLI

from p2mpp.data.datamodule import DataModule
from p2mpp.models.lightning_module_net import LightningModuleNet


def cli_main():
    cli = LightningCLI(
        LightningModuleNet,
        DataModule,
        # trainer_defaults={"callbacks": [checkpoint_callback]},
    )

    # summary(cli.model, input_size=(2, 3, 128, 256))


if __name__ == "__main__":
    cli_main()
