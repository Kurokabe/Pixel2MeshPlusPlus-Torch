from pathlib import Path

# Models root
MODELS_ROOT = Path(__file__).parent / "models"

ELLIPSOID_PATH = MODELS_ROOT / "ellipsoid" / "info_ellipsoid.dat"

PRETRAINED_WEIGHTS_PATH = {
    "vgg16": MODELS_ROOT / "pretrained" / "vgg16-397923af.pth",
    "resnet50": MODELS_ROOT / "pretrained" / "resnet50-19c8e357.pth",
    "vgg16p2m": MODELS_ROOT / "pretrained" / "vgg16-p2m.pth",
}

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
