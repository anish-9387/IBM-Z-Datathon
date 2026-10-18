"""Frozen chest X-ray classifier: ResNet-50 (TorchXRayVision 'resnet50-res512-all' weights)
with a single pneumonia output.

Pure PyTorch (no torchvision) so it runs on s390x where torchvision wheels may be missing.
Layer names match torchvision's resnet50, so the public TorchXRayVision state dict loads as-is.
"""
import io
import hashlib
import numpy as np
import torch
import torch.nn as nn

from . import config

XRV_WEIGHTS_URL = ("https://github.com/mlmed/torchxrayvision/releases/download/v1/"
                   "pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt")
# The model card says these weights were trained on "pc-nih-rsna-siim-vin": RSNA is in the
# pre-training mix, so RSNA test metrics of this backbone are optimistic (data overlap).
XRV_LABELS = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema',
              'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
              'Mass', 'Hernia', 'Lung Lesion', 'Fracture', 'Lung Opacity', 'Enlarged Cardiomediastinum']
PNEUMONIA_IDX = XRV_LABELS.index("Pneumonia")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


class ResNet50(nn.Module):
    """torchvision-compatible ResNet-50 with a 1-channel stem."""

    def __init__(self, num_classes=18):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make(64, 3)
        self.layer2 = self._make(128, 4, 2)
        self.layer3 = self._make(256, 6, 2)
        self.layer4 = self._make(512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def _make(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                 nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, down)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        return self.fc(self.features(x))


class CXRPneumonia(nn.Module):
    """Backbone + one pneumonia logit -> probability. `head` is (weight[2048], bias, temperature)."""

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50(num_classes=18)
        self.head = nn.Linear(2048, 1)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):                      # x: (N,1,512,512) in xrv range [-1024, 1024]
        return torch.sigmoid(self.head(self.backbone.features(x)) / self.temperature).squeeze(1)


# ---------------------------------------------------------------- preprocessing (fixed shape)
def to_fixed_u8(img) -> np.ndarray:
    """Any grayscale/RGB image or array -> uint8 (512,512): centre-crop to square, resize."""
    from PIL import Image
    if not isinstance(img, Image.Image):
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
            a = a.astype(np.uint8)
        img = Image.fromarray(a)
    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8, copy=True)   # writable, contiguous


def u8_into_tensor(u8: np.ndarray, out: torch.Tensor) -> torch.Tensor:
    """Write uint8 pixels into a pre-allocated float tensor in xrv range, in place."""
    out.copy_(torch.from_numpy(u8).view(1, 1, config.IMG_SIZE, config.IMG_SIZE))
    return out.mul_(2048.0 / 255.0).sub_(1024.0)


# ---------------------------------------------------------------- building / loading
def build_zero_shot(xrv_state_path) -> CXRPneumonia:
    """Frozen model using xrv's own 'Pneumonia' output unit as the head (no RSNA fine-tuning)."""
    m = CXRPneumonia()
    try:   # public weights were saved on little-endian machines; say so explicitly on s390x
        from torch.serialization import LoadEndianness, set_default_load_endianness
        set_default_load_endianness(LoadEndianness.LITTLE)
    except ImportError:
        pass
    sd = torch.load(xrv_state_path, map_location="cpu", weights_only=True)
    m.backbone.load_state_dict(sd)
    with torch.no_grad():
        m.head.weight.copy_(m.backbone.fc.weight[PNEUMONIA_IDX:PNEUMONIA_IDX + 1])
        m.head.bias.copy_(m.backbone.fc.bias[PNEUMONIA_IDX:PNEUMONIA_IDX + 1])
    return m.eval()


def save_frozen(model: CXRPneumonia, path) -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    data = buf.getvalue()
    with open(path, "wb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def load_frozen_bytes(data: bytes) -> CXRPneumonia:
    """Load from bytes that were already hashed (no re-read of the file = no TOCTOU)."""
    m = CXRPneumonia()
    m.load_state_dict(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
    return m.eval()


@torch.inference_mode()
def predict_u8(model: CXRPneumonia, u8: np.ndarray) -> float:
    t = torch.empty(1, 1, config.IMG_SIZE, config.IMG_SIZE)
    try:
        return float(model(u8_into_tensor(u8, t))[0])
    finally:
        t.zero_()
