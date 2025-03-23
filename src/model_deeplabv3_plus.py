import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from torchgeo.models import resnet50
from convlstm import ConvLSTM

class TemporalPlanetDeepLab(nn.Module):
    def __init__(self, num_classes=8, hidden_dim=256):
        super().__init__()
        base = deeplabv3_resnet50(weights=None, num_classes=num_classes)

        # Load full Planet‑pretrained ResNet50
        full = resnet50(pretrained="planet")

        # Patch conv1 to accept 4 bands
        old = full.conv1
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
        full.conv1 = new

        # Build encoder sequentially (feature extractor)
        self.encoder = nn.Sequential(
            full.conv1, full.bn1, full.act1, full.maxpool,
            full.layer1, full.layer2, full.layer3, full.layer4
        )
        base.backbone.body = self.encoder
        self.seg_head = base.classifier

        self.convlstm = ConvLSTM(input_dim=2048, hidden_dim=hidden_dim, kernel_size=(3,3), num_layers=1, batch_first=False)
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        B, T, _, H, W = x.shape

        feats = torch.stack([self.encoder(x[:, t]) for t in range(T)], dim=1)  # [B, T, 2048, h, w]
        feats_seq = feats.permute(1, 0, 2, 3, 4)                               # [T, B, 2048, h, w]

        layer_outputs, _ = self.convlstm(feats_seq)
        output = layer_outputs[0]                                              # [T, B, hidden, h, w]
        last = output[-1]                                                       # [B, hidden, h, w]

        logits = self.classifier(last)                                          # [B, num_classes, h, w]
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
