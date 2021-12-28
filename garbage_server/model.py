import typing
import torch
import torch.nn as nn
from torchvision import models as torch_models
from .constants import MODEL_PATH

LEARNING_RATE_DEFAULT = 0.001


class SequentialNetwork(nn.Module):

    def __init__(self, layers: typing.Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        net_out = self(x)
        return torch.max(net_out, 1)[1]


def get_trained_model():
    resnet18 = torch_models.resnet18(pretrained=False)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 6)
    model = SequentialNetwork([resnet18])
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model
