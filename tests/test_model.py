import os
import os.path
import sys
from cgi import test
import src.models.model
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from src.models.model import MyAwesomeModel
from src.data.make_dataset import MNISTdata
from tests import _PROJECT_ROOT

def test_model():
    model = MyAwesomeModel()
    #model.load_state_dict(torch.load(model_filepath))
    #print(_PROJECT_ROOT + "/data/processed/test.pth")
    test_data = torch.load(_PROJECT_ROOT + "/data/processed/test.pth")
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    in_target = 784
    out_target = 10

    with torch.no_grad():
        model.eval()
        for images in testloader:
            images = images.view(images.shape[0], -1)
            out = model(images)
            assert len(images) == in_target, "Model does not have a correct input shape"
            assert len(out) == out_target, "Model does not have a correct output shape"


