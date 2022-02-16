import logging
import torch
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import models
from aimet_common.utils import AimetLogger
from aimet_common.defs import QuantScheme
from aimet_torch.utils import create_fake_data_loader
from aimet_torch.model_preparer import prepare_model
from aimet_torch.quantsim import QuantizationSimModel


AimetLogger.set_level_for_all_areas(logging.INFO)

print("AIMET setup ok")