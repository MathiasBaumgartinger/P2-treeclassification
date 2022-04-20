from torch import Tensor
import torch
import torch.nn as nn
import numpy as np

class WeightedLoss (nn.Module):
    def __init__(self) -> None:
        super(WeightedLoss, self).__init__()
    
    def forward(self, prediction: Tensor, target: Tensor):
        prediction = torch.sigmoid(prediction)
    
        loss = []
        flat_target = target.flatten()
        flat_prediction = prediction.flatten()
        for idx in range(len(flat_prediction.tolist())):
            if flat_target[idx] == 1:
                loss.append(1 - flat_prediction[idx])
            else:
                loss.append(0.1)

        
        return torch.mean(torch.tensor(loss)).to(prediction.get_device())