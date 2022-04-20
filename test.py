import torch
from torch import Tensor

prediction = torch.rand(16, 1, 256, 256) 
target = torch.rand(16, 1, 256, 256)


def forward(prediction: Tensor, target: Tensor):
        prediction = torch.sigmoid(prediction)
    
        loss = []
        flat_target = target.flatten()
        flat_prediction = prediction.flatten()
        for idx in range(len(flat_prediction.tolist())):
            if flat_target[idx] == 1:
                loss.append(1 - flat_prediction[idx])
            else:
                loss.append(0.1)

        
        return torch.mean(torch.tensor(loss))


forward(prediction, target)