import torch
import torch.nn as NN
import torchvision.transforms.functional as TF

# conv 3x3, ReLU (+ batch norm)
class DoubleConv(NN.Module):
    def __init__(self, in_c, out_c) -> None:
        super(DoubleConv, self).__init__()

        self.conv = NN.Sequential(
            NN.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            NN.BatchNorm2d(out_c),
            NN.ReLU(inplace=True),

            NN.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            NN.BatchNorm2d(out_c),
            NN.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)



class UNET(NN.Module):
    r"""
    |    |    |                                                                      ||    |    |    | 
    |    |    |                                                                      ||    |    |    | 
    | -> | -> |    ------------------------------------------------------------>     || -> | -> | -> |
    |  1 |    |                                5                                     ||  1 |    |  4 | 
    |    |    |                                                                      ||    |    |    | 
    |    |    |                                                                      ||    |    |    | 
        
        | 1                                                                               ^
        v                                                                                 | 3 
    
        |    |    |                                                            ||    |    |
        |    |    |                                                            ||    |    |
        | -> | -> |   ----------------------------------------------------->   || -> | -> |
        |  1 |    |                             5                              ||  1 |    |
        |    |    |                                                            ||    |    |
    
            | 2                                                                    ^
            v                                                                      | 3 
    
            |    |    |                                                ||    |    |
            | -> | -> |   ------------------------------------------>  || -> | -> |
            |    |    |                        5                       ||    |    |
            |    |    |                                                ||    |    |
    
                    | 2                                               ^
                    v                                                 | 3
                    
                        |    |    |                     ||    |    |
                        | -> | -> |  -----------------> || -> | -> |
                        |    |    |           5         ||    |    |
    
                                | 2                             ^
                                v                               | 3
    
                                | -> | -> |  ----->  || -> | -> |  
                                |    |    |     5    ||    |    |
                                        
                                        | 2         ^
                                        v           | 3
    
                                        | -> | -> |
    
                                    
    
    1 ... conv 3x3 ReLU + batch norm
    2 ... max pool 2x2
    3 ... up-conv 2x2
    4 ... conv 1x1
    5 ... skip connection (copy and crop)
    
    https://www.cs.cmu.edu/~jeanoh/16-785/papers/ronnenberger-miccai2015-u-net.pdf
    """

    # out_c defines binary or more classification
    def __init__(self, in_c=3, out_c=1, features=[64, 128, 256, 512]) -> None:
        super(UNET, self).__init__()
        self.downs = NN.ModuleList()
        self.ups = NN.ModuleList()

        # max pool 2x2
        self.pool = NN.MaxPool2d(kernel_size=2, stride=2)

        # Downward direction
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upward direction
        for feature in reversed(features):
            self.ups.append(NN.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # From x channels to the number of output channels
        self.final_conv = NN.Conv2d(features[0], out_c, kernel_size=1)


    def forward(self, x):
        # Copy and crop -> skip connection
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # We are now at the bottom of the architecture -> reverse the skip_connections
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            # Because of step-size 2
            skip_connection = skip_connections[idx // 2]

            # Reshape if the current image-size is not divisible by 16
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)