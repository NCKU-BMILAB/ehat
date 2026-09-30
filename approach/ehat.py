import numpy as np
import torch
from approach.common import Base

class Approach(Base):

    def criterion(self, output, targets, masks):
        '''Loss function
        Args:
            output: prediction
            targets: ground truth
            masks: attention masks
        
        Return:
            loss with regularization loss and regularization loss
        '''
        #initialize
        reg = 0
        count = 0

        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                # dynamic regularization strength
                d = 1 / (1 - m.sum() / np.prod(m.size()).item())
                d = torch.clamp(d, 0.5, 10)
                w = 1 - mp
                d = d * (m * w).sum()
                reg += d
                count += w.sum()
        #fist task
        else:
            for m in masks:
                d = 1 / (1 - m.sum() / np.prod(m.size()).item())
                d = torch.clamp(d, 0.5, 10)
                d = d * m.sum()
                reg += d
                count += np.prod(m.size()).item()
        reg /= count

        return self.loss_func(output, targets) + reg, reg


