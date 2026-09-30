import torch
from approach.common import Base

class Approach(Base):

    def criterion(self, output, targets, masks = None):
        '''
        Args:
            task: Task ID.
            output: An array of shape [N, 1, W, H]. N = Number of predictions, W = Width, H = Height.
            targets: An array of shape [N, 1, W, H]. N = Number of labels, W = Width, H = Height.
        
        Returns:
            Cross entropy loss.
        '''

        return self.loss_func(output, targets), torch.tensor(0)


