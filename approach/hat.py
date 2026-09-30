import numpy as np
from approach.common import Base

class Approach(Base):

    def criterion(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks,self.mask_pre):
                aux = 1 - mp
                reg += (m * aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count

        return self.loss_func(outputs,targets) + self.lamb * reg, reg


