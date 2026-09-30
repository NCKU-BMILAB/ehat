import torch.nn as nn
import torch


class Net(nn.Module):
    '''U-net
    Args:
        channel: input channel
        taskcla: number of tasks
    '''

    def __init__(self, channel, taskcla):
        super(Net, self).__init__()
        self.channel = channel
        self.taskcla = taskcla
        self.gate = nn.Sigmoid()

        self.c1 = Down(self.channel, 64, self.taskcla, 0)
        self.c2 = Down(64, 128, self.taskcla, 0.2)
        self.c3 = Down(128, 256, self.taskcla, 0.3)
        self.c4 = Down(256, 512, self.taskcla, 0.3)
        self.c5 = Down(512, 1024, self.taskcla, 0.4)
        self.b = ConvBlock(1024, 2048 // 2, self.taskcla, 0.5, bottom = True)
        self.u1 = Up(2048, 1024 // 2, self.taskcla, 0.5)
        self.u2 = Up(1024, 512 // 2, self.taskcla, 0.4)
        self.u3 = Up(512, 256 // 2, self.taskcla, 0.4)
        self.u4 = Up(256, 128 // 2, self.taskcla, 0.3)
        self.u5 = Up(128, 64, self.taskcla, 0.2)
        self.last = nn.ModuleList()

        for t, n in self.taskcla:
            self.last.append(nn.Conv2d(in_channels = 64, out_channels = n, kernel_size = 1))

    def forward(self, t, x, s = 1):

        x, cat1, c1_gc1, c1_gc2 = self.c1(t, x, s)
        x, cat2, c2_gc1, c2_gc2 = self.c2(t, x, s)
        x, cat3, c3_gc1, c3_gc2 = self.c3(t, x, s)
        x, cat4, c4_gc1, c4_gc2 = self.c4(t, x, s)
        x, cat5, c5_gc1, c5_gc2 = self.c5(t, x, s)
        x, b_gc1, b_gc2 = self.b(t, x, s)
        x, u1_gc1, u1_gc2 = self.u1(t, x, cat5, s)
        x, u2_gc1, u2_gc2 = self.u2(t, x, cat4, s)
        x, u3_gc1, u3_gc2 = self.u3(t, x, cat3, s)
        x, u4_gc1, u4_gc2 = self.u4(t, x, cat2, s)
        x, u5_gc1, u5_gc2 = self.u5(t, x, cat1, s)
        y = self.last[t](x)

        return y, [c1_gc1, c1_gc2, c2_gc1, c2_gc2, c3_gc1, c3_gc2, c4_gc1, c4_gc2, c5_gc1, c5_gc2,\
                b_gc1, b_gc2, u1_gc1, u1_gc2, u2_gc1, u2_gc2, u3_gc1, u3_gc2,\
                 u4_gc1, u4_gc2, u5_gc1, u5_gc2]

    def mask(self, t, s = 1):
        masks = []
        for name, layer in self.named_modules():
                if "ec" in name:
                    masks.append(self.gate(s*layer(t)))
        return masks

    def get_view_for(self, n, masks):
        '''Expand attention masks to size of feature space
        Args:
            n: layer name
            masks: attention masks

        Return:
            if n exists: Expanded masks
            if n not exist: None
        '''
        c1_gc1, c1_gc2, c2_gc1, c2_gc2, c3_gc1, c3_gc2, c4_gc1, c4_gc2, c5_gc1, c5_gc2,\
                b_gc1, b_gc2, u1_gc1, u1_gc2, u2_gc1, u2_gc2, u3_gc1, u3_gc2,\
                 u4_gc1, u4_gc2, u5_gc1, u5_gc2 = masks

        #expansion
        #u5
        if n == "u5.conv.conv2.weight":
            post = u5_gc2.data.view(-1, 1, 1, 1).expand_as(self.u5.conv.conv2.weight)
            pre = u5_gc1.data.view(1, -1, 1, 1).expand_as(self.u5.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "u5.conv.conv1.weight":
            post = u5_gc1.data.view(-1, 1, 1, 1).expand_as(self.u5.conv.conv1.weight)
            return post

        elif n == "u5.conv.conv2.bias":
            return u5_gc2.data.view(-1)

        elif n == "u5.conv.conv1.bias":
            return u5_gc1.data.view(-1)

        #u4
        elif n == "u4.conv.conv2.weight":
            post = u4_gc2.data.view(-1, 1, 1, 1).expand_as(self.u4.conv.conv2.weight)
            pre = u4_gc1.data.view(1, -1, 1, 1).expand_as(self.u4.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "u4.conv.conv1.weight":
            post = u4_gc1.data.view(-1, 1, 1, 1).expand_as(self.u4.conv.conv1.weight)
            return post

        elif n == "u4.conv.conv2.bias":
            return u4_gc2.data.view(-1)

        elif n == "u4.conv.conv1.bias":
            return u4_gc1.data.view(-1)

        #u3
        elif n == "u3.conv.conv2.weight":
            post = u3_gc2.data.view(-1, 1, 1, 1).expand_as(self.u3.conv.conv2.weight)
            pre = u3_gc1.data.view(1, -1, 1, 1).expand_as(self.u3.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "u3.conv.conv1.weight":
            post = u3_gc1.data.view(-1, 1, 1, 1).expand_as(self.u3.conv.conv1.weight)
            return post

        elif n == "u3.conv.conv2.bias":
            return u3_gc2.data.view(-1)

        elif n == "u3.conv.conv1.bias":
            return u3_gc1.data.view(-1)

        #u2
        elif n == "u2.conv.conv2.weight":
            post = u2_gc2.data.view(-1, 1, 1, 1).expand_as(self.u2.conv.conv2.weight)
            pre = u2_gc1.data.view(1, -1, 1, 1).expand_as(self.u2.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "u2.conv.conv1.weight":
            post = u2_gc1.data.view(-1, 1, 1, 1).expand_as(self.u2.conv.conv1.weight)
            return post

        elif n == "u2.conv.conv2.bias":
            return u2_gc2.data.view(-1)
        
        elif n == "u2.conv.conv1.bias":
            return u2_gc1.data.view(-1)

        #u1
        elif n == "u1.conv.conv2.weight":
            post = u1_gc2.data.view(-1, 1, 1, 1).expand_as(self.u1.conv.conv2.weight)
            pre = u1_gc1.data.view(1, -1, 1, 1).expand_as(self.u1.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "u1.conv.conv1.weight":
            post = u1_gc1.data.view(-1, 1, 1, 1).expand_as(self.u1.conv.conv1.weight)
            return post
        
        elif n == "u1.conv.conv2.bias":
            return u1_gc2.data.view(-1)

        elif n == "u1.conv.conv1.bias":
            return u1_gc1.data.view(-1)

        #bottom

        elif n == "b.conv2.weight":
            post = b_gc2.data.view(-1, 1, 1, 1).expand_as(self.b.conv2.weight)
            pre = b_gc1.data.view(1, -1, 1, 1).expand_as(self.b.conv2.weight)
            return torch.min(post, pre)

        elif n == "b.conv1.weight":
            post = b_gc1.data.view(-1, 1, 1, 1).expand_as(self.b.conv1.weight)
            pre = c5_gc2.data.view(1, -1, 1, 1).expand_as(self.b.conv1.weight)
            return torch.min(post, pre)
        
        elif n == "b.conv2.bias":
            return b_gc2.data.view(-1)

        elif n == "b.conv1.bias":
            return b_gc1.data.view(-1)

        #contraction
        #c5
        elif n == "c5.conv.conv2.weight":
            post = c5_gc2.data.view(-1, 1, 1, 1).expand_as(self.c5.conv.conv2.weight)
            pre = c5_gc1.data.view(1, -1, 1, 1).expand_as(self.c5.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "c5.conv.conv1.weight":
            post = c5_gc1.data.view(-1, 1, 1, 1).expand_as(self.c5.conv.conv1.weight)
            pre = c4_gc2.data.view(1, -1, 1, 1).expand_as(self.c5.conv.conv1.weight)
            return torch.min(post, pre)
        
        elif n == "c5.conv.conv2.bias":
            return c5_gc2.data.view(-1)

        elif n == "c5.conv.conv1.bias":
            return c5_gc1.data.view(-1)

        #c4
        elif n == "c4.conv.conv2.weight":
            post = c4_gc2.data.view(-1, 1, 1, 1).expand_as(self.c4.conv.conv2.weight)
            pre = c4_gc1.data.view(1, -1, 1, 1).expand_as(self.c4.conv.conv2.weight)
            return torch.min(post, pre)
        
        elif n == "c4.conv.conv1.weight":
            post = c4_gc1.data.view(-1, 1, 1, 1).expand_as(self.c4.conv.conv1.weight)
            pre = c3_gc2.data.view(1, -1, 1, 1).expand_as(self.c4.conv.conv1.weight)
            return torch.min(post, pre)

        elif n == "c4.conv.conv2.bias":
            return c4_gc2.data.view(-1)
        
        elif n == "c4.conv.conv1.bias":
            return c4_gc1.data.view(-1)

        #c3
        elif n == "c3.conv.conv2.weight":
            post = c3_gc2.data.view(-1, 1, 1, 1).expand_as(self.c3.conv.conv2.weight)
            pre = c3_gc1.data.view(1, -1, 1, 1).expand_as(self.c3.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "c3.conv.conv1.weight":
            post = c3_gc1.data.view(-1, 1, 1, 1).expand_as(self.c3.conv.conv1.weight)
            pre = c2_gc2.data.view(1, -1, 1, 1).expand_as(self.c3.conv.conv1.weight)
            return torch.min(post, pre)
        
        elif n == "c3.conv.conv2.bias":
            return c3_gc2.data.view(-1)

        elif n == "c3.conv.conv1.bias":
            return c3_gc1.data.view(-1)

        #c2
        elif n == "c2.conv.conv2.weight":
            post = c2_gc2.data.view(-1, 1, 1, 1).expand_as(self.c2.conv.conv2.weight)
            pre = c2_gc1.data.view(1, -1, 1, 1).expand_as(self.c2.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "c2.conv.conv1.weight":
            post = c2_gc1.data.view(-1, 1, 1, 1).expand_as(self.c2.conv.conv1.weight)
            pre = c1_gc2.data.view(1, -1, 1, 1).expand_as(self.c2.conv.conv1.weight)
            return torch.min(post, pre)
            
        elif n == "c2.conv.conv2.bias":
            return c2_gc2.data.view(-1)

        elif n == "c2.conv.conv1.bias":
            return c2_gc1.data.view(-1)

        #c1
        elif n == "c1.conv.conv2.weight":

            post = c1_gc2.data.view(-1, 1, 1, 1).expand_as(self.c1.conv.conv2.weight)
            pre = c1_gc1.data.view(1, -1, 1, 1).expand_as(self.c1.conv.conv2.weight)
            return torch.min(post, pre)

        elif n == "c1.conv.conv1.weight":
            return c1_gc1.data.view(-1, 1, 1, 1).expand_as(self.c1.conv.conv1.weight)

        elif n == "c1.conv.conv2.bias":
            return c1_gc2.data.view(-1)

        elif n == "c1.conv.conv1.bias":
            return c1_gc1.data.view(-1)

        return None


class ConvBlock(nn.Module):
    '''Convoluational blocks in U-net
    Args:
        Cin: number of input channel
        Cout: number of output channel
        taskcla: number of tasks
        drop_ration: drop out ration
        bottom: boolean. True: bottom of U-net
        mid: number of middel channel (i.e., Cin -> mid -> Cout)
    '''

    def __init__(self, Cin, Cout, taskcla, drop_ratio = 0, bottom = False, mid = None):
        super(ConvBlock, self).__init__()
        if not mid:
            mid = Cout
        self.taskcla = taskcla
        self.drop_ratio = drop_ratio
        self.bottom = bottom
        self.relu= nn.ReLU(inplace = True)
        self.gate = nn.Sigmoid()

        self.conv1 = nn.Conv2d(in_channels = Cin, out_channels = mid, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = mid, out_channels = Cout, kernel_size = 3, padding = 1)
        self.ec1 = nn.Embedding(len(self.taskcla), mid)
        self.ec2 = nn.Embedding(len(self.taskcla), Cout)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)
        
    def forward(self, t, x, s = 1):
        if self.bottom:
            gc1, gc2 = self.mask(t, s)
        else:
            gc1 = self.mask(t, s)
        x = self.relu(self.conv1(x))
        x = x * gc1.view(1, -1, 1, 1).expand_as(x)
        x = self.relu(self.conv2(x))
        if self.drop_ratio > 0:
            x = self.drop(x)
        if self.bottom:
            x = x * gc2.view(1, -1, 1, 1).expand_as(x)

            return x, gc1, gc2

        return x, gc1

    def mask(self, t, s = 1):
        ''' attention mask
        Args:
            t: number of tasks
            s: hyperparameter
        '''
        gc1 = self.gate(s*self.ec1(t))
        if self.bottom:
            gc2 = self.gate(s*self.ec2(t))
            return gc1, gc2
        return gc1

class Down(nn.Module):
    
    def __init__(self, Cin, Cout, taskcla, drop_ratio):
        super(Down, self).__init__()

        self.taskcla = taskcla
        self.drop_ratio = drop_ratio
        self.gate = nn.Sigmoid()
        self.conv = ConvBlock(Cin, Cout, self.taskcla)
        self.maxpool =  nn.MaxPool2d(2)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)

    def forward(self, t, x, s = 1):
        gc2 = self.mask(t, s)
        cat, gc1 = self.conv(t, x, s)
        if self.drop_ratio > 0:
            x = self.maxpool(self.drop(cat))
        else:
            x = self.maxpool(cat)
        x = x * gc2.view(1, -1, 1, 1).expand_as(x)

        return x, cat, gc1, gc2


    def mask(self, t, s = 1):
        gc2 = self.gate(s*self.conv.ec2(t))

        return gc2



class Up(nn.Module):
    def __init__(self, Cin, Cout, taskcla, drop_ratio):
        super(Up, self).__init__()
        
        self.taskcla = taskcla
        self.drop_ratio = drop_ratio
        self.relu= nn.ReLU(inplace = True)
        self.gate = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(Cin, Cout, self.taskcla, drop_ratio = 0, bottom = False, mid = Cin // 2)
        if self.drop_ratio > 0:
            self.drop = nn.Dropout2d(drop_ratio)
    
    def forward(self, t, x1, cat, s = 1):
        gc2 = self.mask(t, s)
        x1 = self.up(x1)
        x1 = torch.cat([cat, x1], dim = 1)
        x1, gc1 = self.conv(t, x1, s)
        if self.drop_ratio > 0:
            x1 = self.drop(x1)
        x1 = x1 * gc2.view(1, -1, 1, 1).expand_as(x1)

        return x1, gc1, gc2   #up-sampling, conv1, conv2
        
    def mask(self, t, s = 1):
        gc2 = self.gate(s*self.conv.ec2(t))

        return gc2
