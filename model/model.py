import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import _addinden

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm            
        )
        return super(Conv2dWithConstraint, self).forward(x)


class SE_block(nn.Module):
    def __init__(self , in_channels , ratio = 8 ):               
        super(SE_block , self).__init__()
        self.avg_pool   =     nn.AdaptiveAvgPool1d(output_size=1)   
        self.fc_linear1 =     nn.Linear(in_features= in_channels ,out_features= in_channels//ratio ,bias=False)  
        self.relu       =     nn.ReLU()
        self.fc_linear2 =     nn.Linear(in_features= in_channels//ratio ,out_features= in_channels ,bias=False)   
        self.sigmoid    =     nn.Sigmoid()
        self.conv       =     nn.Conv2d(in_channels=1 ,out_channels=1 ,kernel_size=(1,8),padding='same',bias=False)
        self.x_weight   =     nn.Parameter(torch.randn(8, requires_grad=True))
        self.weight_sigmoid =  nn.Sigmoid()

    def forward(self, inputs):
        _ , _ , _ , samp = inputs.size()
        weight = self.weight_sigmoid(self.x_weight)
        weight = torch.unsqueeze(weight ,1).repeat(1 , samp)
        weight = torch.unsqueeze(torch.unsqueeze(weight,0),0)
        
        x = torch.squeeze(inputs, dim=1)                          
        x = self.avg_pool(x)                               
        x = torch.squeeze(x)                                     

        x = self.fc_linear1(x)                                  
        x = self.relu(x)
        x = self.fc_linear2(x)                                  
        x = self.sigmoid(x)

        x = torch.unsqueeze(torch.unsqueeze(x, dim=1), dim=3)     
        y = torch.unsqueeze(self.avg_pool(torch.squeeze(self.conv(inputs) ,dim=1)) , dim=1)
        f_xy = torch.sigmoid( x*weight + y*(1-weight))
        outputs = f_xy * inputs                                   
                                                               
        return outputs    


    
class EEGNet(nn.Module):
    def __init__(self , n_classes=4 , channels=60 , samples = 151 , dropoutRate=0.5 , 
                 kernelLength=64 , kernelLength2=16 , F1=8 , D=2 , F2=16):
        super(EEGNet , self).__init__()                     
        self.samples = samples

        self.SE_layer           = SE_block(in_channels= 8 , ratio= 8)
                                                                                                                           
        self.block1             = nn.Sequential(
                                        nn.Conv2d(in_channels=1 ,out_channels=F1 ,kernel_size=(1,4),
                                                  stride=1 ,padding='same' ,bias=False),                              
                                        nn.BatchNorm2d(num_features=F1),                                        
        )
        self.block2             = nn.Sequential(
                                        nn.Conv2d(in_channels=1 ,out_channels=F1 ,kernel_size=(1,64),
                                                  stride=1 ,padding='same' ,bias=False),                              
                                        nn.BatchNorm2d(num_features=F1),                                        
        )   
        self.block3             = nn.Sequential(
                                        nn.Conv2d(in_channels=1 ,out_channels=F1 ,kernel_size=(1,128),
                                                  stride=1 ,padding='same' ,bias=False),                              
                                        nn.BatchNorm2d(num_features=F1),                                        
        )      
                                         
        self.pointConv2d        = nn.Sequential(
                                        nn.Conv2d(in_channels=F1*3 , out_channels=F1 , kernel_size=(1,1) , stride=1 , padding='same' , bias=False),
                                        nn.BatchNorm2d(num_features=F1),
                                        nn.ELU(),
                                        nn.Dropout(p=dropoutRate)
        )                                                                                   
        self.DepthwiseConv2D    = nn.Sequential(
                                        Conv2dWithConstraint(in_channels=F1 ,out_channels=F1*D ,kernel_size=(channels,1),
                                                             stride=1 ,padding=(0,0) ,groups=F1 ,bias=False ,max_norm=1),  
                                        nn.BatchNorm2d(num_features=F1*D ,momentum=0.01 ,eps=1e-3),
                                        nn.ELU(),
                                        nn.AvgPool2d(kernel_size=(1,4) ,stride=4),  
                                        nn.Dropout(p=dropoutRate)
        )

        self.SeparableConv2d    = nn.Sequential(
                                        nn.Conv2d(in_channels=F1*D ,out_channels=F1*D ,kernel_size=(1,kernelLength2),
                                                  stride=1 ,padding='same' ,groups=F1*D ,bias=False),  
                                        nn.Conv2d(in_channels=F1*D ,out_channels=F1*D ,kernel_size=(1,1) ,stride=1 ,
                                                  padding=(0,0) ,groups=1 ,bias=False),
                                        nn.BatchNorm2d(num_features=F1*D, momentum=0.01, affine=True, eps=1e-3),
                                        nn.ELU(),
                                        nn.AvgPool2d(kernel_size=(1,8) ,stride=(1,8)), 
                                        nn.Dropout(p=dropoutRate)
        )



        self.Flatten_class      = nn.Sequential(
                                        nn.Flatten(), 
                                        nn.Linear(in_features=(F2*(samples//32)) ,out_features=n_classes),
                                        nn.Softmax(dim=-1)
        )


    def forward(self ,x):
        x_o1_o2 = x [:, :, -2:, :] 
        x_tilde = x [:, :, :-2:, :]
        x_tilde = self.SE_layer(x_tilde)
        x0 = torch.cat((x_tilde , x_o1_o2),dim=2)
        x1 = self.block1(x0) 
        x2 = self.block2(x0)
        x3 = self.block3(x0)

        x4 = torch.cat((x1, x2, x3), dim=1) 
        x4 = self.pointConv2d(x4)



        x4 = self.DepthwiseConv2D(x4)
        x4 = self.SeparableConv2d(x4)
        x4 = self.Flatten_class(x4)
        return x4


    
def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

def torch_summarize(model, show_weights=True, show_parameters=True):
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


#%%
###============================ Initialization parameters ============================###
channels = 10
samples = 5000

###============================ main function ============================###
def main():
    input = torch.randn(32, 1, channels, samples)
    model = EEGNet(samples=5000)
    out = model(input)
    print('===============================================================')
    print('out', out.shape)
    print('model', model)


if __name__ == "__main__":
    main()