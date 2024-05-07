import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 205996929

def ID2():
    '''
        Personal ID of the second student. Fill this only if you were allowed to submit in pairs, Otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

class CNN(nn.Module):
    def __init__(self): # Do NOT change the signature of this function
        super(CNN, self).__init__()
        n = 6
        self.n = n
        kernel_size = 3
        padding = (kernel_size - 1) // 2

        # Convolution methods:
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = n,
        kernel_size=kernel_size, padding=padding)

        self.conv2 = nn.Conv2d(in_channels = n, out_channels = 2*n,
        kernel_size=kernel_size, padding=padding)

        self.conv3 = nn.Conv2d(in_channels = 2*n, out_channels = 4*n,
        kernel_size=kernel_size, padding=padding)
 
        self.conv4 = nn.Conv2d(in_channels = 4*n, out_channels = 8*n,
        kernel_size=kernel_size, padding=padding)

        # Normalization methods:
        self.batch_norm1 = nn.BatchNorm2d(n)
        self.batch_norm2 = nn.BatchNorm2d(2*n)
        self.batch_norm3 = nn.BatchNorm2d(4*n)
        self.batch_norm4 = nn.BatchNorm2d(8*n)

        # Droupout method:
        self.dropout = nn.Dropout(0.5)  

        # Fully Connected Layers:
        self.fc1 = nn.Linear(14 * 28 * 8*n, 100)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self,inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor.
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # Cycle 1
        out = self.conv1(inp)                      
        out = F.relu(out)
        out = F.dropout(out)
        out = F.max_pool2d(out, kernel_size=2)     
        out = self.batch_norm1(out)

        # Cycle 2
        out = self.conv2(out)                      
        out = F.relu(out)
        #out = F.dropout(out)
        out = F.max_pool2d(out, kernel_size=2)     
        out = self.batch_norm2(out)
        
        # Cycle 3
        out = self.conv3(out)                      
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)     
        out = self.batch_norm3(out)
        
        # Cycle 4
        out = self.conv4(out)                      
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=2)     
        out = self.batch_norm4(out)
        
        out = out.reshape(-1, self.n*8 * 28 * 14)  # Flatten the output tensor
        out = self.fc1(out)                        
        out = F.relu(out)
        out = self.fc2(out)                        

        return out

class CNNChannel(nn.Module):
    def __init__(self):# Do NOT change the signature of this function
        super(CNNChannel, self).__init__()
        n = 12
        self.n = n
        kernel_size = 3
        padding = (kernel_size - 1) // 2

        # Convolution methods:
        self.conv1 = nn.Conv2d(in_channels = 6, out_channels=n,
        kernel_size = kernel_size, padding = padding)

        self.conv2 = nn.Conv2d(in_channels = n,out_channels = 2*n,
        kernel_size = kernel_size, padding = padding)

        self.conv3 = nn.Conv2d(in_channels = 2*n, out_channels = 4*n,
        kernel_size = kernel_size, padding = padding)

        self.conv4 = nn.Conv2d(in_channels = 4*n,out_channels = 8*n,
        kernel_size = kernel_size, padding = padding)

        # Normalization methods:
        self.batch_norm1 = nn.BatchNorm2d(n)
        self.batch_norm2 = nn.BatchNorm2d(2*n)
        self.batch_norm3 = nn.BatchNorm2d(4*n)
        self.batch_norm4 = nn.BatchNorm2d(8*n)

        # Droupout method:
        self.dropout = nn.Dropout(0.5)  

        # Fully Connected Layers:
        self.fc1 = nn.Linear(14 * 14 * 8*n, 100)
        self.fc2 = nn.Linear(100, 2)

    # TODO: complete this class
    def forward(self, inp):# Do NOT change the signature of this function
        '''
          prerequests:
          parameter inp: the input image, pytorch tensor
          inp.shape == (N,3,448,224):
            N   := batch size
            3   := RGB channels
            448 := Height
            224 := Width
          
          return output, pytorch tensor
          output.shape == (N,2):
            N := batch size
            2 := same/different pair
        '''
        # Reshape from height concetinating to channel concetinating
        first_half = inp[:, :, :224, :  ] # (N, 3, 224, 224)
        second_half = inp[:, :, 224:, :]  # (N, 3, 224, 224)
        inp = torch.cat((first_half, second_half), dim=1) # (N ,6, 224, 224)

        # Cycle 1
        out = self.conv1(inp)                      
        out = F.relu(out)
        #out = F.dropout(out)
        out = F.max_pool2d(out, kernel_size = 2)     
        out = self.batch_norm1(out)

        # Cycle 2
        out = self.conv2(out)                      
        out = F.relu(out)
        #out = F.dropout(out)
        out = F.max_pool2d(out, kernel_size = 2)     
        out = self.batch_norm2(out)
        
        # Cycle 3
        out = self.conv3(out)                      
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size = 2)     
        out = self.batch_norm3(out)
        
        # Cycle 4
        out = self.conv4(out)                      
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size = 2)     
        out = self.batch_norm4(out)
        
        out = out.reshape(-1, self.n*8 * 14 * 14)  # Flatten the output tensor
        out = self.fc1(out)                        
        out = F.relu(out)
        out = self.fc2(out)                        

        return out