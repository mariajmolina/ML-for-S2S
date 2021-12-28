import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch_funcs

"""

Module contains several pytorch model architectures being tested/implemented in S2S machine learning work.

Author: Maria J. Molina, NCAR (molina@ucar.edu)

"""

class Autoencoder(nn.Module):
    
    def __init__(self, input_length, output_length=None, neuron_multiplier=1, sigmoid=False, drop=False, drop_pct=0.3):
        
        """
        Dense autoencoder. 
        
        Args:
            input_length (int): Length (i.e., size) of input sample.
            output_length (int): Length of output sample. Defaults to None, leave as default if 
                                 input and output are equal size.
            neuron_multiplier (int): Number to augment the set number of neurons. Defaults to 1.
            sigmoid (boolean): Defaults to False. Leave as default if output is nn.Linear (not sigmoid).
            drop (boolean): Defaults to False. True activates dropout for each layer except final.
            drop_pct (float): Amount of dropout to use (if drop is True). Defaults to 0.3.
            
        """
        super(Autoencoder, self).__init__()
        
        self.input_length = input_length
        
        if not output_length:
            self.output_length = input_length
            
        if output_length:
            self.output_length = output_length
        
        self.neuron_multiplier = neuron_multiplier
        self.sigmoid = sigmoid
        self.drop = drop
        self.drop_pct = drop_pct
        
        # encoder
        self.layer1 = nn.Linear(in_features=self.input_length,            out_features=256  * self.neuron_multiplier)
        self.layer2 = nn.Linear(in_features=256 * self.neuron_multiplier, out_features=128  * self.neuron_multiplier)
        self.layer3 = nn.Linear(in_features=128 * self.neuron_multiplier, out_features=64   * self.neuron_multiplier)
        
        # decoder 
        self.layer4 = nn.Linear(in_features=64  * self.neuron_multiplier, out_features=128 * self.neuron_multiplier)
        self.layer5 = nn.Linear(in_features=128 * self.neuron_multiplier, out_features=256 * self.neuron_multiplier)
        self.layer6 = nn.Linear(in_features=256 * self.neuron_multiplier, out_features=self.output_length)
        
        # output
        self.out = nn.Linear(in_features=self.output_length, out_features=self.output_length)
        
        
    def forward(self, x):
        
        if self.drop:
            drp = nn.Dropout(p=self.drop_pct)
        
        x = F.relu(self.layer1(x))
            
        if self.drop:
            x = drp(x)
        
        x = F.relu(self.layer2(x))
        
        if self.drop:
            x = drp(x)
            
        x = F.relu(self.layer3(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer4(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer5(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer6(x))
        
        if self.drop:
            x = drp(x)
        
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        
        if not self.sigmoid:
            x = self.out(x)
        
        return x


class DeepAutoencoder(nn.Module):
    
    def __init__(self, input_length, output_length=None, neuron_multiplier=1, sigmoid=False, drop=False, drop_pct=0.3):
        
        """
        Dense deep autoencoder. 
        
        Args:
            input_length (int): Length (i.e., size) of input sample.
            output_length (int): Length of output sample. Defaults to None, leave as default if 
                                 input and output are equal size.
            neuron_multiplier (int): Number to augment the set number of neurons. Defaults to 1.
            sigmoid (boolean): Defaults to False. Leave as default if output is nn.Linear (not sigmoid).
            drop (boolean): Defaults to False. True activates dropout for each layer except final.
            drop_pct (float): Amount of dropout to use (if drop is True). Defaults to 0.3.
            
        """
        super(DeepAutoencoder, self).__init__()
        
        self.input_length = input_length
        
        if not output_length:
            self.output_length = input_length
            
        if output_length:
            self.output_length = output_length
        
        self.neuron_multiplier = neuron_multiplier
        self.sigmoid = sigmoid
        self.drop = drop
        self.drop_pct = drop_pct
        
        # input
        self.inplay = nn.Linear(in_features=self.input_length, out_features=2 * (256  * self.neuron_multiplier))
        
        # encoder
        self.layer1 = nn.Linear(in_features=2 * (256  * self.neuron_multiplier), out_features=256  * self.neuron_multiplier)
        self.layer2 = nn.Linear(in_features=256 * self.neuron_multiplier, out_features=128  * self.neuron_multiplier)
        self.layer3 = nn.Linear(in_features=128 * self.neuron_multiplier, out_features=64   * self.neuron_multiplier)
        
        # decoder 
        self.layer4 = nn.Linear(in_features=64  * self.neuron_multiplier, out_features=128 * self.neuron_multiplier)
        self.layer5 = nn.Linear(in_features=128 * self.neuron_multiplier, out_features=256 * self.neuron_multiplier)
        self.layer6 = nn.Linear(in_features=256 * self.neuron_multiplier, out_features=self.output_length)
        
        # output
        self.out = nn.Linear(in_features=self.output_length, out_features=self.output_length)
        
        
    def forward(self, x):
        
        if self.drop:
            drp = nn.Dropout(p=self.drop_pct)
            
        x = F.relu(self.inplay(x))
        
        if self.drop:
            x = drp(x)
        
        x = F.relu(self.layer1(x))
            
        if self.drop:
            x = drp(x)
        
        x = F.relu(self.layer2(x))
        
        if self.drop:
            x = drp(x)
            
        x = F.relu(self.layer3(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer4(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer5(x))
        
        if self.drop:
            x = drp(x)
                
        x = F.relu(self.layer6(x))
        
        if self.drop:
            x = drp(x)
        
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        
        if not self.sigmoid:
            x = self.out(x)
        
        return x

    
class DeeperAutoencoder(nn.Module):
    
    def __init__(self, input_length, output_length=None, neuron_multiplier=1, sigmoid=False, drop=False, drop_pct=0.3):
        
        """
        Dense deeper autoencoder. 
        
        Args:
            input_length (int): Length (i.e., size) of input sample.
            output_length (int): Length of output sample. Defaults to None, leave as default if 
                                 input and output are equal size.
            neuron_multiplier (int): Number to augment the set number of neurons. Defaults to 1.
            sigmoid (boolean): Defaults to False. Leave as default if output is nn.Linear (not sigmoid).
            drop (boolean): Defaults to False. True activates dropout for each layer except final.
            drop_pct (float): Amount of dropout to use (if drop is True). Defaults to 0.3.
            
        """
        super(DeeperAutoencoder, self).__init__()
        
        self.input_length = input_length
        
        if not output_length:
            self.output_length = input_length
            
        if output_length:
            self.output_length = output_length
        
        self.neuron_multiplier = neuron_multiplier
        self.sigmoid = sigmoid
        self.drop = drop
        self.drop_pct = drop_pct
        
        # input
        
        # encoder
        self.layer1 = nn.Linear(in_features=self.input_length,                out_features=4 * 256  * self.neuron_multiplier)
        self.layer2 = nn.Linear(in_features=4 * 256 * self.neuron_multiplier, out_features=256  * self.neuron_multiplier)
        #self.layer3 = nn.Linear(in_features=256 * self.neuron_multiplier,     out_features=192  * self.neuron_multiplier)
        
        # decoder
        #self.layer4 = nn.Linear(in_features=192 * self.neuron_multiplier, out_features=256 * self.neuron_multiplier)
        self.layer5 = nn.Linear(in_features=256 * self.neuron_multiplier, out_features=4 * 256  * self.neuron_multiplier)
        self.layer6 = nn.Linear(in_features=4 * 256  * self.neuron_multiplier, out_features=self.output_length)
        self.layer7 = nn.Linear(in_features=self.output_length, out_features=int(self.input_length / 3) + self.input_length)
        #self.layer8 = nn.Linear(in_features=int(self.input_length / 3) + self.input_length, out_features=self.output_length)
        
        # output
        self.out = nn.Linear(in_features=int(self.input_length / 3) + self.input_length, out_features=self.output_length)
        
        
    def forward(self, x):
        
        x = F.relu(self.layer1(x))
            
        x = F.relu(self.layer2(x))
        
        #x = F.relu(self.layer3(x))
                
        #x = F.relu(self.layer4(x))
                
        x = F.relu(self.layer5(x))
            
        x = F.relu(self.layer6(x))
            
        x = F.relu(self.layer7(x))
            
        #x = F.relu(self.layer8(x))
        
        if self.sigmoid:
            x = torch.sigmoid(self.out(x))
        
        if not self.sigmoid:
            x = self.out(x)
        
        return x
    
    
class LSTMModel(nn.Module):
    
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, drop_out=0):
        
        """
        LSTM. 
        
        Args:
            num_classes (int): number of output classes
            input_size (int): length of input
            hidden_size (int): size of hidden state
            num_layers (int): number of stacked lstm layers
            seq_length (int): length of sequence (i.e., length of time series)
            drop_out (float): amount of dropout to use. defaults to 0.
            
        """
        super(LSTMModel, self).__init__()
        
        self.num_classes = num_classes
        self.input_size  = input_size
        self.hidden_size = hidden_size            
        self.num_layers  = num_layers  
        self.seq_length  = seq_length        
        self.dropout     = drop_out

        self.lstm = nn.LSTM(input_size  = input_size, 
                            hidden_size = hidden_size,
                            num_layers  = num_layers, 
                            batch_first = True,
                            dropout     = drop_out)
        
        self.classifier = nn.Linear(hidden_size, num_classes)  

        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        device = torch_funcs.get_device()
        
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        
        out = hn[-1]
        out = self.classifier(out)
        
        return out


class CNNAutoencoder(nn.Module):
    
    def __init__(self, depth_0=1, depth_1=64, depth_2=32, depth_3=16, lastdepth=1):
        
        super(CNNAutoencoder, self).__init__()
        
        self.depth_0 = depth_0 
        self.depth_1 = depth_1 
        self.depth_2 = depth_2 
        self.depth_3 = depth_3
        
        # encoder layers
        self.enc1 = nn.Conv2d(self.depth_0, self.depth_1, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(self.depth_1, self.depth_2, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv2d(self.depth_2, self.depth_3, kernel_size=3, stride=1, padding=1)
        
        # latent space
        self.ltnt = nn.Conv2d(self.depth_3, self.depth_3, kernel_size=3, stride=1, padding=1)
        
        # decoder layers
        self.dec3 = nn.ConvTranspose2d(self.depth_3, self.depth_2, kernel_size=1, stride=1)
        self.dec2 = nn.ConvTranspose2d(self.depth_2, self.depth_1, kernel_size=1, stride=1)
        self.dec1 = nn.ConvTranspose2d(self.depth_1, self.depth_0, kernel_size=1, stride=1)
        
        # output
        self.out = nn.Conv2d(self.depth_0, 1, 1)
        
        # pooling and unpooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=False)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
    def forward(self, x):
        #print(x.size())
        # encode
        x = F.relu(self.enc1(x))
        x, indx1 = self.pool(x)
        #print(x.size())
        x = F.relu(self.enc2(x))
        x, indx2 = self.pool(x)
        #print(x.size())
        x = F.relu(self.enc3(x))
        x, indx3 = self.pool(x)
        #print(x.size())
        # latent space
        x = F.relu(self.ltnt(x))
        #print(x.size())
        # decode
        x = self.unpool(x, indx3, output_size=indx2.size())
        x = F.relu(self.dec3(x))
        #print(x.size())
        x = self.unpool(x, indx2, output_size=indx1.size())
        x = F.relu(self.dec2(x))
        #print(x.size())
        x = self.unpool(x, indx1)
        x = F.relu(self.dec1(x))
        #print(x.size())
        # output
        x = self.out(x)
        #print(x.size())
        return x
