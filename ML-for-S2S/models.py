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
    
    def __init__(self):
        
        super(CNNAutoencoder, self).__init__()
        
        # encoder layers
        
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # decoder layers
        
        self.dec1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2)  
        self.dec2 = nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2)
        self.dec3 = nn.ConvTranspose2d(16, 32, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        
        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        
        # encode
        
        x = F.relu(self.enc1(x))
        x = self.pool(x)
        
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        
        x = F.relu(self.enc3(x))
        x = self.pool(x)
        
        x = F.relu(self.enc4(x))
        x = self.pool(x) # the latent space representation
        
        # decode
        
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        
        x = torch.sigmoid(self.out(x))
        
        return x
