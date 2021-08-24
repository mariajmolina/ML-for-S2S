import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    
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
        
        # input
        self.inplay = nn.Linear(in_features=self.input_length, out_features=self.output_length)
        
        # encoder
        self.layer1 = nn.Linear(in_features=self.output_length,           out_features=256  * self.neuron_multiplier)
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
