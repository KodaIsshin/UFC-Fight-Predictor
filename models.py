import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class FighterProfileLSTM(nn.Module):
    """
            LSTM Neural Network for Fighter Profile
                        MAIN FUNCTION
        ___________________________________________________
        The main function of this LSTM Neural network is to 
        generate fighter profiles based on the sequences given
        they don't identify the fighter based on name, so 
        there is no name value bias or rank bias, purely
        statistics and a profile given from the history given
        _____________________________________________________
        TRAIN:
        The training of this model will just be feeding roughly 180 fighters and predicting match ups
        of their 6 fights.
        
        TEST:
        There is no test feature for this, it will adjust it self based on more data.  
    """
    def __init__(self, input_size, hidden_size, num_layers=3, dropout=0.6):
        super(FighterProfileLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #LSTM LAYER OF NEURAL NETWORK
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x, lengths):

        #Packing sequences for new fighters (1-5 fights instead of the six)
        #This is also for training, using history to train lstm to create fighter profiles for fighters that are prominent
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        #LSTM OUTPUT
        #OUTPUT OF TENSOR, (We only need the hidden state to find the LSTM interp of fighter)
        _, (hidden, _) = self.lstm(packed_input) # hidden: (num_laters, batch_size, hidden_size)

        #last hidden state contains profile after iterating through sequence
        fighter_profile = hidden[-1]

        return fighter_profile
    


class FightPredictor(nn.Module):
    """
        FIGHT PREDICTOR NEURAL NETWORK
                MAIN FUNCTION
        ______________________________

        Neural network that grabs fighter
        profiles and then tries to predict
        the outcome of the fight given the 
        profile
        ______________________________
        TRAIN:
        The training of this model is from
        the Train_Test txt file, iterates
        through and then grabs the fight
        history to get tests

        TEST:
        Tests itself on recent fights of
        fighters in train_test txt file.
    """
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(FightPredictor, self).__init__()
        self.fc1 = nn.Linear(3 * input_size, hidden_size) #profile a, profile b, and diff between profiles
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()

    def forward(self, profile_a, profile_b):
        #compute difference between profiles"
        diff = torch.abs(profile_a-profile_b)

        #concatenate the profiles and the difference together
        combined = torch.cat([profile_a, profile_b, diff], dim=-1)

        #pass it through the layers
        #NOTE: Finally adding dropout to the layers to prevent overfitting
        x = self.activation(self.fc1(combined))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

