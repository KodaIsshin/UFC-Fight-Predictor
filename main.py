import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import UFCDataset, find_input_sequences
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

df = pd.read_csv("UFC Fighter Dataset.csv")
dataset = UFCDataset(df)
of = pd.read_csv("UFC Outcomes.csv")
outcomes_dict = of.groupby('Fighter A').apply(lambda x: x.values.tolist(), include_groups=False).to_dict()
train_test_fighters = open("Train_Test Fighters.txt")
train_list = train_test_fighters.readlines()


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
        The training of this model will just be feeding roughly 100 fighters and predicting match ups
        of their 6 fights.
        
        TEST:
        There is no test feature for this, it will adjust it self based on more data.  
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
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
    def __init__(self):
        pass


def test_train_dataloader(fighter_name, fighter_data):
    data_transfig = fighter_data[1:]
    total_fights = len(data_transfig) 
    if total_fights == 1:
        train_fights = (data_transfig)
        test_fights = ()
    elif total_fights == 2:
        train_fights = (data_transfig[:1])
        test_fights = (data_transfig[1:])
    else:
        index_split = (int(total_fights * .8))
        train_fights = (data_transfig[:index_split])
        test_fights = (data_transfig[index_split:])

    train_fights.append(fighter_name)
    test_fights.append(fighter_name)


    return train_fights, test_fights


INPUT_SIZE = 17
HIDDEN_SIZE = 17

for i in train_list[:3]:
    fighter_name = i.strip()
    train_data, test_data = test_train_dataloader(fighter_name, outcomes_dict[fighter_name])
    print(f"{i} Train Data: ")
    print(train_data)
    print(f"{i} Test Data:")
    print(test_data)
    
        