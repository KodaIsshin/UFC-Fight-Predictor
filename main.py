import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import UFCDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

df = pd.read_csv("UFC Dataset 1.csv")
dataset = UFCDataset(df)



class FighterPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(FighterPredictorLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        #fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, fight_history, seq_lengths):
        #packing the sequence 
        packed_input = pack_padded_sequence(fight_history, seq_lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (hn, cn) = self.lstm(packed_input)

        #LSTM forward pass (hidden and cell states can be initialized as None for the first batch)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        #take the outputs from the last time step of each sequence 
        idxs  = (seq_lengths - 1).view(-1, 1).expand(len(seq_lengths), lstm_out.size(2))
        time_step_outputs = lstm_out.gather(1, idxs.unsqueeze(1).squeeze(1))

        out = self.fc(time_step_outputs)
        
        return out
    


def test_train_dataloader(fighter_data):
    train_data = []
    test_data = []
    for fighter_name, fighter_history, labels in fighter_data:
        total_fights = len(fighter_history)
        if total_fights == 1:
            train_fights = (fighter_name, fighter_history, labels)
            test_fights = ()
        elif total_fights == 2:
            train_fights = (fighter_name, fighter_history[:1], labels[:1])
            test_fights = (fighter_name, fighter_history[1:], labels[1:])
        else:
            index_split = (int(total_fights * .8) - 1)
            train_fights = (fighter_name, fighter_history[:index_split], labels[:index_split])
            test_fights = (fighter_name, fighter_history[index_split:], labels[index_split:])

        train_data.append(train_fights)
        test_data.append(test_fights)

    return [train_data, test_data]


INPUT_SIZE = 18
HIDDEN_SIZE = 50
OUTPUT_SIZE = 1
NUM_LAYERS = 2

#uses the dataloader method above to organize the test data and train data into an array
#train data loader and test dataloader are made to hold their respective tensor lists
fighter_dataloader = test_train_dataloader(dataset)
train_dataloader = fighter_dataloader[0]
test_dataloader = fighter_dataloader[1]

#model
model = FighterPredictorLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS)

#loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)





def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for fighter_name, fighter_history, labels in train_loader:
            #split the values from the packed batch into the three values, name, fight_history, labels
            fighter_history, labels = fighter_history.to(device), labels.to(device)
            #compute the seq lengths
            seq_lengths = len(fighter_history)


