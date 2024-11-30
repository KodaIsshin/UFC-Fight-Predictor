import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import UFCDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

df = pd.read_csv("UFC Fighter Dataset.csv")
dataset = UFCDataset(df)
of = pd.read_csv("UFC Outcomes.csv")
outcomes_dict = of.groupby('Fighter Name').apply(lambda x: x.values.tolist(), include_groups=False).to_dict()
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
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.4):
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
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, profile_a, profile_b):
        #compute difference between profiles"
        diff = torch.abs(profile_a-profile_b)

        #concatenate the profiles and the difference together
        combined = torch.cat([profile_a, profile_b, diff], dim=-1)

        #pass it through the layers
        x = self.activation(self.fc1(combined))
        x = self.activation(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



def find_input_sequences(fighter_name, fight_indice):
    fight_history = outcomes_dict[fighter_name]
    fighter_a_input = dataset.fighter_name_index[fighter_name][:fight_indice-1]
    fighter_b = fight_history[fight_indice-1][1]
    opponent_history = outcomes_dict[fighter_b]
    fighter_b_input = None
    for i in opponent_history:
        if i[1] == fighter_name:
            if (dataset.fighter_name_index[fighter_b][i[3]-1][6:11] == dataset.fighter_name_index[fighter_name][fight_indice-1][11:16]).all():
                fighter_b_input = dataset.fighter_name_index[fighter_b][:i[3]-1]
                break
    
    if fighter_b_input is None:
        fighter_b_input = [dataset.fighter_name_index[fighter_b][0]]
    fighter_a_input = [torch.tensor(i, dtype=torch.float32) for i in fighter_a_input] if len(fighter_a_input) > 1 else torch.tensor(dataset.fighter_name_index[fighter_name][0], dtype=torch.float32)
    fighter_b_input = [torch.tensor(j,dtype=torch.float32) for j in fighter_b_input] if len(fighter_b_input) > 1 else torch.tensor(dataset.fighter_name_index[fighter_b][0], dtype=torch.float32)
    return fighter_name, fighter_b,fighter_a_input, fighter_b_input



def test_train_dataloader(fighter_name, fighter_data):
    data_transfig = fighter_data[1:]
    total_fights = len(data_transfig)
    if total_fights <= 1:
        train_fights = (data_transfig)
        test_fights = ()
    elif total_fights == 2:
        train_fights = (data_transfig[:1])
        test_fights = (data_transfig[1:])
    else:
        index_split = (int(total_fights * .8))
        train_fights = (data_transfig[:index_split])
        test_fights = (data_transfig[index_split:])

    return train_fights, test_fights


INPUT_SIZE = 17 #input size of the LSTM model
HIDDEN_SIZE = 17 #hidden size of the LSTM model
train_fights = [] #list of fights to train models
test_fights = [] #list of fights to test model
num_epochs = 100
for i in train_list:
    fighter_name = i.strip()
    train_data, test_data = test_train_dataloader(fighter_name, outcomes_dict[fighter_name])
    train_fights.extend(train_data)
    test_fights.extend(test_data)

lstm_model = FighterProfileLSTM(INPUT_SIZE, HIDDEN_SIZE)
predictor_model = FightPredictor(HIDDEN_SIZE)

criterion = nn.BCELoss() #Binary Cross Entropy Loss
optimizer = torch.optim.Adam(
    list(lstm_model.parameters()) + list(predictor_model.parameters()), lr=0.001)

#optimizer that holds the parameters of the lstm model and predictor model, this gets changed as the program continues running

def train():
    for epoch in range(num_epochs):
        lstm_model.train()
        predictor_model.train()
        total_loss = 0
        for i in train_fights:
            #clear gradients (Biases from any past fights it analyzed)
            optimizer.zero_grad()
            #grab the outcome for model to train on
            outcome = torch.tensor(float(i[2]), dtype=torch.float32)
            #get the input sequence for the fighters fighting
            fighter_a, fighter_b, fighter_a_input, fighter_b_input = find_input_sequences(i[0], i[3])
            #generate the profiles for fighter a and b (will judge outcome based on fighter A)
            if isinstance(fighter_a_input, list):
                fighter_a_input = torch.stack(fighter_a_input, dim=0)
            if isinstance(fighter_b_input, list):
                fighter_b_input = torch.stack(fighter_b_input, dim=0)
            lengths_a = torch.tensor([len(fighter_a_input)])
            lengths_b = torch.tensor([len(fighter_b_input)])
            fighter_a_profile = lstm_model(fighter_a_input.unsqueeze(0), lengths_a)
            fighter_b_profile = lstm_model(fighter_b_input.unsqueeze(0), lengths_b)
            print(f"Processing {fighter_a} vs {fighter_b}")
            #Predict outcome
            prediction = predictor_model(fighter_a_profile, fighter_b_profile).squeeze()

            #Compute the loss (how far from actual 'prediction')
            loss = criterion(prediction, outcome)

            #back propogation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss {total_loss:.4f}")



def test():
    lstm_model.eval()
    predictor_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in test_fights:
            outcome = torch.tensor(float(i[2]), dtype=torch.float32)
            fighter_a, fighter_b, fighter_a_input, fighter_b_input = find_input_sequences(i[0], i[3])
            #generate the profiles for fighter a and b (will judge outcome based on fighter A)
            if isinstance(fighter_a_input, list):
                fighter_a_input = torch.stack(fighter_a_input, dim=0)
            if isinstance(fighter_b_input, list):
                fighter_b_input = torch.stack(fighter_b_input, dim=0)
            lengths_a = torch.tensor([len(fighter_a_input)])
            lengths_b = torch.tensor([len(fighter_b_input)])
            fighter_a_profile = lstm_model(fighter_a_input.unsqueeze(0), lengths_a)
            fighter_b_profile = lstm_model(fighter_b_input.unsqueeze(0), lengths_b)
            print(f"Processing {fighter_a} vs {fighter_b}")
            
            prediction = predictor_model(fighter_a_profile, fighter_b_profile).squeeze()
            predicted = (prediction > 0.5).float() #converts the probability to a binary prediction

            correct += (predicted == outcome).sum().item()
            total += 1
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


train()
test()
torch.save(lstm_model.state_dict(), "lstm_model.pth")
torch.save(predictor_model.state_dict(), "predictor_model.pth")

def make_prediction():
    pass
    
        