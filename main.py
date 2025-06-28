import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from models import FighterProfileLSTM, FightPredictor

df = pd.read_csv("UFC Fighter Dataset.csv")
of = pd.read_csv("UFC Outcomes.csv")
outcomes_dict = of.groupby('Fighter Name').apply(lambda x: x.values.tolist(), include_groups=False).to_dict()
train_test_fighters = open("Train_Test Fighters.txt")
train_list = train_test_fighters.readlines()

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
        index_split = (int(total_fights * .66))
        train_fights = (data_transfig[:index_split])
        test_fights = (data_transfig[index_split:])
    return train_fights, test_fights

INPUT_SIZE = 17 #input size of the LSTM model
HIDDEN_SIZE = 17 #hidden size of the LSTM model
train_fights = [] #list of fights to train models
test_fights = [] #list of fights to test model
num_epochs = 100 #number of epochs to train the model
for i in train_list:
    fighter_name = i.strip()
    train_data, test_data = test_train_dataloader(fighter_name, outcomes_dict[fighter_name])
    train_fights.extend(train_data)
    test_fights.extend(test_data)

lstm_model = FighterProfileLSTM(INPUT_SIZE, HIDDEN_SIZE)
predictor_model = FightPredictor(HIDDEN_SIZE)
optimizer = torch.optim.Adam(
    list(lstm_model.parameters()) + list(predictor_model.parameters()), lr=5e-4)
#optimizer that holds the parameters of the lstm model and predictor model, this gets changed as the program continues running

criterion = nn.BCEWithLogitsLoss() #Binary Cross Entropy Loss
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95) #Learning rate scheduler to reduce learning rate every 10 epochs

def train():
    for epoch in range(num_epochs):
        lstm_model.train()
        predictor_model.train()
        total_loss = 0
        for i in train_fights:
            #clear gradients (Biases from any past fights it analyzed)
            optimizer.zero_grad()
            #grab the outcome for model to train on
            outcome = torch.tensor([float(i[2])], dtype=torch.float32)
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
            #print(f"Processing {fighter_a} vs {fighter_b}")
            #Predict outcome
            prediction = predictor_model(fighter_a_profile, fighter_b_profile).view(1)
            loss = criterion(prediction, outcome)
            #back propogation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss {total_loss:.4f}")
    


def test():
    lstm_model.eval()
    predictor_model.eval()
    correct = 0
    total = 0
    win = 0
    loss = 0
    with torch.no_grad():
        for i in test_fights:
            outcome = torch.tensor(float(i[2]), dtype=torch.float32)
            win = (win + 1) if int(i[2]) == 1 else win
            loss = (loss + 1) if int(i[2]) == 0 else loss


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
            print(f"Predicting {fighter_a} vs {fighter_b} = {outcome.item()}")
            prediction = predictor_model(fighter_a_profile, fighter_b_profile).view(1)
            prob = torch.sigmoid(prediction)  # Convert logits to probabilities
            predicted = (prob > 0.5).float() #converts the probability to a binary prediction
            correct += (predicted == outcome).sum().item()
            total += 1
    accuracy = correct / total
    print(f"Losses: {loss} Wins: {win}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy * 100

train()
new_accuracy = test()
with open("Current Accuracy.txt", "r") as read_list:
    accuracy = float(read_list.read().strip())

# Compare and overwrite if the new accuracy if it reaches a certain accuracy threshold
if new_accuracy >= 75.0:
    with open("Current Accuracy.txt", "w") as write_list:
        write_list.write(f"{new_accuracy:.2f}")
        torch.save(lstm_model.state_dict(), 'lstm_model.pth')
        torch.save(predictor_model.state_dict(), 'predictor_model.pth')

