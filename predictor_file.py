import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from data_processing import dataset
from models import FighterProfileLSTM, FightPredictor
import check_input


def get_fighter_input():
    while True:
        fighter_a = str(input("Enter Fighter 1: "))
        if fighter_a in dataset.fighter_name_index:
            a_list = dataset.fighter_name_index[fighter_a]
            a_input = [torch.tensor(i, dtype = torch.float32) for i in a_list]
        else:
            print(f"{fighter_a} is not in UFC dataset, try again.")
            continue
        while True:
            fighter_b = str(input("Enter Fighter 2: "))
            if fighter_b == fighter_a:
                print(f"Fighter cannot fight themself, try again")
                continue
            if fighter_b in dataset.fighter_name_index:
                b_list = dataset.fighter_name_index[fighter_b]
                b_input = [torch.tensor(i, dtype=torch.float32) for i in b_list]
                break
            else:
                print(f"{fighter_b} is not in UFC dataset, try again.")
                continue
        return fighter_a, fighter_b, a_input, b_input


def main():
    INPUT_SIZE = 17
    HIDDEN_SIZE = 17
    lstm_model = FighterProfileLSTM(INPUT_SIZE, HIDDEN_SIZE)
    predictor_model = FightPredictor(HIDDEN_SIZE)
    #Loading lstm model paths and predictor model paths
    lstm_model.load_state_dict(torch.load("lstm_model.pth", weights_only=True))
    predictor_model.load_state_dict(torch.load("predictor_model.pth", weights_only=False))
    #models to eval mode
    lstm_model.eval()
    predictor_model.eval()
    while True:
        menu_option = check_input.get_int_range("1. Predict a Fight\n2. Quit\n", 1, 2)
        match menu_option:
            case 1:
                fighter_a, fighter_b, a_input, b_input = get_fighter_input()
                with torch.no_grad():
                    fighter_a_input = torch.stack(a_input, dim=0)
                    fighter_b_input = torch.stack(b_input, dim=0)
                    lengths_a = torch.tensor([len(fighter_a_input)])
                    lengths_b = torch.tensor([len(fighter_b_input)])
                    fighter_a_profile = lstm_model(fighter_a_input.unsqueeze(0), lengths_a)
                    fighter_b_profile = lstm_model(fighter_b_input.unsqueeze(0), lengths_b)
                    print(f"Predicting {fighter_a} vs {fighter_b}")

                    prediction_a = predictor_model(fighter_a_profile, fighter_b_profile).squeeze()
                    prediction_b = predictor_model(fighter_b_profile, fighter_a_profile).squeeze()
                    if prediction_a.item() >= prediction_b.item():
                        print(f"{fighter_a} is projected to win with odds of {prediction_a.item() * 100:.2f}% chance of winning\nOdds of {fighter_a} winning: {prediction_a.item()* 100:.2f}%\nOdds of {fighter_b} winning: {prediction_b.item()*100:.2f}%")
                    else:
                        print(f"{fighter_b} is projected to win with odds of {prediction_b.item() * 100:.2f}% chance of winning\nOdds of {fighter_a} winning: {prediction_a.item() * 100:.2f}%\nOdds of {fighter_b} winning: {prediction_b.item()*100:.2f}%")
            case 2:
                break
                


main()