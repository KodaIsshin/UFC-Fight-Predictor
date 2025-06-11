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

                    # Get raw logits
                    logit_ab = predictor_model(fighter_a_profile, fighter_b_profile).squeeze()
                    logit_ba = predictor_model(fighter_b_profile, fighter_a_profile).squeeze()

                    # Convert to probabilities
                    prob_ab = torch.sigmoid(logit_ab).item()
                    prob_ba = torch.sigmoid(logit_ba).item()

                    if prob_ab >= prob_ba:
                        winner = fighter_a
                        print(f"{winner} is projected to win with {prob_ab:.2%} confidence.")
                    else:
                        winner = fighter_b
                        print(f"{winner} is projected to win with {prob_ba:.2%} confidence.")

                    print(f"Odds:\n  {fighter_a}: {prob_ab:.2%}\n  {fighter_b}: {prob_ba:.2%}")
            case 2:
                break
                


main()