import torch
import pandas as pd

fight_dict = {}
train_test_fighters = open("Train_Test Fighters.txt")
train_list = train_test_fighters.readlines()

for i in train_list:
    fighter = i.strip()  # Remove any extra whitespace or newline characters
    if fighter not in fight_dict:
        fight_dict[fighter] = 0  # Initialize the count to 0 if the fighter is not in the dictionary
    fight_dict[fighter] += 1  # Increment the count for the fighter

# Now, to check which fighters appear twice or not
for fighter, count in fight_dict.items():
    if count == 2:
        print(f"{fighter} appears twice")
    elif count == 1:
        print(f"{fighter} appears once")
    else:
        print(f"{fighter} appears more than twice")
