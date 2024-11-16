import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import check_input

#load the CSV file
df = pd.read_csv("UFC Dataset.csv")

#identify and drop duplicate columns
columns_to_drop = ["Height", "Reach", "Wins", "Losses", "Age"]

def process_fighter_group(group):
    # Keep only the first instance of the general stats
    general_stats = group.iloc[0][columns_to_drop]

    # Keep all fight history, including the opponent's name
    fight_history = group[['Weight', 'Fight_ID', 'Result', 'Sig_Strikes', 'Takedowns', 'Knockdowns', 'Control_Time', 'Sub_Attempts', 'Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Control_Time', 'Opp_Sub_Attempts' , 'Round']]

    # Return both general stats and fight history as a tuple
    return general_stats, fight_history
 
def time_to_seconds(time_str):
        time_list = time_str.split(":")
        return (float(time_list[0]) * 60) + float(time_list[1])
    

def normalize(data):
     return (data - np.min(data)) / (np.max(data) - np.min(data))


class UFCDataset(Dataset):
    def __init__(self, data_frame) -> None:
        self.data = data_frame.groupby("Fighter Name").apply(process_fighter_group)
        self.inputs = []
        self.labels = []
        self.fighter_name_index = {}
        self.fighter_names = []
        index = 0
        for fighter_name, (general_stats, fight_history) in self.data.items():
            fighter_data = []
            fighter_outcomes = []
            general_stats = pd.to_numeric(general_stats, errors="coerce")
            for idx, row in fight_history.iterrows():
                 print(f"reading {fighter_name} data")
                 round_duration = pd.Series(row['Round'])
                 #statistics for the fighter
                 fighter_stats = pd.to_numeric(row[['Weight','Fight_ID','Sig_Strikes', 'Takedowns', 'Knockdowns', 'Sub_Attempts']], errors='coerce')
                 fight_control_time = pd.Series(time_to_seconds(row['Control_Time']), index=['Control_Time'])
                 fighter_stats = pd.concat([fighter_stats, fight_control_time])
                 #statistics for the opponent
                 opponent_stats = pd.to_numeric(row[['Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Sub_Attempts']], errors='coerce')
                 opp_control_time = pd.Series(time_to_seconds(row['Opp_Control_Time']), index=['Opp_Control_Time'])
                 opponent_stats = pd.concat([opponent_stats, opp_control_time])
                 combined_inputs = pd.concat([general_stats,fighter_stats, opponent_stats])
                 combined_inputs = pd.concat([combined_inputs, round_duration])
                 fighter_data.append(combined_inputs.values)
                 fighter_outcomes.append(row['Result'])

            fighter_data = normalize(np.array(fighter_data))
            self.fighter_names.append(str(fighter_name))
            self.inputs.append(np.array(fighter_data))
            self.labels.append(np.array(fighter_outcomes, dtype=np.int32))
            self.fighter_name_index[fighter_name] = index
            index += 1
        print("finished reading all fighters")
        print()
        print()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.fighter_names[index], torch.tensor(self.inputs[index], dtype=torch.float32), torch.tensor(self.labels[index], dtype=torch.float32)
        
    def fighter_by_name(self, fighter_name):
         if fighter_name in self.fighter_name_index:
              index = self.fighter_name_index[fighter_name]
              return self.__getitem__(index)
         else:
              raise KeyError(f"Fighter {fighter_name} not found in dataset")

dataset = UFCDataset(df)

def display_data_by_fighter():
    while True:
        print("1. Read Tensor of Fighter\n2. Quit")
        menu_option = check_input.get_int_range("", 1, 2)
        match menu_option:
             case 1:   
                try:
                    fighter_input = str(input("Input Fighter Name: "))
                    name, inputdata, labels = dataset.fighter_by_name(fighter_input)
                    print(f"Data for {fighter_input}:")
                    print(f"Name: {name}")
                    print(f"Input data:", inputdata)
                    print(f"Label:", labels)
                except KeyError as e:
                    print(e)
             case 2:
                  print("Goodbye.")
                  break

display_data_by_fighter()

               

            

        


