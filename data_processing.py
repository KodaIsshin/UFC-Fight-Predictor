import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import check_input

#load the CSV file
df = pd.read_csv("UFC Fighter Dataset.csv")
of = pd.read_csv("UFC Outcomes.csv")
outcomes_dict = of.groupby('Fighter A').apply(lambda x: x.values.tolist(), include_groups=False).to_dict()
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
    

def normalize(data, min_val=None, max_val=None):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


class UFCDataset(Dataset):
    def __init__(self, data_frame) -> None:
        #groups all the statistic by the fighter name in the csv
        self.data = data_frame.groupby("Fighter Name").apply(process_fighter_group, include_groups=False)
        #dictionary to link fighter statistic to name
        self.fighter_name_index = {}
        for fighter_name, (general_stats, fight_history) in self.data.items():
            fighter_data = []
            general_stats = pd.to_numeric(general_stats, errors="coerce")
            for idx, row in fight_history.iterrows():
                 round_duration = pd.Series(row['Round'])
                 #statistics for the fighter
                 fighter_stats = pd.to_numeric(row[['Weight','Sig_Strikes', 'Takedowns', 'Knockdowns', 'Sub_Attempts']], errors='coerce')
                 fight_control_time = pd.Series(time_to_seconds(row['Control_Time']), index=['Control_Time']) if row["Control_Time"] != "--" else pd.Series(time_to_seconds("0:00"), index=['Control_Time'])
                 fighter_stats = pd.concat([fighter_stats, fight_control_time])
                 #statistics for the opponent
                 opponent_stats = pd.to_numeric(row[['Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Sub_Attempts']], errors='coerce') 
                 opp_control_time = pd.Series(time_to_seconds(row['Opp_Control_Time']), index=['Opp_Control_Time']) if row["Opp_Control_Time"] != "--" else pd.Series(time_to_seconds("0:00"), index=['Opp_Control_Time'])
                 opponent_stats = pd.concat([opponent_stats, opp_control_time])

                 combined_inputs = pd.concat([general_stats,fighter_stats, opponent_stats, round_duration])
                 combined_inputs = np.array(combined_inputs)
                 combined_inputs = normalize(combined_inputs)
                 fighter_data.append(combined_inputs)
            fighter_data = np.array(fighter_data)
            self.fighter_name_index[fighter_name] = fighter_data
        print("finished reading all fighters")
        print()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return index, self.fighter_name_index[index]
        
    def fighter_by_name(self, fighter_name):
         if fighter_name in self.fighter_name_index:
              return self.__getitem__(fighter_name)
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
                    name, inputdata = dataset.fighter_by_name(fighter_input)
                    print(f"Data for {fighter_input[1]}:")
                    print(f"Name: {name}")
                    print(f"Input data:", inputdata)
                except KeyError as e:
                    print(e)
             case 2:
                  print("Goodbye.")
                  break

def find_input_sequences(fighter_name, fight_indice):
    fight_history = outcomes_dict[fighter_name]
    fighter_a_input = dataset.fighter_name_index[fighter_name][:fight_indice-1]
    fighter_b = fight_history[fight_indice-1][0]
    opponent_history = outcomes_dict[fighter_b]
    for i in opponent_history:
        if i[0] == fighter_name:
            if (dataset.fighter_name_index[fighter_b][i[2]-1][6:11] == dataset.fighter_name_index[fighter_name][fight_indice-1][11:16]).all():
                fighter_b_input = dataset.fighter_name_index[fighter_b][:i[2]-1]
                break
            else:
                fighter_b_input = [dataset.fighter_name_index[fighter_b][0]]
                break
    fighter_a_input = [torch.tensor(i, dtype=torch.float32) for i in fighter_a_input] if len(fighter_a_input) > 1 else torch.tensor(dataset.fighter_name_index[fighter_name][0], dtype=torch.float32)
    fighter_b_input = [torch.tensor(j,dtype=torch.float32) for j in fighter_b_input] if len(fighter_b_input) > 1 else torch.tensor(dataset.fighter_name_index[fighter_b][0], dtype=torch.float32)
    return fighter_name, fighter_b,fighter_a_input, fighter_b_input


def display_outcomes_by_fighter():
    while True:
         print("1. Get Fighter Outcomes\n2. Quit")
         menu_option = check_input.get_int_range("", 1, 2)
         match menu_option:
            case 1:
                fighter_name = str(input("Input Fighter Name: "))
                fighter_rows = outcomes_dict[fighter_name]
                print(fighter_rows)
                while True:
                    print(f"Find Input Sequences (2-{len(fighter_rows)}).\nPress 1 to exit")
                    menu_option = check_input.get_int_range("", 1, len(fighter_rows))
                    if menu_option == 1:
                        break
                    else:
                        find_input_sequences(fighter_name, menu_option)
            case 2:
                 break
             

               

            

        


