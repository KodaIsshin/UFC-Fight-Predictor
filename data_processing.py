import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import check_input

#load the CSV file
df = pd.read_csv("UFC Fighter Dataset.csv")
of = pd.read_csv("UFC Outcomes.csv")
outcomes_dict = of.groupby('Fighter Name').apply(lambda x: x.values.tolist(), include_groups=False).to_dict()
#identify and drop duplicate columns
columns_to_drop = ["Height", "Reach", "Wins", "Losses", "Age"]

FEATURE_RANGES = {
    'Weight': (125, 265),
    'Sig_Strikes': (0, 400),
    'Takedowns': (0, 60),
    'Knockdowns': (0, 10),
    'Sub_Attempts': (0, 20),
    'Control_Time': (0, 1500),
    'Opp_Sig_Strikes': (0, 400),
    'Opp_Takedowns': (0, 60),
    'Opp_Knockdowns': (0, 10),
    'Opp_Sub_Attempts': (0, 20),
    'Opp_Control_Time': (0, 1500),
    'Round': (0, 6),
    'Height': (50, 100),  # Example: 150 cm to 220 cm
    'Reach': (50, 100),   # Example: 150 cm to 220 cm
    'Wins': (0, 50),      # Example: Number of wins
    'Losses': (0, 50),    # Example: Number of losses
    'Age': (18, 50), 
    'Fight_ID': (0,7)      
}
def process_fighter_group(group):
    # Keep only the first instance of the general stats
    general_stats = group.iloc[0][columns_to_drop]

    # Keep all fight history, including the opponent's name
    fight_history = group[['Weight', 'Fight_ID', 'Result', 'Sig_Strikes', 'Takedowns', 'Knockdowns', 'Control_Time', 'Sub_Attempts', 'Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Control_Time', 'Opp_Sub_Attempts','Round']]

    # Return both general stats and fight history as a tuple
    return general_stats, fight_history
 

def time_to_seconds(time_str):
        time_list = time_str.split(":")
        return (float(time_list[0]) * 60) + float(time_list[1])
    
def normalize(data, min_val, max_val):
    """
    Normalizes data to the range [0, 1] using fixed min and max values.
    Args:
        data: A single value, Pandas Series, or NumPy array to normalize.
        min_val: Fixed minimum value for normalization.
        max_val: Fixed maximum value for normalization.
    Returns:
        Normalized data or a constant value if min_val == max_val.
    """ # Assign constant value if range is 0
    return (data - min_val) / (max_val - min_val)  # Add epsilon to avoid division by zero


class UFCDataset(Dataset):
    def __init__(self, data_frame) -> None:
        # Group all statistics by fighter name
        self.data = data_frame.groupby("Fighter Name").apply(process_fighter_group, include_groups=False)
        # Dictionary to link fighter statistics to name
        self.fighter_name_index = {}
        for fighter_name, (general_stats, fight_history) in self.data.items():
            fighter_data = []
            
            # Normalize general stats
            general_stats = pd.to_numeric(general_stats, errors="coerce")
            general_stats = self.normalize_features(general_stats, ['Height', 'Reach', 'Wins', 'Losses', 'Age'])
            
            # Process each fight
            for idx, row in fight_history.iterrows():
                # Round duration
                round_duration = normalize(row['Round'], *FEATURE_RANGES['Round'])
                
                # Normalize fighter stats
                fighter_stats = self.normalize_features(
                    row[['Weight', 'Sig_Strikes', 'Takedowns', 'Knockdowns', 'Sub_Attempts']],
                    ['Weight', 'Sig_Strikes', 'Takedowns', 'Knockdowns', 'Sub_Attempts']
                )
                fight_control_time = normalize(time_to_seconds(row['Control_Time']), *FEATURE_RANGES['Control_Time']) if row['Control_Time'] != "--" else normalize(0, *FEATURE_RANGES['Control_Time'])

                # Normalize opponent stats
                opponent_stats = self.normalize_features(
                    row[['Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Sub_Attempts']],
                    ['Opp_Sig_Strikes', 'Opp_Takedowns', 'Opp_Knockdowns', 'Opp_Sub_Attempts']
                )
                opp_control_time = normalize(time_to_seconds(row['Opp_Control_Time']), *FEATURE_RANGES['Opp_Control_Time']) if row['Opp_Control_Time'] != "--" else normalize(0, *FEATURE_RANGES['Opp_Control_Time'])

                # Combine all inputs
                combined_inputs = np.concatenate([
                    general_stats, fighter_stats, [fight_control_time],
                    opponent_stats, [opp_control_time], [round_duration]
                ])
                fighter_data.append(combined_inputs)
            
            fighter_data = np.array(fighter_data)
            self.fighter_name_index[fighter_name] = fighter_data
        
        print("Finished reading all fighters.\n")

    def normalize_features(self, data, feature_names):
        """
        Normalize multiple features using predefined ranges.
        Args:
            data: Pandas Series or row containing the features to normalize.
            feature_names: List of feature names corresponding to the data.
        Returns:
            Numpy array of normalized features.
        """
        normalized_data = []
        for feature in feature_names:
            min_val, max_val = FEATURE_RANGES[feature]
            normalized_data.append(normalize(data[feature], min_val, max_val))
        return np.array(normalized_data)

    def __len__(self):
        return len(self.fighter_name_index)

    def __getitem__(self, index):
        return index, self.fighter_name_index[index]
        
    def fighter_by_name(self, fighter_name):
        if fighter_name in self.fighter_name_index:
            return self.__getitem__(fighter_name)
        else:
            raise KeyError(f"Fighter {fighter_name} not found in dataset")
        
dataset = UFCDataset(df)        

# def display_data_by_fighter():
#     while True:
#         print("1. Read Tensor of Fighter\n2. Quit")
#         menu_option = check_input.get_int_range("", 1, 2)
#         match menu_option:
#              case 1:   
#                 try:
#                     fighter_input = str(input("Input Fighter Name: "))
#                     name, inputdata = dataset.fighter_by_name(fighter_input)
#                     print(f"Data for {fighter_input[1]}:")
#                     print(f"Name: {name}")
#                     print(f"Input data:", inputdata)
#                 except KeyError as e:
#                     print(e)
#              case 2:
#                   print("Goodbye.")
#                   break

# def display_outcomes_by_fighter():
#     while True:
#          print("1. Get Fighter Outcomes\n2. Quit")
#          menu_option = check_input.get_int_range("", 1, 2)
#          match menu_option:
#             case 1:
#                 fighter_name = str(input("Input Fighter Name: "))
#                 fighter_rows = outcomes_dict[fighter_name]
#                 print(fighter_rows)
#                 while True:
#                     print(f"Find Input Sequences (2-{len(fighter_rows)}).\nPress 1 to exit")
#                     menu_option = check_input.get_int_range("", 1, len(fighter_rows))
#                     if menu_option == 1:
#                         break
#                     else:
#                         fighter_name, fighter_b, fighter_a_input, fighter_b_input = find_input_sequences(fighter_name, menu_option)
#                         print(fighter_a_input)

#             case 2:
#                  break
             



