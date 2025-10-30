"""Split dataset into training, validation and test data."""

import pandas as pd
import numpy as np


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=23)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]

    print(
        '\nSize of splitted data:' 
        '\nTraining data: ' + str(indices_or_sections[0]) +
        '\nValidation data: ' + str(indices_or_sections[1] - indices_or_sections[0]) +
        '\nTest data: ' + str(indices_or_sections[1] - indices_or_sections[0]) + '\n'
    )
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return [train_ds, val_ds, test_ds]


# define data folder and pickle file
data_folder = 'data_EU'
whole_data_file = 'EU_south_scandinavia_all_glaciers.pkl'
train_data_name = 'EU_south_scandinavia_train_data.pkl'
val_data_name = 'EU_south_scandinavia_val_data.pkl'
test_data_name = 'EU_south_scandinavia_test_data.pkl'

whole_data_path = data_folder + '/' + whole_data_file

# load whole dataframe
whole_data = pd.read_pickle(whole_data_path)

# split into training, validation and test data
split_data = get_dataset_partitions_pd(whole_data)

# save dataframes
for index, subset_name in enumerate([train_data_name, val_data_name, test_data_name]):
    data_path = data_folder + '/' + subset_name
    split_data[index].to_pickle(data_path)
    print('Dataframe saved to \'' + data_path + '\'.')
