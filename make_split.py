import json
import os
import pickle

from sklearn.model_selection import KFold

available_files_path = 'workdir/mha2nnunet_settings/Task2202_picai_baseline.json'

with open(available_files_path, 'r') as file:
    jfile = json.load(file)

available_files = jfile['archive']

files = []
for subject in available_files:
    files.append(subject['annotation_path'])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Generate the folds
json_list = []
split_path = './src/picai_baseline/splits/picai_human'
os.makedirs(split_path, exist_ok=True)
for fold, (train_idx, test_idx) in enumerate(kf.split(files)):
    json_train_dict = {'subject_list': [files[x] for x in train_idx]}
    json_test_dict = {'subject_list': [files[x] for x in test_idx]}
    with open(split_path + f'/ds-config-train-fold-{fold}.json', 'w') as file:
        # Write the dictionary to file as JSON
        json.dump(json_train_dict, file, indent=4)
    with open(split_path + f'/ds-config-valid-fold-{fold}.json', 'w') as file:
        # Write the dictionary to file as JSON
        json.dump(json_test_dict, file, indent=4)
    json_list.append({'train':[files[x].replace('.nii.gz', '') for x in train_idx], 'val': [files[x].replace('.nii.gz', '') for x in test_idx]})

with open(split_path + f'/splits.p', 'wb') as file:
    pickle.dump(json_list, file)