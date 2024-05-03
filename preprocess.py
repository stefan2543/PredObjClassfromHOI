import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def read_npz_file(file_path):
    try:
        npzfile = np.load(file_path, allow_pickle=True)
        concatenated_values = []
        total_frames = npzfile['body'].item()['params'][next(iter(npzfile['body'].item()['params']))].shape[0]
        obj_name = npzfile['obj_name'] if 'obj_name' in npzfile else None
        keys_to_concatenate = ['body', 'lhand', 'rhand']
        subkey_to_print = 'params'
        body_params = []
        lhand_params = []
        rhand_params = []
        for key in keys_to_concatenate:
            if key in npzfile.keys() and isinstance(npzfile[key].item(), dict) and subkey_to_print in npzfile[key].item().keys():
                if isinstance(npzfile[key].item()[subkey_to_print], dict):
                    for sub_subkey_to_print in npzfile[key].item()[subkey_to_print].keys():
                        if key == 'body':
                            n_params = body_params
                        elif key == 'lhand':
                             n_params = lhand_params
                        else:
                            n_params = rhand_params
                        n_params.append(npzfile[key].item()[subkey_to_print][sub_subkey_to_print])   
        b_max_y = max(arr.shape[1] for arr in body_params)
        l_max_y = max(arr.shape[1] for arr in lhand_params)
        r_max_y = max(arr.shape[1] for arr in rhand_params)
        b = len(body_params)
        l = len(lhand_params)
        r = len(rhand_params)
        for i in range(total_frames):
            transposed_b = np.zeros((b, b_max_y))
            for j, entry in enumerate(body_params):
                transposed_b[j, :entry.shape[1]] = entry[i]
            transposed_l = np.zeros((l, l_max_y))
            for j, entry in enumerate(lhand_params):
                transposed_l[j, :entry.shape[1]] = entry[i]
            transposed_r = np.zeros((r, r_max_y))
            for j, entry in enumerate(rhand_params):
                transposed_r[j, :entry.shape[1]] = entry[i]
            concatenated_values.append(np.concatenate((transposed_b.flatten(), transposed_l.flatten(), transposed_r.flatten())))
        return concatenated_values, np.repeat(obj_name, total_frames)
    except Exception as e:
        print("Error reading file:", file_path)
        print(e)
        return None, None

def process_directory(directory):
    all_data = []
    all_labels = []
    counter = 0
    for file in os.listdir(directory):
        #if counter <= 2:
        if file.endswith(".npz"):
            file_path = os.path.join(directory, file)
            data, label = read_npz_file(file_path)
            if data is not None and label is not None:
                all_data.append(data)
                all_labels.append(label)
                print(f"Data and label processed successfully for {file}")
            else:
                print(f"No data or label to process for {file}")
            #counter += 1
    all_data_f = np.concatenate(all_data)
    all_labels_f = np.concatenate(all_labels)
    return all_data_f, all_labels_f


# Directory containing the dataset
directory = "/u/stefan/body/unzip/grab/s1/"
all_data, all_labels = process_directory(directory)
all_data = np.array([np.array(d) for d in all_data if d is not None])
# Convert labels to numeric values
label_encoder = LabelEncoder()
all_labels_encoded = label_encoder.fit_transform(all_labels)

with open('s1_data.pkl', 'wb') as f:
    pickle.dump((all_data, all_labels_encoded, label_encoder.classes_), f)

print("Data preprocessing complete and saved.")