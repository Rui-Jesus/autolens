import numpy as np
import pandas as pd
import cv2
import os
import shutil

from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, path_metadata: str, path_dataset: str, n_data: int) -> None:
        """
        :param path_metadata: path to csv with image paths and respective classes
        :param path_dataset: path leading to dataset entrance to connect with metadata (path)
        """
        self.metadata = path_metadata
        self.path_dataset = path_dataset
        self.n_data = n_data

    def to_path(self, test_seed=1, val_seed=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.metadata)
        # Correction of the filename column
        df['filename'] = '../../BreaKHis_v1/' + df['filename']

        train: pd.DataFrame
        valid: pd.DataFrame
        test: pd.DataFrame

        train_and_valid, test = train_test_split(df, test_size=0.2, random_state=test_seed)

        train, valid = train_test_split(train_and_valid, test_size=0.25, random_state=val_seed)

        del train["Unnamed: 0"]
        del test["Unnamed: 0"]
        del valid["Unnamed: 0"] 

        return train, test, valid


    def to_pixel(self, test_seed=1, val_seed=1, batch=(0, None), target_size=(255, 255)) -> tuple[list[np.ndarray], list[np.ndarray]]:
        print("Converting path dataset to images.")
        data = self.to_path(test_seed=test_seed, val_seed=val_seed)

        data_x: list[np.ndarray] = []
        data_y: list[np.ndarray] = []

        info = ["valid", "test", "train"]
        for x_m in data:
            print("Collecting data for: ", info.pop())
            # runs 3 times for train, test and validation
            if None in batch:
                n_data = x_m.shape[0]
                init = 0
            else:
                n_data = batch[1]
                init = batch[0]

            channels = 3
            x_p = np.zeros([n_data, target_size[0], target_size[1], channels], dtype=np.uint8)
            y = np.zeros([n_data], dtype=np.uint8)
            i = init
            while i < init + n_data:
                matrix = cv2.imread(x_m["filename"].iloc[i])
                # 3 channels
                for c in range(channels):
                    x_p[i-init, :, :, c] = cv2.resize(matrix[:, :, c], target_size) #interpolation=cv2.INTER_NEAREST)

                y[i-init] = x_m[list(x_m.columns)[-1]].iloc[i]

                i += 1

            data_x.append(x_p)
            data_y.append(y)

        # data[0] for x
        # data[1] for y
        # data[i][j] for train, test and validation
        return data_x, data_y

    def to_folders(self, test_seed=1, val_seed=1) -> None:
        data = self.to_path(test_seed=test_seed, val_seed=val_seed)
        print('Recreating the structure of the dataset ...')
        parent_dir = '../tests/resources/'
        directory = 'new_breakhis'
        new_path = os.path.join(parent_dir, directory)

        try:
            shutil.rmtree(new_path)
        except FileNotFoundError:
            pass

        # Create the new directory
        os.mkdir(new_path)

        df_list = ['valid', 'test', 'train']
        df_list_2 = df_list.copy()
        for df in data:  # train, test, valid
            folder_name = df_list.pop()
            os.mkdir(os.path.join(new_path, folder_name))
            for i in df['binary_label'].unique():
                os.mkdir(os.path.join(new_path, folder_name, str(i)))

        # Move images to new directory
        for df in data:
            folder_name_2 = df_list_2.pop()
            for index, row in df.iterrows():
                src = row['filename']
                dst = os.path.join(new_path, folder_name_2, str(row['binary_label']), os.path.basename(src))
                shutil.copyfile(src, dst)
        
        return new_path


