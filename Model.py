import ntpath
import pandas as pd
import os

class Model:

    def __init__(self, input_dir, ext):
        self.input_dir = input_dir
        self.ext = ext

    def check_ext(self, ext):
        if ext != ".csv":
            raise ValueError("Wrong extension, please choose .csv files")
        return ext

    def load_data(self, path):
        return pd.read_csv(path)

    def set_files_in_directory(self):

        files = []
        for file in os.listdir(self.input_dir):
            if file.endswith(self.ext):
                name = self.input_dir + "/" + file
                files.append(name)

        return files

    def get_sets(self, set_type, files):

        df = None
        for element in files:
            if set_type in ntpath.abspath(element):
                df = self.load_data(element)
        return df

    def set_dataframes(self, files):

        df = {}
        types = ["train", "test"]

        if len(files) == 1:
            df.update({"data": self.load_data(files[0])})
            return df

        elif len(files) > 1:
            for set_type in types:
                df.update({set_type: self.get_sets(set_type, files)})
        else:
            return None

        return df
