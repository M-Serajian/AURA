import cudf

class Config:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.debug = False
        self.test = False
        

    def set_debug(self, debug):
        self.debug = debug

    def set_test(self, test):
        self.test = test


    # def load_dataframe(self, file_path):
    #     # Load a CSV file using cuDF's read_csv function
    #     self.source_dataframe = cudf.read_csv(file_path,usecols=["K-mer"])
    #     self.source_dataframe = self.source_dataframe.sort_values('K-mer')
    #     self.source_dataframe.reset_index(drop=True, inplace=True)

    # def load_parque_files_list(self,parquet_path):
    #     with open('example.txt', 'r') as file:
    #     # Read all lines into a list and strip newline characters
    #         self.lines = [line.strip() for line in file.readlines() if line.strip()]
