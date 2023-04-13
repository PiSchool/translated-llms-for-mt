from torch.utils.data import Dataset
import pandas as pd

class TestPEFTDataset(Dataset):
    def __init__(self, test_path:str, prefix:str):
        self.test_data = pd.read_csv(test_path)
        self.prefix = prefix

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        text = self.prefix+self.test_data["source"].iloc[idx]
        return text