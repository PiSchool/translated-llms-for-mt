from torch.utils.data import Dataset

class PEFTDataset(Dataset):
    def __init__(self, input_id, attention, labels):
        self.attention = attention
        self.input_id = input_id
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        attention = self.attention[idx]
        label = self.labels[idx]
        input_id = self.input_id[idx]
        sample = {"attention_mask": attention,
                  "input_ids": input_id, "labels": label}
        return sample