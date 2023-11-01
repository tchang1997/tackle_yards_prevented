from collections import defaultdict
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_VALUE = -1.
TIME_SERIES_KEYS = ["offense_geometric", "offense_raw", "defense_geometric", "defense_raw", "ball_carrier_raw", "event_timeseries"]
TARGET_KEY = "yards_after_contact"
TREATMENT_KEY = "tackle_successful"
STATIC_KEYS = []  # future: play features and on-field player info

def collate_padded_play_data(batch):
    batchdict = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            if k in TIME_SERIES_KEYS + STATIC_KEYS:
                batchdict[k].append(torch.from_numpy(v.to_numpy()))
            elif k in [TARGET_KEY, TREATMENT_KEY]:
                batchdict[k].append(v)
    X_padded = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in TIME_SERIES_KEYS], dim=2)
    X_padded_static = torch.empty((len(batch), 0))
    return {
        "time_series_features": X_padded,
        "features": X_padded_static,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }


class PlayByPlayDataset(Dataset):
    def __init__(self, path, combine_offense_defense=False):
        self.path = path
        with open(path, "rb") as f:
            self.data = pickle.load(f)

        self.combine_offense_defense = combine_offense_defense

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
