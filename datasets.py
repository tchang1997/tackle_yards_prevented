from collections import defaultdict
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_VALUE = -1.
GEOMETRIC_KEYS = ["offense_geometric", "defense_geometric"]
RAW_KEYS = ["offense_raw", "defense_raw"]
SPECIAL_RAW_KEYS = ["ball_carrier_raw", "tacklers_raw"]
EVENT_KEYS = ["event_timeseries"]
TIME_SERIES_KEYS = GEOMETRIC_KEYS + RAW_KEYS + EVENT_KEYS

TARGET_KEY = "yards_after_contact"
TREATMENT_KEY = "tackle_successful"
STATIC_KEYS = []  # future: play features and on-field player info


def create_batchdict(batch):
    batchdict = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            if k in TIME_SERIES_KEYS + STATIC_KEYS:
                batchdict[k].append(torch.from_numpy(v.to_numpy()))
            elif k in [TARGET_KEY, TREATMENT_KEY]:
                batchdict[k].append(v)
    return batchdict


def collate_padded_play_data(batch):
    """
        Deprecated. Used for earlier models.
    """
    batchdict = create_batchdict(batch)
    X_padded = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in TIME_SERIES_KEYS], dim=2)
    X_padded_static = torch.empty((len(batch), 0))
    return {
        "time_series_features": X_padded,
        "features": X_padded_static,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

def collate_padded_play_data_geometric_only(batch):
    batchdict = create_batchdict(batch)
    X_padded = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS], dim=2) # not even event data
    X_padded_static = torch.empty((len(batch), 0))
    return {
        "time_series_features": X_padded,
        "features": X_padded_static,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

def collate_padded_play_data_with_carrier_tackler_info(batch):
    batchdict = create_batchdict(batch)
    X_geometric = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS], dim=2)
    X_ball_carrier = pad_sequence(batchdict["ball_carrier_raw"], batch_first=True, padding_value=PAD_VALUE)
    X_tacklers = pad_sequence(batchdict["tacklers_raw"], batch_first=True, padding_value=PAD_VALUE)
    n_tacklers = torch.tensor([len(item) for item in batchdict["tacklers_raw"]], dtype=torch.long)

    X_padded_static = torch.empty((len(batch), 0))
    return {
        "time_series_features": (X_geometric, X_ball_carrier, X_tacklers, n_tacklers),
        "features": X_padded_static,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }
    # X_geometric, X_ball_carrier, X_tacklers, n_tacklers = batch

from dragonnet.models import TransformerRepresentor, TransformerRepresentorWithPlayContext
COLLATE_FN_DICT = {
    TransformerRepresentor.__name__: collate_padded_play_data_geometric_only,
    TransformerRepresentorWithPlayContext.__name__: collate_padded_play_data_with_carrier_tackler_info,
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
