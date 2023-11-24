from collections import defaultdict
import pickle
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

TACKLER_PAD_VALUE = -2.
PAD_VALUE = -1.
N_FEATS_PER_PLAYER_PER_TIMESTEP = 6

GEOMETRIC_KEYS = ["offense_geometric", "defense_geometric"]
RAW_KEYS = ["offense_raw", "defense_raw"]
SPECIAL_RAW_KEYS = ["ball_carrier_raw", "tacklers_raw"]
EVENT_KEYS = ["event_timeseries"]
TIME_SERIES_KEYS = GEOMETRIC_KEYS + RAW_KEYS + EVENT_KEYS

TARGET_KEY = "yards_after_contact"
TREATMENT_KEY = "tackle_successful"
STATIC_KEYS = ["play_features_encoded"]  # future: play features and on-field player info


def create_batchdict(batch):
    batchdict = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            if k in TIME_SERIES_KEYS + STATIC_KEYS + RAW_KEYS + SPECIAL_RAW_KEYS:
                batchdict[k].append(torch.from_numpy(v.to_numpy()))
            elif k in [TARGET_KEY, TREATMENT_KEY]:
                batchdict[k].append(v)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    warnings.warn(
                        f"Key `{k}` not found -- no logic provided for handling this data. " +
                        "You may safely ignore this if this key does not contain values passed directly to the model. " +
                        f"Type: {type(v)}"
                    )
    return batchdict


def collate_padded_play_data(batch):
    """
        Deprecated. Used for earlier models.
    """
    batchdict = create_batchdict(batch)
    X_padded = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in TIME_SERIES_KEYS], dim=2)
    return {
        "time_series_features": X_padded,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

def collate_padded_play_data_geometric_only(batch):
    batchdict = create_batchdict(batch)
    X_padded = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS], dim=2)
    return {
        "time_series_features": X_padded,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

def pad_tacklers(tackler_list):
    n_tacklers = torch.tensor([item.size(1) / N_FEATS_PER_PLAYER_PER_TIMESTEP for item in tackler_list], dtype=torch.long)  # Use during forward pass to index correctly in the tackler dimension.
    tackler_reshaped = [tackler_data.view(tackler_data.size(0), -1, N_FEATS_PER_PLAYER_PER_TIMESTEP) for tackler_data in tackler_list]  # (batch_size, t, n_tacklers, n_feats) -- swap n_tacklers, t
    padded_tacklers = []
    for single_play_data in tackler_reshaped:
        n_tacklers_below_max = max(n_tacklers) - single_play_data.size(1)
        if n_tacklers_below_max > 0:  # then padding is required
            pad_shape = [single_play_data.size(0), n_tacklers_below_max, single_play_data.size(2)]  # each single play has shape t, n_tacklers, n_feats
            padded_single_play_data = torch.cat([single_play_data, TACKLER_PAD_VALUE * torch.ones(*pad_shape)], dim=1)
            padded_tacklers.append(padded_single_play_data)
        else:
            padded_tacklers.append(single_play_data)
    X_tacklers = pad_sequence(padded_tacklers, batch_first=True, padding_value=PAD_VALUE)  # finally pad along time axis.
    return X_tacklers, n_tacklers

def collate_padded_play_data_with_carrier_tackler_info(batch):
    batchdict = create_batchdict(batch)
    X_geometric = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS], dim=2)
    X_ball_carrier = pad_sequence(batchdict["ball_carrier_raw"], batch_first=True, padding_value=PAD_VALUE)
    X_tacklers, n_tacklers = pad_tacklers(batchdict["tacklers_raw"])
    return {
        "time_series_features": (X_geometric, X_ball_carrier, X_tacklers, n_tacklers),
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }
    # X_geometric, X_ball_carrier, X_tacklers, n_tacklers = batch

def collate_padded_play_data_with_carrier_tackler_and_raw_info(batch):
    batchdict = create_batchdict(batch)
    X_geometric = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS + RAW_KEYS], dim=2)
    X_ball_carrier = pad_sequence(batchdict["ball_carrier_raw"], batch_first=True, padding_value=PAD_VALUE)
    X_tacklers, n_tacklers = pad_tacklers(batchdict["tacklers_raw"])
    return {
        "time_series_features": (X_geometric, X_ball_carrier, X_tacklers, n_tacklers),
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

def collate_padded_play_data_with_context(batch):
    batchdict = create_batchdict(batch)
    X_geometric = torch.cat([pad_sequence(batchdict[k], batch_first=True, padding_value=PAD_VALUE) for k in GEOMETRIC_KEYS + RAW_KEYS], dim=2)
    X_ball_carrier = pad_sequence(batchdict["ball_carrier_raw"], batch_first=True, padding_value=PAD_VALUE)
    X_tacklers, n_tacklers = pad_tacklers(batchdict["tacklers_raw"])
    #X_padded_static = # TODO: tile in time, then pad_sequence
    return {
        "time_series_features": (X_geometric, X_ball_carrier, X_tacklers, n_tacklers),
        "features": X_padded_static,
        "target": torch.tensor(batchdict[TARGET_KEY], dtype=torch.float),
        "treatment": torch.tensor(batchdict[TREATMENT_KEY], dtype=torch.float),
    }

from dragonnet.models import TransformerRepresentor, MultiLevelTransformerRepresentor
COLLATE_FN_DICT = {
    TransformerRepresentor.__name__: collate_padded_play_data_geometric_only,
    MultiLevelTransformerRepresentor.__name__: collate_padded_play_data_with_carrier_tackler_and_raw_info,  # collate_padded_play_data_with_carrier_tackler_info,
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
