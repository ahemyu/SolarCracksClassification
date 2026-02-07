from pathlib import Path
from datetime import datetime
import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42
TEST_SPLIT_SEED = 42
VAL_SPLIT_SEED = 43
TEST_SIZE = 0.15
VAL_SIZE = 0.15
VAL_SIZE_WITHIN_TRAIN = VAL_SIZE / (1.0 - TEST_SIZE)


def iterative_multilabel_split(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    x = data.index.to_numpy()
    y = data[["crack", "inactive"]].to_numpy(dtype=int)
    train_idx, test_idx = next(splitter.split(x, y))
    train_part = data.iloc[train_idx].reset_index(drop=True)
    test_part = data.iloc[test_idx].reset_index(drop=True)
    return train_part, test_part

# Load labels and create reproducible splits.
np.random.seed(SEED)
t.manual_seed(SEED)
if t.cuda.is_available():
    t.cuda.manual_seed_all(SEED)

data = pd.read_csv("labels.csv", sep=";")
train_val_data, test_data = iterative_multilabel_split(
    data=data,
    test_size=TEST_SIZE,
    seed=TEST_SPLIT_SEED,
)
#split up training into training and validation
train_data, validation_data = iterative_multilabel_split(
    data=train_val_data,
    test_size=VAL_SIZE_WITHIN_TRAIN,
    seed=VAL_SPLIT_SEED,
)

train_dataset = ChallengeDataset(train_data)
validation_dataset = ChallengeDataset(validation_data)
test_dataset = ChallengeDataset(test_data)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataloader = t.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    drop_last=True
)

val_dataloader = t.utils.data.DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

# create an instance of our ResNet model
model = model.ResNet()
loss = t.nn.BCELoss() #TODO: try different ones
optimizer = t.optim.AdamW(params=model.parameters()) #TODO: choose lr

Path("logs").mkdir(parents=True, exist_ok=True)
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_log_path = Path("logs") / f"train_{run_timestamp}.log"

trainer = Trainer(
    model,
    loss,
    optimizer,
    train_dataloader,
    val_dataloader,
    cuda=True,
    early_stopping_patience=15,
    run_log_path=str(run_log_path),

) #TODO: research early stopping
# call fit on trainer
res = trainer.fit(300)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
Path("losses").mkdir(parents=True, exist_ok=True)
plt.savefig(f"losses/losses_{run_timestamp}.png")
