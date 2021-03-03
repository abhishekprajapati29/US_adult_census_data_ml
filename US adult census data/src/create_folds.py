import pandas as pd
from sklearn import model_selection
import os
from . import config


INPUT = config.INPUT


if __name__ == '__main__':

    df = pd.read_csv(os.path.join(INPUT, 'adult.csv'))
    df['kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=df, y=df.income.values)):
        df.loc[valid_idx, 'kfold'] = fold
    
    df.to_csv(os.path.join(INPUT, 'adult_folds.csv'), index=False)