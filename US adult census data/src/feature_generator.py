import itertools



def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))

    for c1, c2 in combi:
        df.loc[:, f'{c1}_{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)
    
    return df


