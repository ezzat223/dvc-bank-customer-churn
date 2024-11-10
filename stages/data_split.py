from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.load_params import load_params 


def data_split(
    data_dir, 
    data_fname, 
    cat_cols,
    num_cols, 
    targ_col, 
    test_size, 
    random_state
):
    df = pd.read_csv(data_dir/data_fname)
    X, y = df[cat_cols + num_cols], df[targ_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.to_pickle(data_dir/'X_train.pkl')
    X_test.to_pickle(data_dir/'X_test.pkl')
    y_train.to_pickle(data_dir/'y_train.pkl')
    y_test.to_pickle(data_dir/'y_test.pkl')


if __name__ == '__main__':
    # library to handle command-line arguments
    args_parser = argparse.ArgumentParser()
    # --config specifies that a flag named --config is expected.
    # dest='config': Sets the destination name for this argument. In this case, the value provided to --config will be accessible as args.config.
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    
    data_dir = Path(params.base.data_dir)
    data_fname = params.base.data_fname
    cat_cols = params.base.cat_cols
    num_cols = params.base.num_cols
    targ_col = params.base.targ_col
    random_state = params.base.random_state
    test_size = params.data_split.test_size
    
    data_split(
        data_dir=data_dir, 
        data_fname=data_fname, 
        cat_cols=cat_cols,
        num_cols=num_cols, 
        targ_col=targ_col, 
        test_size=test_size, 
        random_state=random_state
    )