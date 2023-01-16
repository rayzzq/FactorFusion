import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def create_fake_data():
    np.random.seed(42)
    num_samples = 10000
    df_X = pd.DataFrame(np.random.randn(num_samples, 20), columns = [f"X_{i}" for i in range(20)])
    df_Y_cls = pd.DataFrame(np.random.randint(0, 2, size=(num_samples, 4)), columns = [f"Y_cls_{i}" for i in range(4)])
    df_Y_reg = pd.DataFrame(np.random.randn(num_samples, 4), columns = [f"Y_reg_{i}" for i in range(4)])
    df = pd.concat([df_X, df_Y_cls, df_Y_reg], axis=1)
    
    Xcols = [f"X_{i}" for i in range(20)]
    regYcols = [f"Y_reg_{i}" for i in range(4)]
    clsYcols = [f"Y_cls_{i}" for i in range(4)]
    
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    return train_df, test_df, Xcols, regYcols, clsYcols

if __name__ == "__main__":
    df = create_fake_data()
    print(df.head())