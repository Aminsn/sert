import numpy as np
import pandas as pd

# Randomly sample 1000 numbers between 10 and 100
n_obs = np.random.randint(10, 100, 1000)

# Randomly N samples with values between 0 and 1 for each number of observations in n_obs
times = [np.random.uniform(0, 100, n) for n in n_obs]
values = [np.random.uniform(0, 1, n) for n in n_obs]
index = [np.repeat(i, n) for i, n in enumerate(n_obs)]

# Create a pandas dataframe with the data
df = pd.DataFrame(
    { "id": np.concatenate(index),
        "time": np.concatenate(times),
        "value": np.concatenate(values)
    }
)

# Sample 30% of the a unique id's
unique_ids = df["id"].unique()
sampled_ids = np.random.choice(unique_ids, int(len(unique_ids) * 0.3), replace=False)

outliers_values = np.random.uniform(2, 3, sampled_ids.shape[0])
outliers_times = np.random.uniform(0, 100, sampled_ids.shape[0])

df_outliers = pd.DataFrame(
    { "id": sampled_ids,
        "time": outliers_times,
        "value": outliers_values
    }
)

df = pd.concat([df, df_outliers]).sort_values(by=["id", "time"])

# Response variable
y = np.isin(np.sort(unique_ids), sampled_ids)
