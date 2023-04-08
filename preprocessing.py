import pandas as pd
import math
import random

def balance_dataset(X, y, sample_quantity='undersample', shuffle=True):
    # get the unique labels and their respective counts
    unique_labels = list(y.unique())
    label_counts = [(y == l).sum() for l in unique_labels]
    label_count_dict = dict(zip(unique_labels, label_counts))

    # determing number of samples for each label if sampling method is passed
    if sample_quantity == 'undersample':
        sample_quantity = min(label_counts)
    elif sample_quantity == 'oversample':
        sample_quantity = max(label_counts)

    # create blank dataframes that will be used to store balanced data
    final_X_data = pd.DataFrame({})
    final_y_data = pd.DataFrame({})

    # begin sampling for each label
    for label, count in label_count_dict.items():
        # get current subset of X data
        X_current = X[y == label]
        X_current.reset_index(inplace=True, drop=True)

        # concatenate repeats of the data if necessary for oversampling
        if count <= sample_quantity:
            n_repeats = math.floor(sample_quantity / count)
            for i in range(n_repeats):
                final_X_data = pd.concat([final_X_data, X_current])

        # randomly sample the data (without repeats of rows) until the sample quantity is met
        n_samples = sample_quantity % count
        if n_samples > 0:
            sample_indices = random.sample(range(count), n_samples)
            sampled_X_data = X_current.iloc[sample_indices]
            final_X_data = pd.concat([final_X_data, sampled_X_data])

        # concatenate the current label to the balanced label output
        final_y_data = pd.concat([final_y_data, pd.DataFrame([label] * sample_quantity)])

    # re-index the balanced data and rename variables
    X_balanced = final_X_data.reset_index(drop=True)
    y_balanced = final_y_data.reset_index(drop=True)

    # shuffle the balanced data if required
    if shuffle:
        index_list = [i for i in range(X_balanced.shape[0])]
        random.shuffle(index_list)
        X_balanced = X_balanced.iloc[index_list]
        y_balanced = y_balanced.iloc[index_list]

    return X_balanced, y_balanced









