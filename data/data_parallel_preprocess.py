import numpy as np


def split_data(
    x_train,
    y_train,
    mp_size,
    dp_size,
    rank,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        please split the data uniformly across data parallel groups and
        do not shuffle the index as we will shuffle them later
    """

    """TODO: Your code here"""

    # Try to get the correct start_idx and end_idx from dp_size, mp_size and rank and return
    # the corresponding data

    x_train_split = np.array_split(x_train, dp_size, axis=0)
    y_train_split = np.array_split(y_train, dp_size, axis=0)

    x_train_idx = rank % dp_size
    y_train_idx = rank % dp_size

    split_x_ret = x_train_split[x_train_idx]
    split_y_ret = y_train_split[y_train_idx]

    print('SPLIT_DATA DEBUGGING START')

    print('X_TRAIN')
    print(x_train)
    print('Y_TRAIN')
    print(y_train)
    print('MP_SIZE')
    print(mp_size)
    print('DP_SIZE')
    print(dp_size)
    print('RANK')
    print(rank)

    print('X_TRAIN_SPLIT')
    print(x_train_split)
    print('Y_TRAIN_SPLIT')
    print(y_train_split)

    #print('SPLIT_X_RET')
    #print(split_x_ret)
    #print('SPLIT_Y_RET')
    #print(split_y_ret)

    print('SPLIT_DATA DEBUGGING END')

    #split_x_train = x_train_split[x_train_idx]
    #split_y_train = y_train_split[y_train_idx]


    return split_x_ret, split_y_ret
    #return x_train_split, y_train_split
    #return x_train, y_train
