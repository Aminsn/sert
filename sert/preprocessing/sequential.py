import numpy as np
import pandas as pd


class SeqDataPreparer:
    """
    A class to preprocess sequential input data in long format (time, variable name, value) and transform it into the appropriate format for the SERT and the SERNN models.


    Example input data:
    index | time  | variable | value
    --------------------------------
    0     | 1.23  |HR        | 80
    1     | 3.6   |RR        | 20

    Attributes:
    -----------
    token_capacity : int
        Maximum length for padding sequences.
    index : str
        Name of the column representing the primary index.
    times : str
        Name of the column representing time-related data.
    names : str
        Name of the column for variable names.
    values : str
        Name of the column for variable values.
    name_to_int : dict
        Dictionary mapping variable names to indices.

    Methods:
    --------
    fit_transform(input_data: pd.DataFrame, index: str, times: str, names: str, values: str) -> list:
        Define column mappings and transform input data.
    transform(input_data: pd.DataFrame) -> list:
        Transform input data based on previously defined column mappings.
    _preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
        Handle NaN values, create a mask for numeric values, and manipulate column names/values.
    _pad_input(x: list) -> np.array:
        Pad data to the specified token_capacity.
    _transform(data: pd.DataFrame) -> list:
        Reformat data after preprocessing to match desired output format.
    """

    def __init__(self, token_capacity):
        """
        Initialize SeqDataTransformer with a specified token_capacity.

        Parameters:
        -----------
        token_capacity : int
            Maximum length to which data sequences will be padded.
        """
        self.token_capacity = token_capacity
        self.index = None
        self.times = None
        self.names = None
        self.values = None
        self.name_to_int = None

    def _pad_input(self, x):
        """
        Pad input data to the specified token_capacity.

        Parameters:
        -----------
        x : list
            Input list to be padded.

        Returns:
        --------
        np.array
            Padded numpy array.
        """
        pad_length = self.token_capacity - len(x)

        # Check if pad_length is negative
        if pad_length < 0:
            raise ValueError(
                "Input length exceeds token_capacity. Increase the token_capacity or remove some variables.")

        return np.pad(x, (0, pad_length), 'constant')

    def _preprocess(self, input_data):
        """
        Preprocesses input data by handling NaN values, creating a mask for numeric values, and manipulating column names/values.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Data to preprocess.

        Returns:
        --------
        pd.DataFrame
            Preprocessed data.
        """
        names = self.names
        values = self.values

        data = input_data.copy()
        data.dropna(inplace=True)
        # Create a mask for categorical variables
        mask = data[values].apply(
            lambda x: isinstance(x, (int, float, np.number)))
        data['mask'] = mask
        # Add the values of the categorical variables to their names to be encoded by the Embedding encoder and replace the values with 0 in the value column
        data[names] = np.where(
            mask, data[names], data[names] + "_" + data[values].astype(str))
        data[values] = np.where(mask, data[values], 0)
        data[values] = data[values].astype(float)
        return data

    def _transform(self, data):
        """
        Reformats preprocessed data to ensure it meets desired output format.

        Parameters:
        -----------
        data : pd.DataFrame
            Data after preprocessing.

        Returns:
        --------
        list
            List of numpy arrays for times_input, values_input, var_id_input, and categorical_mask_input.
        """
        index = self.index
        times = self.times
        names = self.names
        values = self.values
        name_to_int = self.name_to_int

        data['var_id'] = data[names].map(name_to_int)
        data.dropna(subset=['var_id'], inplace=True)
        data = data[[index, times, 'var_id', values, 'mask']]
        data = data.groupby(index).agg(list).reset_index()
        for col in [times, values, 'var_id', 'mask']:
            data[col] = data[col].apply(lambda x: self._pad_input(np.array(x)))

        times_input = np.array(data[times].values.tolist())
        values_input = np.array(data[values].values.tolist())
        var_id_input = np.array(data['var_id'].values.tolist())
        mask_input = np.array(data['mask'].values.tolist())

        return [times_input, values_input, var_id_input, mask_input]

    def fit_transform(self, input_data, index, times, names, values):
        """
        Defines column mappings, preprocesses the input data, and transforms it to the desired format.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Data to preprocess and transform.
        index : str
            Column name representing the primary index.
        times : str
            Column name representing time-related data.
        names : str
            Column name for variable names.
        values : str
            Column name for variable values.

        Returns:
        --------
        list
            List of numpy arrays for times_input, values_input, var_id_input, and categorical_mask_input.
        """
        # Check if input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("The input_data should be a Pandas DataFrame.")

        self.index = index
        self.times = times
        self.names = names
        self.values = values

        data = self._preprocess(input_data)
        var_names = sorted(list(set(data[names])))
        self.name_to_int = {var: i + 1 for i, var in enumerate(var_names)}
        return self._transform(data)

    def transform(self, input_data):
        """
        Transforms the input data based on previously defined column mappings.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Data to transform.

        Returns:
        --------
        list
            List of numpy arrays for times_input, values_input, var_id_input, and categorical_mask_input.
        """
        data = self._preprocess(input_data)
        return self._transform(data)
