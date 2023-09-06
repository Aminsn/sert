import numpy as np
import pandas as pd


class DataPreparer:
    """
    A class to preprocess static input data in long format (variable name, value) and transform it into the appropriate format for the SERT and the SERNN models.

    Example input data:
    index | variable | value
    -------------------------
    0     | HR       | 80
    1     | RR       | 20

    Attributes:
    -----------
    token_capacity : int
        The maximum length to which data sequences will be padded.
    name_to_int : dict
        Dictionary mapping variable names to integers (to be used by the Embedding encoder).

    Methods:
    --------
    fit_transform(input_data: pd.DataFrame) -> list:
        Process the input_data and return the transformed data.
    transform(input_data: pd.DataFrame) -> list:
        Transform the input_data using the previously defined mapping.
    _preprocess(input_data: pd.DataFrame) -> pd.DataFrame:
        Preprocesses the input data.
    _pad_input(x: list) -> np.array:
        Pads the input data to the specified token_capacity.
    _transform(data: pd.DataFrame) -> list:
        Converts the preprocessed data into the desired format.
    """

    def __init__(self, token_capacity):
        """
        Initialize DataPreprocessor with token_capacity.

        Parameters:
        -----------
        token_capacity : int
            The maximum length to which data sequences will be padded.
        """
        self.token_capacity = token_capacity
        self.index = None
        self.names = None
        self.values = None
        self.name_to_int = None

    def _pad_input(self, x):
        """
        Pads the input data with zeros up to the specified token_capacity.

        Parameters:
        -----------
        x : list
            The data to be padded.

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
        Preprocesses the input data by handling NaN values, creating a mask for numeric values, and manipulating variable names and values accordingly.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Input data to preprocess.

        Returns:
        --------
        pd.DataFrame
            Preprocessed data.
        """
        data = input_data.copy()
        data.dropna(inplace=True)
        # Create a mask for categorical variables
        mask = data[self.values].apply(
            lambda x: isinstance(x, (int, float, np.number)))
        data['mask'] = mask
        # Add the values of the categorical variables to their names to be encoded by the Embedding encoder and replace the values with 0 in the value column
        data[self.names] = np.where(
            mask, data[self.names], data[self.names] + "_" + data[self.values].astype(str))
        data[self.values] = np.where(mask, data[self.values], 0)
        data[self.values] = data[self.values].astype(float)
        return data

    def _transform(self, data):
        """
        Converts the preprocessed data into the desired format. 
        It assigns variable IDs, pads data, and groups by index.

        Parameters:
        -----------
        data : pd.DataFrame
            Preprocessed data.

        Returns:
        --------
        list
            List of numpy arrays for values_input, var_id_input, and categorical_mask_input.
        """
        data['var_id'] = data[self.names].map(self.name_to_int)
        data.dropna(subset=['var_id'], inplace=True)
        data = data[[self.index, 'var_id', self.values, 'mask']]
        data = data.groupby(self.index).agg(list).reset_index()
        for col in [self.values, 'var_id', 'mask']:
            data[col] = data[col].apply(lambda x: self._pad_input(np.array(x)))
        values_input = np.array(data[self.values].values.tolist())
        var_id_input = np.array(data['var_id'].values.tolist())
        mask_input = np.array(data['mask'].values.tolist())

        return [values_input, var_id_input, mask_input]

    def fit_transform(self, input_data, index, names, values):
        """
        Processes the input_data and define the mapping for variables to indices.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Input data to preprocess and transform.

        Returns:
        --------
        list
            List of numpy arrays for values_input, var_id_input, and categorical_mask_input.
        """
        # Check if input_data is a DataFrame
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("The input_data should be a Pandas DataFrame.")

        self.index = index
        self.names = names
        self.values = values

        data = self._preprocess(input_data)
        var_names = sorted(list(set(data[self.names])))
        self.name_to_int = {var: i + 1 for i, var in enumerate(var_names)}
        return self._transform(data)

    def transform(self, input_data):
        """
        Transforms the input_data using the previously defined mapping for variables to indices.

        Parameters:
        -----------
        input_data : pd.DataFrame
            Input data to transform.

        Returns:
        --------
        list
            List of numpy arrays for values_input, var_id_input, and categorical_mask_input.
        """
        data = self._preprocess(input_data)
        return self._transform(data)
