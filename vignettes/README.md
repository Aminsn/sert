# Vignettes for `sert`

This folder contains the vignettes in the form of Jupyter notebooks for the `sert` package. Each vignette provides detailed examples, explanations, and use-cases to help users understand how to effectively utilize the functionalities of the `sert` package. The package offers four classes for modeling, two loss function classes, and two classes for data preparation. The details of these classes are as follows:

1. **Model Classes**
    - `SERT`: This stands for "Sparse Encoder Representations from Transformers" and is suitable for both classification and regression tasks on non-temporal data.
    - `TimeSERT`: This is an abbreviation for "Timeseries Sparse Encoder Representations from Transformers" and is designed for both classification and regression tasks on timeseries data.
    - `SERNN`: This stands for "Sparse Encoder Representations from Neural Networks" and can be applied to both classification and regression tasks on non-temporal data.
    - `TimeSERNN`: It stands for "Timeseries Sparse Encoder Representations from Neural Networks", this class is tailored for both classification and regression tasks on timeseries data.

    Note: The `SERNN` and `TimeSERNN` classes are simpler versions of the `SERT` and `TimeSERT` classes, respectively. They do not incorporate the transformer architecture, but they offer faster performance.

2. **Loss Function Classes**
    - `WeightedCrossentropy`: The class for weighted cross-entropy loss.
    - `MaskedMSE`: The class for masked mean squared error loss.

3. **Data Preparation Classes**
    - `DataPreparer`: A class tailored for preparing non-sequential data for use with the model classes.
    - `SeqDataPreparer`: Designed specifically for preparing sequential data to be used with the model classes.

## List of Vignettes

1. **classification.ipynb**
    - **Description:** In this vignette, we will demonstrate how to use the `DataPreparer` class for preparing a tabular data to be used with `SERT` and `WeightedCrossentropy` classes for a classification problem.

2. **multivariate_forecasting.ipynb**
    - **Description:** In this vignette, we will demonstrate how to use the `SeqDataPreparer` class to prepare multivariate time series data for use with the `TimeSERT` and `MaskedMSE` classes in a forecasting problem.

3. **regression.ipynb**
    - **Description:** In this vignette, we will demonstrate how to use the `DataPreparer` class for preparing a tabular data to be used with `SERT` and `MaskedMSE` classes for a regression problem.

4. **timeseries_classification.ipynb**
    - **Description:** In this vignette, we will demonstrate how to use the `SeqDataPreparer` class for preparing multiple timeseries data to be used with `TimeSERT` and `WeightedCrossentropy` classes for a timeseries classification problem.



