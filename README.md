<img src="sert_logo.png" width="130" height="130" align="left"> 

## Sparse Encoder Representations from Transformers

<br clear="left" />


[![PyPI version](https://badge.fury.io/py/sert.svg)](https://badge.fury.io/py/sert)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
[![DOI](https://zenodo.org/badge/688131344.svg)](https://zenodo.org/badge/latestdoi/688131344)
![Static Badge](https://img.shields.io/badge/Lifecycle-experimental-orange)


`sert` is a Python module for machine learning built on top of Tensorflow. It is designed for deep learning on sets, focused on predictive modeling tasks, including both regression and classification, on time series and non-time series data.

Many datasets, including standard tabular ones with p columns and N rows, can be represented as a set of observations in the format [row id, column name, value]. This is equivalent to pivoting a wide Nxp table into a long format:

### Wide Dataframe:

| Name  | Apple | Banana | Cherry |
|-------|-------|--------|--------|
| Alice | 5     | 7      | NA     |
| Bob   | 3     | 4      | 2      |

---

### Long Dataframe:

| Name  | Fruit  | Rating |
|-------|--------|--------|
| Alice | Apple  | 5      |
| Alice | Banana | 7      |
| Alice | Cherry | NA     | 
| Bob   | Apple  | 3      |
| Bob   | Banana | 4      |
| Bob   | Cherry | 2      |


However, in some applications, storing data in a wide format is not feasible. For instance, with log data that's stored as [timestamp, user id, action], the data is inherently in a long format. Pivoting this data into a wide format isn't sensible due to the large number of unique values in the timestamp column.

`sert` is designed to work with data in this long format, making it suitable for a variety of problems, including tabular, time series, and log data. The benefits of using long format data with `sert` are:

1. It allows for the removal of individual cells with missing values without having to remove the entire row in the wide format. As such, `sert` can handle missing values without requiring imputation or the removal of observed data from the same row.

2. It can be applied to datasets that are best represented in a long format, such as log data or multivariate non-aligned time series data.

`sert` leverages the powerful transformer architecture to learn from sets by discerning how to focus on important observations. By utilizing the transformer architecture, `sert` is parallelizable and scalable to large multivariate time series datasets, unlike RNNs which are sequential and cannot be parallelized.

## Installation

To install `sert`, simply run:

```bash
pip install sert
```

**Note**: Dependencies will be installed automatically.

## Dependencies

`sert` requires:

- NumPy
- Pandas
- tensorflow (>= 2.0.0)
- keras_nlp (>= 0.6.0)


## Quick Start

Below is an example of how to use `sert` to classify irregular time series. For more examples and details on how to use the package for different problems, please refer to the [vignettes](https://github.com/Aminsn/sert/tree/master/vignettes).

```python
from sert.models import TimeSERT
from sert.preprocessing import SeqDataPreparer
from sert.datasets import ts_classification
import tensorflow as tf

# Load the dataset
X_train, X_test, y_train, y_test = ts_classification()

# Prepare the data for SERT
token_cap = X_train.groupby('id').size().max()

processor = SeqDataPreparer(token_capacity=token_cap)
train_input = processor.fit_transform(
    X_train, index='id', times='time', names='var_name', values='value')
test_input = processor.transform(X_test)

# Instantiate the model
model = TimeSERT(num_var=1,
                 emb_dim=15,
                 num_head=3,
                 ffn_dim=5,
                 num_repeat=2,
                 num_out=y_train.shape[1],
                 task='classification')

# Compile the model
categorical_loss = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer='adam', loss=categorical_loss, metrics=['accuracy'])

# Fit the model
model.fit(train_input, y_train, epochs=100, batch_size=250)

# Predictions
y_pred = model.predict(test_input)

# Evaluate the model
_, accuracy = model.evaluate(test_input, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```


## License

Distributed under the Apache License. See `LICENSE` for more information.

## Contact

- **Name**: Amin Shoari Nejad
- **Email**: amin.shoarinejad@gmail.com
