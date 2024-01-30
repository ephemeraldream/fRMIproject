# Time Series Classification with Topological Data Analysis

![xxx](https://github.com/ephemeraldream/fRMIproject/blob/main/lands.png)

## Introduction

In the realm of Machine Learning and big data, the complexity of data structures demands the development of innovative methods. This repository presents an approach that leverages Topological Data Analysis (TDA) to classify time series data using the Time Delay Embedding technique. While classical one-dimensional time series may be topologically trivial, Time Delay Embedding can reveal non-trivial topological structures. Our focus is on calculating persistent entropy and applying its kernel to popular classifiers, enhancing the dataset with topological insights.

## Classification of ECG Time Series

We propose a pipeline that begins with open-access multidimensional time series data, which we reduce to a one-dimensional form. Through Taker Delay Embedding and triangulation, we compute homology and derive barcode entropy. The resulting topological features are then used to train a Support Vector Machine (SVM) for series classification based solely on topological information.

### Data

![xxx](https://github.com/ephemeraldream/fRMIproject/blob/main/2healthy.png)

The dataset comprises 45,152 ECGs from patients, with each ECG having a 17-dimensional time series. To simplify analysis and due to strong correlations between readings, we compress these series into a one-dimensional form.

### Taken's Time Delay Embedding


![xxx](https://github.com/ephemeraldream/fRMIproject/blob/main/1_KZjxg-nN9Zy6v6OvpvXAng.jpg)


We transform the series by mapping neighboring points as coordinates in the embedding space, altering the series' "appearance" and revealing its topological features.

### Persistence Entropy

After topological analysis, we obtain a dataset of persistent entropy, a 2-dimensional vector representing the feature space. We visualize the raw, truncated, and normalized collections of all persistence entropy for all observations, considering the impact of outliers on ML models.

### Accuracy on Test Data

![xxx](https://github.com/ephemeraldream/fRMIproject/blob/main/truncated.png)


We evaluate various models on the raw, truncated, and normalized data structures, using an 80/20 train/test split and 5-fold cross-validation. The SVM with RBK achieves the highest accuracy, demonstrating the potential of learning from topological data.

## Discussion

The SVM with Radial Basis Kernel outperforms other models, suggesting the viability of topological features for classification. However, the effectiveness of this approach is limited compared to recurrent and convolutional neural networks. Future work may involve combining persistent entropy with other transformations to improve accuracy.

## Conclusion

This study explores the classification of time series using TDA. We delve into the theoretical aspects of TDA and apply them to classify time series with Taken's Time Delay embedding. The results indicate that TDA can detect shifted patterns and heart functionality crises, with certain classifiers like SVM showing promise for future research at the intersection of TDA and time series classification.

## Libraries

- `giotto-tda`: Topological Data Analysis
- `gudhi`: TDA library
- `plotly`: Visualization
- `sklearn`: Classical analysis

## Usage

To use this repository for classifying time series with TDA, please ensure you have the above libraries installed. Follow the provided notebooks for a step-by-step guide on the classification pipeline.

## Contributions

Contributions to this project are welcome. Please submit issues or pull requests with improvements or bug fixes.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Contact

For any inquiries or collaboration requests, please open an issue in this repository.

