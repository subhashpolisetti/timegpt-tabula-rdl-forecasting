# TimeGPT_Tabula9_Relational_Deep_Learning

This repository provides a comprehensive suite of notebooks that demonstrate the application of TimeGPT for relational deep learning (RDL) tasks, with a specific focus on time series forecasting, anomaly detection, and other predictive models. These projects leverage cutting-edge machine learning techniques to address real-world challenges such as energy demand forecasting, Bitcoin price prediction, synthetic data generation, and more. The following sections outline each project and provide access to the corresponding Google Colab notebooks.

## Project Links

### **TimeGPT Multivariate Forecasting**  
[TimeGPT_Multivariate Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/TimeGPT_Multivariate.ipynb)  
**Description:** This notebook explores the use of TimeGPT for multivariate time series forecasting. The model leverages relational deep learning techniques to predict multiple time series variables simultaneously, offering a holistic view of temporal dependencies. This approach is ideal for datasets where multiple features evolve over time and interact with each other.

### **Fine-Tuning Time Series with TimeGPT**  
[TimeSeries_FineTuning_with_TimeGPT Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/TimeSeries_FineTuning_with_TimeGPT.ipynb)  
**Description:** This notebook demonstrates the fine-tuning process of TimeGPT on time series data. Fine-tuning allows for adapting a pre-trained model to more specific time series tasks, improving accuracy by leveraging knowledge from broader datasets. This method is particularly useful when working with domain-specific time series data where training a model from scratch may be computationally expensive.

### **Anomaly Detection with Nixtla**  
[TimeSeries_Anomaly_Detection_with_Nixtla Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/TimeSeries_Anomaly_Detection_with_Nixtla.ipynb)  
**Description:** In this notebook, we focus on anomaly detection for time series data using Nixtla, a library specifically designed for time series forecasting and anomaly detection. By identifying outliers or unexpected trends in the data, this notebook helps to detect unusual patterns that could signal important events or errors in the data, critical for applications such as fraud detection or system health monitoring.

### **Forecasting Energy Demand**  
[Forecasting_Energy_Demand Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/Forecasting_Energy_Demand.ipynb)  
**Description:** This notebook demonstrates forecasting energy demand using time series analysis. Energy demand is a classic application for time series forecasting, where patterns in past energy consumption are used to predict future needs. This notebook implements TimeGPT to generate accurate forecasts, helping utilities optimize energy distribution and prepare for peak load periods.

### **Bitcoin Price Prediction Using Nixtla**  
[Bitcoin_Price_Prediction_Using_Nixtla Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/Bitcoin_Price_Prediction_Using_Nixtla.ipynb)  
**Description:** This notebook focuses on predicting Bitcoin prices using Nixtla’s anomaly detection and forecasting capabilities. Time series models are applied to Bitcoin price data, enabling predictive insights into market behavior. The model’s ability to capture the volatility and trends within cryptocurrency markets is essential for financial analysts and traders.

### **Synthetic Data Generation and Analysis for Insurance Dataset**  
[Synthetic_Data_Generation_and_Analysis_Insurance_dataset Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/Synthetic_Data_Generation_and_Analysis_Insurance_dataset.ipynb)  
**Description:** In this notebook, synthetic data generation is used to create a simulated dataset that mimics real-world insurance data. This method is useful when real datasets are scarce or sensitive. The notebook demonstrates how synthetic data can be used to train models for insurance claim predictions, customer segmentation, and risk assessment while ensuring privacy and compliance with data protection regulations.

### **Price Classification Prediction Using Tabula Model Inference**  
[Price_Classification_Prediction_Using_Tabula_Model_inference Colab Link](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/Price_Classification_Prediction_Using_Tabula_Model_inference.ipynb)  
**Description:** This notebook demonstrates the use of Tabula models for price classification tasks. The model classifies products or services into predefined price categories based on features such as brand, quality, and specifications. Tabula, a state-of-the-art deep learning model, is used for feature extraction and classification, showcasing its power in handling complex datasets and making high-accuracy predictions.



## Graph Neural Network for Tabular Prediction Task

### **Graph Neural Network for Driver Position Prediction (using RelBench)**  
[Graph_Neural_Network_for_Driver_Position_Prediction_RelBench](https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting/blob/main/Graph_Neural_Network_for_Driver_Position_Prediction_RelBench.ipynb)  
**Description:** This notebook demonstrates the use of graph neural networks (GNNs) for a tabular prediction task using RelBench, a benchmarking framework for relational deep learning tasks. The focus is on predicting the position of drivers based on relational data such as previous locations and surrounding environment features. GNNs are powerful tools for modeling relationships in tabular data and excel in tasks where entities are interconnected.


## Video Demonstration

A comprehensive video demonstration of all the projects can be found here:  
[Click Here for Video Demonstration](https://drive.google.com/drive/folders/1vJrBywHMl0vOnp3ODRoLJnVKnIQ1c6IA?usp=sharing)

## Requirements

To run the code and notebooks, ensure you have the following dependencies installed:

- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow` or `pytorch`, and `nixtla`
- Jupyter Notebook/Colab environment

## How to Use

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/subhashpolisetti/timegpt-tabula-rdl-forecasting.git
