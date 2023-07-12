# Spatial Multi-attention Conditional Neural Process.
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang

This code implemented by Li-Li Bao for Spatial Multi-attention Conditional Neural Process is based on PyTorch 1.11.0 Python 3.8.
GPU is NVIDIA GeForce RTX 3080.

epoch=300,
learning rate=0.001
hidden number=128
n_tasks =30
device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")


## The main contributions of this paper are as follows:
1. We propose a spatial prediction model, named SMACNP, where we design an auto-encoder module that is associated with GPs and NNs for spatial small sample prediction. Our method learns a specific joint embedding between the geographic coordinates and potential other inputs (i.e., explanatory features), having the advantage of making good predictions of targets and evaluating the reliability of prediction results.
2. The approach derives the mean and variance of GPs from two modules to predict the distribution of targets. In particular, inspired by GPs, the proposed method quantifies uncertainty that is only related to the input of the target point and not to the output target value, which
is beneficial for the model to provide robust estimates.
3. Based on the experiments implemented on simulation datasets and real-world datasets to examine and compare SMACNP with other methods, it is shown to perform best in spatial small sample prediction tasks.


## Dataset and Experiment
We validated the superiority of the proposed method on one simulation dataset and 3 real-world data sets using different ratios of training and testing sets. To demonstrate the stability of SMACNP, we recorded the mean and standard deviation of the outcomes of 10 experiments. The model is assessed by MAE (Mean Absolute Error), and RMSE (Root Mean Squared Error). 

## Implementation of SMACNP on simulation dataset
You can run data_split.py first to spilt generation_data_noise.csv into train.csv and valid.csv.
generation_data_noise1.csv is a new simulation data set that can be used to test the performance of SMANP. 
Then, you can run train_smacnp to implement the experiment.
Last, you can run test_smacnp to show the prediction performance.
Fig. 1 displays the ability of the SMACNP to capture spatial relationships on a simulated dataset.
Fig. 1(a) displays the ground truth in the test set, while the prediction maps produced by SMACNP
for various division ratios of the training set are shown in Fig. 1(b) and Fig. 1(c).

![Figure](https://github.com/bll744958765/SMACNP/assets/92556725/d98a331f-6e34-43a5-802a-ee53e27a51e7)




## 
  * [Conditional Neural Process] (Attentive Neural Processes)
  * The code is rewritten from the Tensorflow implementation provided by Deep Mind to the PyTorch implementation

