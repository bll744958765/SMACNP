# Spatial Multi-attention Conditional Neural Process.
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang

This code implemented by Li-Li Bao for Spatial Multi-attention Conditional Neural Process is based on PyTorch 1.11.0 Python 3.8.
GPU is NVIDIA GeForce RTX 3080.

The main contributions of this paper are as follows:
1. We propose a spatial prediction model, named SMACNP, where we design an auto-encoder module that is associated with GPs and NNs for spatial small sample prediction. Our method learns a specific joint embedding between the geographic coordinates and potential other inputs (i.e., explanatory features), having the advantage of making good predictions of targets and evaluating the reliability of prediction results.
2, The approach derives the mean and variance of GPs from two modules to predict the distribution of targets. In particular, inspired by GPs, the proposed method quantifies uncertainty that is only related to the input of the target point and not to the output target value, which
is beneficial for the model to provide robust estimates.
3. Based on the experiments implemented on simulation datasets and real-world datasets to examine and compare SMACNP with other methods, it is shown to perform best in spatial small sample prediction tasks.


## 
  * [Conditional Neural Process] (Attentive Neural Processes)代码根据Deep Mind提供的Tensorflow实现改写为PyTorch实现

All network is optimized by the Adam optimizer on a computer with RTX3080 GPU memory.

epoch=300,
learning rate=0.001
hidden number=128
n_tasks =30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
