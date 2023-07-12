# Spatial Multi-attention Conditional Neural Process.
Li-Li Bao, Jiang-She Zhang, Chun-Xia Zhang

This code implemented by Li-Li Bao for Spatial Multi-attention Conditional Neural Process is based on PyTorch 1.11.0 python 3.8.
GPU is NVIDIA GeForce RTX 3080.


PyTorch for Spatial Multi-attention Conditional Neural Process.

## 
  * [Conditional Neural Process] (Attentive Neural Processes)代码根据Deep Mind提供的Tensorflow实现改写为PyTorch实现

All network is optimized by the Adam optimizer on a computer with RTX3080 GPU memory.

epoch=300,
learning rate=0.001
hidden number=128
n_tasks =30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
