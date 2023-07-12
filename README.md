# Spatial Multi-attention Conditional Neural Process.

PyTorch for Spatial Multi-attention Conditional Neural Process.

## 
  * [Conditional Neural Process] (Attentive Neural Processes)代码根据Deep Mind提供的Tensorflow实现改写为PyTorch实现

All network is optimized by the Adam optimizer on a computer with RTX3080 GPU memory.

epoch=300,
learning rate=0.001
hidden number=128
n_tasks =30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")