import skimage as ski
import torch

from model import MNIST_Model

# load the model
model = MNIST_Model()
model.load_state_dict(torch.load("model/mnist4_.pth"), weights_only=True)

# load img
