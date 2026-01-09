import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
    def forward(self, x):
        return self.linear(x)

model = LogisticRegression()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()

def predict_digit(img):
    img = img.convert("L").resize((28,28))
    arr = np.array(img) / 255.0
    tensor = torch.tensor(arr.reshape(1,784), dtype=torch.float32)
    with torch.no_grad():
        out = model(tensor)
        _, pred = torch.max(out,1)
    return pred.item()

st.title("MNIST Digit Recognition Demo")
file = st.file_uploader("Upload digit image", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, width=150)
    st.write("Predicted Digit:", predict_digit(img))
