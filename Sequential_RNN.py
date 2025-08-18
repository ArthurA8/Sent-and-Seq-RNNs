from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import median_abs_deviation
import torch
import torch.nn as nn
#import torch.optim as optim

def plot_npa_x_seq(arg: np.ndarray, title: str) -> tuple:
    # use tensor.detach().numpy
    fig, ax = plt.subplots()
    leny = len(arg)
    assert leny > 0
    xindex = np.arange(leny)
    ax.plot(xindex, arg)
    plt.title(title)
    return fig, ax

def sine_gen(seq_len, num_samples):
    X = []
    y = []
    for i in range(num_samples): 
        X_samp = [] 
        y_samp = []
        x = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, seq_len + 1)
        for i in x: 
            X_samp.append(np.sin(i))
            y_samp.append(np.sin(i))
        X_samp.pop(-1)
        y_samp.pop(0)
        X.append(X_samp)
        y.append(y_samp)
    return np.array(X), np.array(y)

def is_npa_constant(npa: np.ndarray) -> bool:
    """is numpy array all constant"""
    assert len(npa) >= 1, npa
    return bool(np.all(npa == npa[0]))


def is_npa_approx_constant(npa: np.ndarray) -> bool:
    """is numpy array approximately constant"""
    assert len(npa) >= 1, npa
    return np.allclose(npa, npa[0])

def vector_summary(data: np.ndarray) -> dict[str, Any]:
    """
    basic statistical summary of numeric data
    dysfunctional vectors (empty, zero-len) only have 1 stat, "overall"
    """
    result: dict[str, Any] = {}
    if data is None:
        result["overall"] = "vector_summary_one, None"
        return result  # Early Return
    assert data.ndim == 1, data.ndim
    if len(data) == 0:
        result["overall"] = "vector_summary_one, zero-length"
        return result  # Early Return
    if is_npa_constant(data):
        result["summary"] = f"vector_summary_one, constant {data[0]}"
    elif is_npa_approx_constant(data):
        result["summary"] = f"vector_summary_one, approx constant {data[0]}"
    result["length"] = len(data)
    result["dtype"] = data.dtype
    result["mean"] = np.mean(data)
    result["median"] = np.median(data)
    result["mode"], _ = scipy.stats.mode(data, keepdims=False)
    result["sum"] = np.sum(data)
    data_std_dev = np.std(data)
    result["std-dev"] = data_std_dev
    result["var"] = data_std_dev**2
    result["mad"] = median_abs_deviation(data)
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    result["min"] = data_min
    result["max"] = data_max
    result["range"] = data_range
    if len(data) <= 5:
        result["data"] = data
    return result

###############################################################################

prompt_inputs = False
print(f"{prompt_inputs=}")
if prompt_inputs:
    seq_len = int(input('Input data points per sample: '))
    num_samples = int(input('Input num of sin data samples: '))
    input_size = int(input('RNN input size: '))
    hidden_size = int(input('RNN hidden size: '))
    output_size = int(input('RNN output size: '))
    num_epochs = int(input('Input num EPOCHS: '))
    lr = float(input('Input learning rate: '))
else:
    seq_len = 50
    num_samples = 10
    input_size = 1
    hidden_size = 20
    output_size = 1
    num_epochs = 100 # 1000
    lr = 0.01
print(f"{seq_len=}")
print(f"{num_samples=}")
print(f"{input_size=}")
print(f"{hidden_size=}")
print(f"{output_size=}")
print(f"{num_epochs=}")
print(f"{lr=}")

###############################################################################

X, y = sine_gen(seq_len, num_samples)
# X, y are numpy matrices, nrow=num_samples, ncol=seq_len, eg 10 examples (rows) of len 50 each
print(f"npas {X.shape=}, {y.shape=}")

plot_and_stats = True
print(f"{plot_and_stats=}")
if plot_and_stats:
    # check and plot
    # X, y same shape and pretty close
    chosen_Xindex = 0
    print(f"{chosen_Xindex=}")
    plot_npa_x_seq(X[chosen_Xindex], f"X_{chosen_Xindex}")
    plt.show(); plt.close()
    plot_npa_x_seq(y[chosen_Xindex], f"y_{chosen_Xindex}")
    plt.show(); plt.close()
    deltas = X - y
    plot_npa_x_seq(deltas[chosen_Xindex], f"deltas_{chosen_Xindex}")
    plt.show(); plt.close()
    # stats summaries
    summary_X = vector_summary(X[chosen_Xindex])
    summary_y = vector_summary(y[chosen_Xindex])
    summary_deltas = vector_summary(deltas[chosen_Xindex])
    print(f"{summary_X=}")
    print("*" * 80 + "\n")
    print(f"{summary_y=}")
    print("*" * 80 + "\n")
    print(f"{summary_deltas=}")
    print("*" * 80 + "\n")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
print(f"tensors {X.shape=}, {y.shape=}")

###############################################################################

class RNN(nn.Module): 

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() 
        self.network = nn.RNN(input_size, hidden_size, batch_first=True)
        # fl = final layer?
        self.fl = nn.Linear(hidden_size, output_size)
    
    def forward(self, x): 
        h0 = torch.zeros(1, hidden_size).to(x.device)
        out, hn = self.network(x, h0)
        #print(out)
        out = self.fl(out)
        return out

###############################################################################

model = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

predictions = []

def train(): 

    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        n = 0
        for xinput in X: 
            reshaped = np.reshape(xinput, (seq_len, 1))
            pred = model(reshaped)
            predictions.append(pred)
            y_corr = y[n]
            reshaped_ycorr = np.reshape(y_corr, (seq_len, 1))
            loss = criterion(pred, reshaped_ycorr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

        print("EPOCH {}, LOSS {}".format(epoch, loss))

###############################################################################
    
train()

chosen_y_index = 0
prediction_index = len(predictions) - num_samples + chosen_y_index
plt.plot(y[chosen_y_index].detach().numpy(), label='True')
plt.plot(predictions[prediction_index].detach().numpy(), label='Prediction')
plt.legend()
plt.show()
# calc deltas, and stats on deltas
y_chosen_npa = y[chosen_y_index].detach().numpy()
pred_chosen_npa_old = predictions[prediction_index].detach().numpy()
pred_chosen_npa = pred_chosen_npa_old.reshape(seq_len)
print(f"{y_chosen_npa.shape=}")
print(f"{pred_chosen_npa.shape=}")
delta_pred = y_chosen_npa - pred_chosen_npa
summary_delta_pred = vector_summary(delta_pred)
print(f"{summary_delta_pred=}")
print("*" * 80 + "\n")

plt.plot(y_chosen_npa, label='True')
plt.plot(pred_chosen_npa, label='Prediction')
plt.plot(delta_pred, label='DeltaPred')
plt.legend()
plt.show()