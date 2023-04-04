import torch
import torch.nn as nn
from typing import Tuple
import torchvision.models as models

batch_size = 3

class CNN(nn.Module):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len

        baseModel = models.resnet18(pretrained=True)
        lt = 8
        cntr = 0
        for child in baseModel.children():
            cntr += 1
            if cntr < lt:
                for param in child.parameters():
                    param.requires_grad = False
        num_features = baseModel.fc.in_features
        baseModel.fc = Identity()

        self.baseModel = baseModel

    def forward(self, x):
        self.batch_size, self.seq_len, c, h, w = x.size()
        x = x.view(self.batch_size * self.seq_len, c, h, w)
        x = self.baseModel(x)
        x = x.view(self.batch_size, self.seq_len, -1)
        return x


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class LRCN(nn.Module):
    def __init__(self, input_size: int, num_hiddens: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.num_hiddens, self.num_layers)

    def forward(self, x) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, hidden_state = self.lstm(x)
        return lstm_out, hidden_state


class CustomAttention(nn.Module):
    def __init__(self, num_hiddens, attention_dim):
        super().__init__()
        self.lstmoutput_attention_projection = nn.Linear(num_hiddens, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute z_t but will use only alphas_t for visualization @ testing
        Args:
        lstm_outputs (torch.Tensor): [batch_size, seq_len, num_hiddens]
        """
        lstmoutput_attention = self.lstmoutput_attention_projection(lstm_outputs)
        # In: (batch_dim, seq_len, num_hiddens), Out: (batch_dim, seq_len, attention_dim)
        attention = self.attention(self.ReLU(lstmoutput_attention)).squeeze(2)
        # In: (batch_dim, seq_len, attention_dim), Out: (batch_size, seq_len)
        alphas_t = self.softmax(attention)  # Out: (batch_dim, seq_len)
        attention_weighted_encoding = (lstm_outputs * alphas_t.unsqueeze(2)).sum(
            dim=1
        )  # Out: (batch_diim, num_hiddens)
        return attention_weighted_encoding, alphas_t


class CustomMLP(nn.Module):
    def __init__(self, num_hiddens, dropout, output_size):
        super().__init__()
        self.linear1 = nn.Linear(num_hiddens, num_hiddens)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(num_hiddens, output_size)

        self.net = nn.Sequential(self.linear1, self.relu, self.drop1, self.linear2)

    def forward(self, X):
        return self.net(X)  # return [1, 2] batch_size, class_size


class Resnet18Rnn(nn.Module):
    def __init__(
        self,
        batch_size,
        input_size,
        output_size,
        seq_len,
        num_hiddens,
        num_layers,
        dropout,
        attention_dim,
    ):
        super().__init__()
        self.cnn = CNN(batch_size, seq_len)
        self.lrcn = LRCN(input_size, num_hiddens, num_layers)
        self.attention = CustomAttention(num_hiddens, attention_dim)
        self.mlp = CustomMLP(num_hiddens, dropout, output_size)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: x [batch_size, seq_len, 3, 224, 224]
        Returns: output, attention
        """
        x = self.cnn(x)
        x, _ = self.lrcn(x)  # X: [150, 128]
        z_t, alphas_t = self.attention(x)
        return self.mlp(z_t), alphas_t


if __name__ == "__main__":
    # model1 = CNN(batch_size=3, seq_len=100)
    # x = model1(x)
    # print(x.shape)

    model5 = Resnt18Rnn(
        batch_size=batch_size,
        input_size=512,
        output_size=2,
        seq_len=100,
        num_hiddens=512,
        num_layers=2,
        dropout=0.5,
        attention_dim=100,
    )
    # print(model5)
    x = torch.rand(size=(3, 100, 3, 224, 224))
    out, alphas_t = model5(x)
    print(out.shape)  # out: [3, 2]
    print(alphas_t.shape)  # alphas_t: [3, 100]

#     for name, param in vgg.named_parameters():
#           if param.requires_grad:print(name)
