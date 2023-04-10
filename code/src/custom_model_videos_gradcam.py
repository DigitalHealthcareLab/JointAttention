import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class GradCamModel(nn.Module):
    def __init__(self, model):
        super(GradCamModel, self).__init__()
        model_shits_1 = nn.Sequential(*list(model.cnn.baseModel.children())[:-2])
        model_shits_2 = nn.Sequential(*list(model.cnn.baseModel.children())[-2:])
        self.model_shits_1 = model_shits_1
        self.model_shits_2 = model_shits_2
        self.rnn = model.lrcn
        self.attn = nn.Sequential(*list(model.attention.children())[:2])
        # self.attn1 = nn.Sequential(*list(model.attention.children())[:2])
        # self.attn2 = nn.Sequential(*list(model.attention.children())[:2])
        self.fc1 = model.mlp.net[0]        
        self.v = model.mlp.net[3]

        
    def activations_hook(self, grad):
        self.gradients.append(grad)
        
    def forward(self, x):
        self.gradients = []
        b_z, ts, c, h, w = x.shape
        ii = 0

        y_1 = self.model_shits_1((x[:,ii]))

        h = y_1.register_hook(self.activations_hook)
        y = self.model_shits_2((y_1))
        y = y.view(b_z,-1)
        
        output_list = []
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        output_list.append(output)
        for ii in range(1, ts):

            y_1 = self.model_shits_1((x[:,ii]))
#             y_1.requires_grad = True
            h = y_1.register_hook(self.activations_hook)
            y = self.model_shits_2((y_1))
            y = y.view(b_z,-1)
#             rnn = model.rnn
            out, (hn, cn) = self.rnn(y.unsqueeze(1))
            output_list.append(out)
#         out = self.dropout(out[:,-1])
        
#         out = self.fc1(out[:,-1])
    
    
        output_list = torch.stack(output_list).squeeze(2)
        output_list = output_list.permute(1,0,2)
        #output_list: [batch, seq_len, hidden_dim]
        
        
#         out = self.dropout(out[:,-1])
        out = out[:,-1].squeeze(1)
        #out [batch, hidden_dim]
        
        src_len = output_list.shape[1]
        
        hidden = out.unsqueeze(1).repeat(1, src_len, 1)
        
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        attention = torch.tanh(self.attn(output_list)).squeeze(2)
        # energy = torch.tanh(self.attn(torch.cat((output_list), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]
        
        # attention = self.v(energy).squeeze(2)
        #attention = [batch size, src len]
        
        a = F.softmax(attention, dim = 1)
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
#         encoder_outputs = output_list.permute(1, 0, 2)
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, output_list)
        #weighted = [batch size, 1, enc hid dim * 2]
        
        out = weighted[:,0,:]
        out = self.fc1(out)
        return out 
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return torch.stack(self.gradients, dim=2)
    
    # method for the activation exctraction
    def get_activations(self, x):
        ts = x.shape[1]
        ii = 0
#         model = self.grad_models[ii]
#         baseModel = model.baseModel
#         model_shit_1 = nn.Sequential(*list(baseModel.children())[:-2])
        activations = []
        for ii in range(ts):
            activations.append(self.model_shits_1((x[:,ii])))
        return torch.stack(activations, dim=2)