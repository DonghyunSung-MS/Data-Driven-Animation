import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        input 70*240 (dof*frame)
        output 256*120 (hidden unit*frame)
        """
        kernel_size = 25
        input_channel = 70
        out_channel = 256
        drop_prob = 0.25
        self.conv = nn.Conv1d(input_channel, out_channel, kernel_size, padding=12, padding_mode = 'zeros')
        self.dropout1 = torch.nn.Dropout(p=drop_prob)
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, return_indices=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.dropout2 = torch.nn.Dropout(p=drop_prob)
        self.deconv = nn.ConvTranspose1d(out_channel, input_channel, kernel_size, padding=12, padding_mode = 'zeros')


    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.dropout1(x)
        x = self.max_pool1d(x)
        x = self.upsample(x)
        x = torch.relu(self.deconv(x))
        x = self.dropout2(x)
        return x

    def decoding(self, x): #256*120 to 70*240 (batch, channel L)
        x = self.upsample(x)
        x = torch.relu(self.deconv(x))
        x = self.dropout2(x)
        return x

    def encoding(self, x):
        x = torch.relu(self.conv(x))
        x = self.dropout1(x)
        x = self.max_pool1d(x)
        return x



from torchsummary import summary
x = torch.randn(1,70,240).to('cuda')
model = AutoEncoder().to('cuda')
#print(model(x).shape)
summary(model,(70,240))
#print(model.encoding(x).detach().shape)
y = torch.randn(1,256,120).to('cuda')
print(model.decoding(y).detach().shape)
