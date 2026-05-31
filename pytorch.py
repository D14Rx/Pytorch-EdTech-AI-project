import torch
import torch.nn as nn

#torch.manual_seed(1)

class EdTech(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 12)
        self.layer2 = nn.Linear(12, 1)

    def forward(self, x):       
        out = self.layer1(x)  #X = is a INPUT DATA
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.01) #relu = is a compilator that deletes - numbers after 0
        out = self.layer2(out)
        out = torch.sigmoid(out)
        return out

                #TENSORS:
data = torch.tensor([
    [0.9, 1.0, 0.8, 1.0], #: great knowledge
    [0.8, 0.9, 0.8, 1.0], #: great knowledge
    [1.0, 0.9, 0.8, 0.8], #: great knowledge
    [0.5, 0.5, 0.5, 0.5], #: mid knowledge
    [0.4, 0.6, 0.5, 0.5], #: mid knowledge
    [0.5, 0.6, 0.3, 0.6], #: mid knowledge
    [0.2, 0.2, 0.2, 0.2], #: bad knowledge
    [0.1, 0.3, 0.2, 0.1], #: bad knowledge
    [0.2, 0.1, 0.1, 0.3], #: bad knowledge
    [1.0, 1.0, 1.0, 1.0], #: ideal knowledge
    [0.0, 0.0, 0.0, 0.0], #: the skipper
], dtype=torch.float32)

target = torch.tensor([
    [1.0],  #: great knowledge
    [1.0],
    [1.0],  
    [0.5],  #: mid knowledge
    [0.5],  
    [0.5],  
    [0.1],  #: bad knowledge
    [0.1],  
    [0.1],  
    [1.0],  #: ideal knowledge
    [0.0],  #: the skipper
    ], dtype=torch.float32)

model = EdTech()

torch.nn.init.kaiming_normal_(model.layer1.weight, nonlinearity = 'leaky_relu')


            #TRAINING:
#optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)               
#mse_loss = nn.MSELoss()
#for epoch in range(3000):
#    optimizer.zero_grad()
#    prediction = model(data)
#    loss = mse_loss(prediction, target)
#    loss.backward()
#    optimizer.step()
#    if epoch % 100 == 0:
#        print(f'epochs: {epoch}, Error: {loss.item():.6f}')
#
#torch.save(model.state_dict(), 'edtech_model.pth')

model.load_state_dict(torch.load('edtech_model.pth'))
model.eval()


           #SOLVING RESULTS:
f_prediction = model(data).detach()
print(f_prediction)