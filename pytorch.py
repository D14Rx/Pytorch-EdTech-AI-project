import torch
import torch.nn as nn

#torch.manual_seed(1)

class EdTech(nn.Module):
    def __init__(self):
        super().__init__()

        self.marks1 = nn.Linear(4, 12)
        self.marks2 = nn.Linear(12, 1)

        self.blanket1 = nn.Linear(2, 12)
        self.blanket2 = nn.Linear(12, 1)

        self.wishes1 = nn.Linear(2, 12)
        self.wishes2 = nn.Linear(12, 1)


    def forward_marks(self, x_marks):       
        out = self.marks1(x_marks)  #X = is a INPUT DATA
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
        out = self.marks2(out)
        out = torch.sigmoid(out)
        return out
    
    def forward_blanket(self, x_blanket):       
        out = self.blanket1(x_blanket)  #X = is a INPUT DATA
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
        out = self.blanket2(out)
        out = torch.sigmoid(out)
        return out
    
    def forward_wishes(self, x_wishes):       
        out = self.wishes1(x_wishes)  #X = is a INPUT DATA
        out = torch.nn.functional.leaky_relu(out, negative_slope=0.01)
        out = self.wishes2(out)
        out = torch.sigmoid(out)
        return out
    
    marks_result = 

    def forward(self, x_marks, x_blanket, x_wishes):
        return 

marks = torch.tensor([

#1-num: math marks
#2-num: physics marks
#3-num: biology marks
#4-num: chemistry marks

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


blanket = torch.tensor([

#1-num: alarming or confident (a[0.0]/c[1.0])
#2-num: humanitary or technary (h[0.0]/t[1.0])

    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=torch.float32)

wishes = torch.tensor([

#1-num: student wants to phys-math OR chem-bio(P-M[0.0]/C-B[1.0])

    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
], dtype=torch.float32)

            #TARGETS:

target_marks = torch.tensor([
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

target_blanket = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
], dtype= torch.float32)

target_wishes = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.0],
    [1.0],
    [0.1],
    [1.0],
    [0.0],
    [1.0],
], dtype= torch.float32)


#fusion_data = torch.cat((data, blanket), dim = 1) #dim = 1 == ось для Вертикальных Строк
#main_data = torch.cat((fusion_data, wishes), dim = 1)

model = EdTech()

torch.nn.init.kaiming_normal_(model.marks1.weight, nonlinearity = 'leaky_relu')
torch.nn.init.kaiming_normal_(model.blanket1.weight, nonlinearity = 'leaky_relu')
torch.nn.init.kaiming_normal_(model.wishes1.weight, nonlinearity = 'leaky_relu')


            #TRAINING:
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)               
mse_loss = nn.MSELoss()
for epoch in range(3000):
    optimizer.zero_grad()
    prediction = model(marks)
    loss = mse_loss(prediction, target_marks)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epochs: {epoch}, Error: {loss.item():.6f}')


optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)               
mse_loss = nn.MSELoss()
for epoch in range(3000):
    optimizer.zero_grad()
    prediction = model(blanket)
    loss = mse_loss(prediction, target_blanket)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epochs: {epoch}, Error: {loss.item():.6f}')


optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)               
mse_loss = nn.MSELoss()
for epoch in range(3000):
    optimizer.zero_grad()
    prediction = model(wishes)
    loss = mse_loss(prediction, target_wishes)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epochs: {epoch}, Error: {loss.item():.6f}')


torch.save(model.state_dict(), 'edtech_model.pth')

model.load_state_dict(torch.load('edtech_model.pth'))
model.eval()

           #SOLVING RESULTS:
prediction1 = model(marks).detach()
print(prediction1)

prediction2 = model(blanket).detach()
print(prediction2)

prediction3 = model(wishes).detach()
print(prediction3)
