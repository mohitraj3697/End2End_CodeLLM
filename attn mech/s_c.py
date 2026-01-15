import torch
import torch.nn as nn



class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    



class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
          
        ])

    def forward(self, x):
        for layer in self.layers:
            
            layer_output = layer(x)
            
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x




layer_sizes = [3, 3, 3, 3,  1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123) 
model_without_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=False          #shortcut disabled
)



def print_gradients(model, x):
   
    output = model(x)
    target = torch.tensor([[0.]])

    
    loss = nn.MSELoss()
    loss = loss(output, target)
    
   
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

print_gradients(model_without_shortcut, sample_input)


print("\n")



model_with_shortcut = ExampleDeepNeuralNetwork(
layer_sizes, use_shortcut=True   #shortcut enabled
)

print_gradients(model_with_shortcut, sample_input)