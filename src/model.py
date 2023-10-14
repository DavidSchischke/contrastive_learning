from torch import nn 
from torch.nn.functional import softmax

class ExampleNet(nn.Module): 
    """
    Somewhat similar to the LeNet architecture, but a bit bigger and with 
    PReLU activation.
    """
    def __init__(self, num_classes: int = 47) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        

    def forward(self, x): 
        y = self.conv1(x)
        y = self.prelu1(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = self.prelu2(y)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.prelu3(y)
        y = self.fc2(y)
        y = self.prelu4(y)
        y = self.fc3(y)
        return y
    
if __name__ == "__main__": 
    import sys 
    sys.path.append(".")

    from src.utils import get_device
    from src.dataloader import load_emnist

    device = get_device()
    train, _ = load_emnist()
    model = ExampleNet().to(device)

    model.forward(next(iter(train))[0].to(device))