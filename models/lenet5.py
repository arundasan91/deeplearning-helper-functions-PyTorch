class LeNet5(nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given 
        # as input. Hence padding of 2 is done below)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
                                     kernel_size=5, stride=1, padding=2)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, 
                                     kernel_size=5, stride=1, padding=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, 
                                     kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(7*7*120, 120)
        # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(120, 84)       
        # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = nn.Linear(84, 10)        
        # convert matrix with 84 features to a matrix of 10 features (columns)

            
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = F.relu(self.conv1(x))  
        # max-pooling with 2x2 grid 
        x = self.max_pool_1(x) 
        # Conv2 + ReLU
        x = F.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # Conv3 + ReLU
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7*7*120)
        # FC-1, then perform ReLU non-linearity
        x = F.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = F.relu(self.fc2(x))
        # FC-3
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)
