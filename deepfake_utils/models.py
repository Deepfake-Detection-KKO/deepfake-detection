
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, vit_b_32, ViT_B_32_Weights

class MyModel(nn.Module):
    def __init__(self, model_type, device, num_classes=2, dropout_rate=0):
        """Initializes the MyModel class.

        If the model is pretrained, its parameters are frozen, and 
        the final classification layer is replaced with a new 
        sequential layer that includes dropout and a linear layer 
        for the specified number of classes.

        Parameters
        ----------
        model_type : str
            The type of pretrained model to use.
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') on which the model will be run.
        num_classes : int, optional
            The number of output classes for the final classification layer.
            Defaults to 2.
        dropout_rate : float, optional
            The dropout probability for the new classification layer.
        """
        super().__init__()
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.device = device
        self.num_classes=num_classes

        # Set model parameters based on model type
        if self.model_type == "ResNet-50-pretrained":
            # Initialize the Weight Transforms
            self.weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=self.weights)

            # Freeze parameters within ResNet
            for param in self.model.parameters():
                param.requires_grad = False

            # Replace last fully connected layer
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_ftrs, self.num_classes)
            )

        elif self.model_type == "ViT-b32-pretrained":
            # Set the weights
            self.weights = ViT_B_32_Weights.DEFAULT
            self.model = vit_b_32(weights=self.weights)

            # Freeze parameters within Vision Transformer
            for param in self.model.parameters():
                param.requires_grad = False

            # Replace last fully connected layer
            num_ftrs = self.model.heads.head.in_features
            self.model.heads = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_ftrs, self.num_classes)
            )

        self.model = self.model.to(self.device)

    def forward(self, x):
        outs = self.model(x)
        return outs