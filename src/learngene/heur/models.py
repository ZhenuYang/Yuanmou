import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
#           Ancestor Model: Netwider (from utils/network_wider.py)
# =============================================================================
class Netwider(nn.Module):
    def __init__(self, num_layers):
        super(Netwider, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 5)
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def wider(self, layer_idx):
        conv_layer = self.layers[layer_idx]
        next_conv_layer = None
        for i in range(layer_idx + 1, len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                next_conv_layer = self.layers[i]
                break
        
        if next_conv_layer is None: return

        w1 = conv_layer.weight.data
        b1 = conv_layer.bias.data
        w2 = next_conv_layer.weight.data

        c_out, c_in, k, _ = w1.size()
        c_out_next, _, _, _ = w2.size()

        new_width = int(c_out * 1.5)
        
        new_w1 = torch.randn(new_width, c_in, k, k).to(w1.device) * 1e-3
        new_b1 = torch.zeros(new_width).to(b1.device)
        new_w2 = torch.randn(c_out_next, new_width, k, k).to(w2.device) * 1e-3

        rand_map = torch.randperm(c_out)
        
        for i in range(c_out):
            new_w1[i, :, :, :] = w1[i, :, :, :]
            new_b1[i] = b1[i]
            new_w2[:, i, :, :] = w2[:, i, :, :]

        for i in range(c_out, new_width):
            rand_idx = rand_map[i - c_out]
            new_w1[i, :, :, :] = w1[rand_idx, :, :, :]
            new_b1[i] = b1[rand_idx]
            new_w2[:, i, :, :] = w2[:, rand_idx, :, :] / 2
            new_w2[:, rand_idx, :, :] /= 2

        new_conv_layer = nn.Conv2d(c_in, new_width, kernel_size=k, padding=conv_layer.padding)
        new_conv_layer.weight.data = new_w1
        new_conv_layer.bias.data = new_b1
        self.layers[layer_idx] = new_conv_layer

        new_next_conv_layer = nn.Conv2d(new_width, c_out_next, kernel_size=k, padding=next_conv_layer.padding)
        new_next_conv_layer.weight.data = new_w2
        new_next_conv_layer.bias.data = next_conv_layer.bias.data
        
        for i in range(layer_idx + 1, len(self.layers)):
            if isinstance(self.layers[i], nn.Conv2d):
                self.layers[i] = new_next_conv_layer
                break

    def get_layers_19_20(self):
        # This is the "extraction" part, returning the specific layers
        # that constitute the "learngene".
        extracted_layers = nn.ModuleList()
        extracted_layers.append(self.layers[25])
        extracted_layers.append(self.layers[26])
        return extracted_layers

    def printf(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                print(f"Layer {i}: Conv2d({layer.in_channels}, {layer.out_channels}, kernel_size={layer.kernel_size})")
        print("Classifier:", self.classifier)


# =============================================================================
#           Descendant Model: vgg_compression_ONE (from utils/models/models.py)
# =============================================================================
class vgg_compression_ONE(nn.Module):
    def __init__(self, layers_to_inherit, linear_in_features, num_class=5):
        super(vgg_compression_ONE, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Inherit the "learngene" layers
        self.compression_features = layers_to_inherit

        self.classifier = nn.Sequential(
            nn.Linear(linear_in_features, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        for layer in self.compression_features:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def printf(self):
        print("Features:", self.features)
        print("Inherited Compression Features (Learngene):", self.compression_features)
        print("Classifier:", self.classifier)