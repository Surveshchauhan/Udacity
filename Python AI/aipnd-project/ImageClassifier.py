import warnings
from collections import OrderedDict

import PIL.Image as Img
import numpy as np
import torch
import torchvision.datasets as d
import torchvision.models as m
import torchvision.transforms as t
from torch import nn
from torch import optim


class ImageClassifier:

    def __init__(self, learn_rate=0.001, num_output=2, compute_device="gpu"):
        warnings.filterwarnings('ignore')

        self.__model = m.densenet121(True)

        self.__learn_rate = None
        self.__num_output = None
        self.__optimizer = None
        self.__model.classifier = None
        self.__criterion = nn.NLLLoss()

        self.__image_size = 255
        self.__image_crop = 224
        self.__image_mean = [0.485, 0.456, 0.406]
        self.__image_std = [0.229, 0.224, 0.225]

        if compute_device in ["cuda", "gpu"]:
            if torch.cuda.is_available():
                self.__device = torch.device("cuda:0")
            else:
                self.__device = torch.device("cpu")
        else:
            self.__device = torch.device("cpu")

        print('Model will be inferred using ' + str(self.__device))

        self.__crop_transform = t.Compose([t.Resize(self.__image_size), t.CenterCrop(self.__image_crop)])
        self.__tensor_transforms = t.Compose([self.__crop_transform,
                                              t.ToTensor(),
                                              t.Normalize(self.__image_mean, self.__image_std)])

        self.__initialize(learn_rate, num_output)

    def __initialize(self, learn_rate, num_output):
        self.__model = m.densenet121(True)
        self.__learn_rate = float(learn_rate)
        self.__num_output = int(num_output)

        for param in self.__model.parameters():
            param.requires_grad = False

        self.__model.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, 512)),
            ('relu1', nn.ReLU()),
            ('drop1', nn.Dropout(0.1)),
            ('fc2', nn.Linear(512, 256)),
            ('relu2', nn.ReLU()),
            ('drop2', nn.Dropout(0.1)),
            ('fc3', nn.Linear(256, self.__num_output)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.__optimizer = optim.Adam(self.__model.classifier.parameters(), lr=self.__learn_rate)

        self.__model = self.__model.to(self.__device)
        self.__model.eval()

    def train(self, data_dir, epochs, print_every):
        train_data = d.ImageFolder(data_dir, transform=self.__tensor_transforms)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

        self.__model.train()
        steps = 0

        for e in range(epochs):
            running_loss = 0
            for inputs, labels in iter(train_loader):
                steps += 1
                inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                self.__optimizer.zero_grad()

                outputs = self.__model.forward(inputs)
                loss = self.__criterion(outputs, labels)
                loss.backward()
                self.__optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e + 1, epochs),
                          "Loss: {:.4f}".format(running_loss / print_every))
                    running_loss = 0

        self.__model.model_dict = {y: x for x, y in train_data.class_to_idx.items()}
        self.__model.eval()

    def validate(self, data_dir):
        test_data = d.ImageFolder(data_dir, transform=self.__tensor_transforms)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(self.__device)
                labels = labels.to(self.__device)
                output = self.__model.forward(images)
                _, y_hat = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (y_hat == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    def predict(self, image_path, top_k=1):

        m_image: Img.Image = Img.open(image_path)

        t_image = self.__tensor_transforms(m_image)
        t_image = t_image.unsqueeze_(0).to(self.__device)

        out = self.__model.forward(t_image)
        data = out.cpu().data.numpy()
        top_predictions = sorted([x for x in enumerate(data[0])], key=lambda x: x[1], reverse=True)[:top_k]

        result = {
            'img': self.__crop_transform(m_image),
            'tags': [(np.exp(idx[1]), self.__model.model_dict[idx[0]]) for idx in top_predictions]
        }

        return result

    def save(self, path):
        state_dict = {
            'classifier': self.__model.classifier,
            'classifier_state': self.__model.classifier.state_dict(),
            'state_dict': self.__model.state_dict(),
            'model_dict': self.__model.model_dict,
            'learn_rate': self.__learn_rate,
            'num_output': self.__num_output
        }

        torch.save(state_dict, path)

    def load(self, path):
        state_dict: dict = torch.load(path)
        self.__initialize(state_dict['learn_rate'], state_dict['num_output'])
        self.__model.classifier = state_dict['classifier']
        self.__model.load_state_dict(state_dict['state_dict'])
        self.__model.classifier.load_state_dict(state_dict['classifier_state'])
        self.__model.model_dict = state_dict['model_dict']
