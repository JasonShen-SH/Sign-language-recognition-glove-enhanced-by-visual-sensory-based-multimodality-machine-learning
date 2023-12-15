import torch
import torch.nn as nn

class Combined(nn.Module):
    def __init__(self, resnet_model, lstm_model1, lstm_model2, final_model):
        super(Combined, self).__init__()
        self.resnet_model = resnet_model
        self.lstm_model1 = lstm_model1
        self.lstm_model2 = lstm_model2
        self.final_model = final_model

    def forward(self, x1, x2, x3):
        resnet_output = self.resnet_model(x1)
        lstm_output1 = self.lstm_model1(x2)
        lstm_output2 = self.lstm_model2(x3)
        combined_output = torch.cat((resnet_output, lstm_output1, lstm_output2), dim=1)  

        final_output = self.final_model(combined_output)

        return final_output

    
class Combined_img(nn.Module):
    def __init__(self, resnet_model, final_model):
        super(Combined_img, self).__init__()
        self.resnet_model = resnet_model
        self.final_model = final_model

    def forward(self, x):
        resnet_output = self.resnet_model(x)
        combined_output = resnet_output

        final_output = self.final_model(combined_output)

        return final_output
    

class Combined_teng_imu(nn.Module):
    def __init__(self, lstm_model1, lstm_model2, final_model):
        super(Combined_teng_imu, self).__init__()
        self.lstm_model1 = lstm_model1
        self.lstm_model2 = lstm_model2
        self.final_model = final_model

    def forward(self, x1, x2):
        lstm_output1 = self.lstm_model1(x1)
        lstm_output2 = self.lstm_model2(x2)
        combined_output = torch.cat((lstm_output1, lstm_output2), dim=1)  

        final_output = self.final_model(combined_output)

        return final_output

    
class Combined_teng(nn.Module):
    def __init__(self, lstm_model, final_model):
        super(Combined_teng, self).__init__()
        self.lstm_model = lstm_model
        self.final_model = final_model

    def forward(self, x):
        lstm_output = self.lstm_model(x)
        combined_output = lstm_output

        final_output = self.final_model(combined_output)

        return final_output