# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class ViewBlock(nn.Module):
    def __init__(self, code, input_feature_num, output_feature_num):
        super(ViewBlock, self).__init__()
        self.code = code
        self.fc_extract_comm = nn.Linear(input_feature_num, output_feature_num)
        self.fc_private = nn.Linear(input_feature_num, output_feature_num)

    def forward(self, input):
        x_private = F.relu(self.fc_private(input),)
        x_comm_feature = F.relu(self.fc_extract_comm(input),)
        return x_private, x_comm_feature

class EncoderBlock(nn.Module):
    def __init__(self, input_feature_num, output_feature_num):
        super(EncoderBlock, self).__init__()
        # self.code = code
        # self.fc_extract_comm = nn.Linear(input_feature_num, output_feature_num)
        self.fc_private = nn.Linear(input_feature_num, output_feature_num)

    def forward(self, input):
        x_private = F.relu(self.fc_private(input),)
        # x_comm_feature = F.relu(self.fc_extract_comm(input),)
        return x_private

class DecoderBlock(nn.Module):
    def __init__(self, code, hidden_feature_num, output_feature_num):
        super(DecoderBlock, self).__init__()
        self.code = code
        self.fc_private = nn.Linear(hidden_feature_num, output_feature_num)

    def forward(self, input):
        x_private = F.relu(self.fc_private(input),)
        return x_private

class LabelEmbedding(nn.Module):
    def __init__(self, input_label_num, hidden_label_num, output_label_num):
        super(LabelEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_label_num, hidden_label_num)
        self.fc2 = nn.Linear(hidden_label_num, hidden_label_num)
        self.fc = nn.Linear(hidden_label_num, output_label_num)

    def forward(self, labels):
        outputs = F.relu(self.fc1(labels))
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc(outputs)

        return outputs

