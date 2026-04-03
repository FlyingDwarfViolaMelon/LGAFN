import torch
import torch.nn as nn


class FusionLayer2(nn.Module):

    def __init__(self, num_views=2, fusion_type='weighted'):
        """
        :param fusion_type: include concatenate/average
        """
        super(FusionLayer2, self).__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        # define the attention weights for feature matrix and adjacent matrix
        self.pai_fea = nn.Parameter(torch.ones(self.num_views) / self.num_views, requires_grad=True)

    def forward(self, feature1, feature2):
        if self.fusion_type == "concatenate":
            combined_feature = torch.cat((feature1, feature2), dim=1)
        elif self.fusion_type == "weighted":
            # combine the feature matrix
            exp_sum_pai_fea = 0
            for i in range(self.num_views):
                exp_sum_pai_fea += torch.exp(self.pai_fea[i])
            combined_feature = (torch.exp(self.pai_fea[0]) / exp_sum_pai_fea) * feature1
            for i in range(1, self.num_views):
                combined_feature = combined_feature + (torch.exp(self.pai_fea[i]) / exp_sum_pai_fea) * feature2
        else:
            print("Please using a correct fusion type")
            exit()
        return combined_feature