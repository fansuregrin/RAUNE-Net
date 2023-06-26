import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


class SemanticContentLoss(nn.Module):
    """Semantic Content Loss.
    """
    def __init__(self, weights_k=(1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0)):
        """Initialize the loss.

        Args:
            weights_k: Five weights for summing.
        """
        super(SemanticContentLoss, self).__init__()
        self.weights_k = weights_k
        self.weights = VGG19_BN_Weights.DEFAULT
        model_ = vgg19_bn(weights=self.weights)
        _, self.eval_nodes = get_graph_node_names(model_)
        self.return_nodes = {
            'features.36': 'feature_1',
            'features.40': 'feature_2',
            'features.43': 'feature_3',
            'features.46': 'feature_4',
            'features.49': 'feature_5',
        }
        self.model = create_feature_extractor(model_, return_nodes=self.return_nodes)
        self.preprocess = self.weights.transforms()

    def forward(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x_features = self.model(x)
            y_features = self.model(y)
        loss = 0.0
        for i in range(5):
            loss += (self.weights_k[i] * F.l1_loss(x_features[f'feature_{i+1}'], y_features[f'feature_{i+1}']))
        return loss