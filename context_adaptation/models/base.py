import abc

class OnlineTraversabilityEstimator(abc.ABC):
    """
    Interface class for Traversability Estimator Inference models
    """
    def __init__(self):


    def get_required_tfs(self):
        """
        Get the tfs required in order to process data for this node
        (realistically, most nodes should leave this blank apart from ones that explicitly transform the data)
        """
        return {}

    @abc.abstractmethod
    def run(self, data_dict):
        """
        Execute this node on the data dict. This should read some data from the data dict and either modify it or add more data
        """
        pass

    @abc.abstractmethod
    def to(self, device):
        """run this node on device
        """
        pass