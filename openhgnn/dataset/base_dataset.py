from abc import ABC, ABCMeta, abstractmethod

class BaseDataset(ABC):
    def __init__(self, ):
        super(BaseDataset, self).__init__()
        self.meta_paths = None
        self.meta_paths_dict = None

