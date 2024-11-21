from typing import Iterable, Tuple
import warnings
import torch
from deva.model.network import DEVA


class ImageFeatureStore:
    """
    A cache for image features.
    These features might be reused at different parts of the inference pipeline.
    This class provide a easy interface for reusing these features.
    It is the user's responsibility to delete features.

    Feature of a frame should be associated with a unique index -- typically the frame id.
    """
    
    def __init__(self, network: DEVA, no_warning: bool = False):
        self.network = network
        self._store = {}
        self._yolo_store = {}
        self.no_warning = no_warning
    
    def _encode_feature(self, index: int, image: torch.Tensor) -> None:
        ms_features, feat = self.network.encode_image(image)
        key, shrinkage, selection = self.network.transform_key(feat)
        
        self._store[index] = (ms_features, feat, key, shrinkage, selection)
    
    def push_yolo_features(self, index: int, yolo_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        self._yolo_store[index] = yolo_features
    
    def get_ms_features(self, index, image) -> Iterable[torch.Tensor]:
        if index not in self._store:
            self._encode_feature(index, image)
        
        return self._store[index][0]
    
    def get_yolo_features(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._yolo_store[index]
    
    def get_key(self, index, image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index not in self._store:
            self._encode_feature(index, image)
        
        return self._store[index][2:]
    
    def delete(self, index) -> None:
        if index in self._store:
            del self._store[index]
    
    def __len__(self):
        return len(self._store)
    
    def __del__(self):
        if len(self._store) > 0 and not self.no_warning:
            warnings.warn(f'Leaking {self._store.keys()} in the image feature store')
