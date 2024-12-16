from ultralytics import YOLO


class EmbeddingExtractor:
    def __init__(self, model, layer_indices=[16, 19, 22]):
        """
        初始化嵌入提取器，默认提取 backbone 末尾的三个不同尺度的特征
        
        :param model: YOLO 模型实例
        :param layer_indices: 要注册的层索引列表
        """
        self.model = model
        self.layer_indices = layer_indices
        self.embeddings = {}  # 保存嵌入
        
        # 注册钩子到指定层
        self._register_hooks()
    
    def _hook_fn(self, module, input, output):
        """钩子函数，用于保存嵌入"""
        layer_name = module.name
        self.embeddings[layer_name] = output[0].detach().cpu()
    
    def _register_hooks(self):
        """注册钩子到指定层"""
        for i in self.layer_indices:
            layer_name = f"model.model.{i}"
            layer = dict(self.model.named_modules())[layer_name]
            layer.name = layer_name
            layer.register_forward_hook(self._hook_fn)
    
    def attach_to_results(self, results):
        """
        将嵌入附加到结果对象
        :param results: YOLO 结果对象
        """
        results.embeddings = self.embeddings
