class CustomLoaderForCausalLM:
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, config):
        self.model = object()

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights)
        for name, weight in weights:
            if "scale" in name:
                name = name.replace("scale", "kv_scale")
            param = getattr(self, name, None)
            if param is not None:
                param.weight_loader(param, weight)
        return loaded
