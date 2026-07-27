class MultiModalForConditionalGeneration(SupportsMultiModal):
    def __init__(self, config):
        self.vision_encoder = VisionEncoder(config.vision_config)
        self.text_model = TextDecoder(config.text_config)

    def forward(self, input_ids, image_pixels=None):
        image_features = self.vision_encoder(image_pixels)
        return self.text_model(input_ids, image_features)
