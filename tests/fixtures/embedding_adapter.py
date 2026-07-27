class TextEmbeddingModel:
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.pooler = PoolingHead(config.hidden_size)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def pooling(self, hidden_states):
        return self.pooler(hidden_states)
