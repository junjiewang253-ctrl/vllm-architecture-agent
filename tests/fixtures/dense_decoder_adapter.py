from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class DenseBlock:
    def __init__(self, config):
        self.attn = Attention(config)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, residual=None):
        hidden_states = self.attn(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DenseModel:
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = make_layers(config.num_hidden_layers, lambda idx: DenseBlock(config))
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states)
        return self.norm(hidden_states)


class DenseForCausalLM:
    def __init__(self, config):
        self.model = DenseModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        return hidden_states

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)
