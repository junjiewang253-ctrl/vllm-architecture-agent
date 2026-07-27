class MoeBlock:
    def __init__(self, config):
        self.gate = GateLinear(config.hidden_size, config.num_experts)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            enable_eplb=True,
        )

    def forward(self, hidden_states):
        router_logits = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, router_logits)
        return hidden_states


class MoeForCausalLM(MixtureOfExperts, SupportsPP):
    def __init__(self, config):
        self.layers = make_layers(config.num_hidden_layers, lambda idx: MoeBlock(config))
        self.ep_group = get_ep_group()

    def forward(self, input_ids):
        hidden_states = input_ids
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
