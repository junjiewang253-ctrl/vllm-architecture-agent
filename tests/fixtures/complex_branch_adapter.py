class BranchyForCausalLM:
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(self, config):
        self.attn = Attention(config)
        if config.use_moe:
            self.moe = FusedMoE(num_experts=config.num_experts)
        else:
            self.mlp = FeedForward(config)

    def forward(self, input_ids, inputs_embeds=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = embed(input_ids)
        match self.mode:
            case "fast":
                hidden_states = self.attn(hidden_states)
            case _:
                hidden_states = fallback(hidden_states)
        return hidden_states

    def load_weights(self, weights):
        for name, weight in weights:
            if "scale" in name:
                name = remap_scale_name(name)
            if is_pp_missing_parameter(name):
                continue
            param = lookup_parameter(name)
            if hasattr(param, "weight_loader"):
                param.weight_loader(param, weight)
            else:
                default_weight_loader(param, weight)
        return None
