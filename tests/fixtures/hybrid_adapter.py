class HybridStateModel:
    def __init__(self, config):
        self.ssm = MambaBlock(config)
        self.recurrent = RecurrentLayer(config)

    def forward(self, input_ids, state=None):
        hidden_states, state = self.ssm(input_ids, state)
        return self.recurrent(hidden_states, state)
