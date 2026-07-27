def module_helper(x):
    return x


class OuterModel:
    class InnerBlock:
        def forward(self, hidden_states):
            return inner_runtime(hidden_states)

    def __init__(self, config):
        self.block = OuterModel.InnerBlock()

    def forward(self, input_ids):
        def local_helper(value):
            return should_not_be_counted(value)

        hidden_states = self.block.forward(input_ids)
        return local_helper(hidden_states)

    @property
    def hidden_size(self):
        return 1
