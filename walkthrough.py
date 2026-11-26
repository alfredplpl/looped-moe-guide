"""Show the execution order, routing choices and shared gradients on CPU."""

import torch

from moe import MoEConfig, MoELanguageModel


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = MoEConfig(
        vocab_size=64, context_length=16, layers=8, width=32, heads=4,
        experts=4, expert_width=64, loop_repeats=2,
    )
    model = MoELanguageModel(config)
    visits = []
    routes = []

    def record_block(index):
        def hook(_module, _inputs, _output):
            visits.append(index)
        return hook

    def record_route(layer, inputs):
        # The router uses FP32 linear algebra before taking the top expert scores.
        first_token = inputs[0][0, 0].float()
        logits = torch.nn.functional.linear(first_token, layer.router.weight.float())
        routes.append(logits.topk(config.experts_per_token).indices.tolist())

    handles = [block.register_forward_hook(record_block(index))
               for index, block in enumerate(model.blocks)]
    handles.append(model.blocks[2].experts.register_forward_pre_hook(record_route))
    sequence = torch.tensor([[3, 7, 11, 5, 9]])
    loss = model.training_loss(sequence[:, :-1], sequence[:, 1:])
    for handle in handles:
        handle.remove()

    loss.backward()
    shared_grad = model.blocks[2].experts.gate_up.grad.norm().item()
    assert visits == list(config.block_order) and len(routes) == 2 and shared_grad > 0
    total, active_uses = model.parameter_counts()
    print(f"Independent blocks: {len(model.blocks)}; block calls: {len(visits)}")
    print("Call order:", " → ".join(map(str, visits)))
    print("Block 2, first token, selected experts:", routes)
    print(f"Stored weights: {total:,}; active parameter applications per token: {active_uses:,}")
    print(f"Training loss: {loss.item():.4f}; shared expert gradient norm: {shared_grad:.6f}")


if __name__ == "__main__":
    main()
