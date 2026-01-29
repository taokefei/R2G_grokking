import torch
import sys
# import wandb
import os
from transformers import AutoModelForCausalLM, AutoConfig
from torch.nn import functional as F
from dataset import ModData
import pandas as pd
import random

class QKVHook:
    def __init__(self, module):
        self.q, self.k, self.v = None, None, None
        self.handle = module.register_forward_hook(self._hook_fn)
    def _hook_fn(self, module, inputs, outputs):
        self.q = getattr(module, 'last_q', None)
        self.k = getattr(module, 'last_k', None)
        self.v = getattr(module, 'last_v', None)
    def clear(self):
        self.q, self.k, self.v = None, None, None
    def remove(self):
        self.handle.remove()


class AttentionMatrixHook:
    def __init__(self, model, save_dir="attention_matrices"):
        self.model = model
        self.attention_matrices = {}
        self.hooks = []
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._register_hooks()

    def _register_hooks(self):
        attn_module = self.model.transformer.h[0].attn
        def get_attention_hook(name):
            def hook(module, input, output):
                self.attention_matrices[name] = output[1].detach().cpu()#
            return hook
        self.hooks.append(attn_module.register_forward_hook(get_attention_hook("self_attention")))

    def save_matrices(self, epoch):
        if not self.attention_matrices:
            return

        save_path = f"{self.save_dir}/attention_epoch_{epoch}.pt"
        torch.save(self.attention_matrices, save_path)

    def clear(self):
        self.attention_matrices.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

for seed in [42]:#random.sample(range(1, 1001), 10):
    print(seed)
    mod_size = 500
    epochs = 18001
    group = "save_info"
    saves_dict = {
        8000: [50, 200, 300, 500, 750, 1000, 1700, 2000, 4800, 7500, 14000, 18000],
        25975: [50, 200, 300, 500, 750, 1000, 1700, 2000, 4800, 7500, 14000, 18000],
        40000: [50, 200, 300, 500, 750, 1000, 1700, 2000, 4800, 7500, 14000, 18000]
    }
    save_dir = "TEST_checkpoints/"

    device = "cuda:0"
    print_seq = 200

    config = AutoConfig.from_pretrained("one-layer-openai-gpt")
    n_embd = config.n_embd
    val_size = 2000
    torch.manual_seed(seed)
    diff_symbol = False
    symbol = "+"

    args = sys.argv[1:]
    if len(args) == 0:
        train_size = 2000
    elif len(args) == 1:
        train_size = int(args[0])  # python transformer.py 8000
    elif len(args) == 2:
        train_size = int(args[1])
        device = "cuda:" + args[0]
    elif len(args) == 3:
        train_size = int(args[2])
        device = "cuda:" + args[1]
        n_embd = int(args[0])
    elif len(args) == 4:
        train_size = int(args[3])
        device = "cuda:" + args[2]
        n_embd = int(args[1])
        diff_symbol = bool(args[0])
    else:
        train_size = int(args[4])
        device = "cuda:" + args[3]
        n_embd = int(args[2])
        diff_symbol = bool(args[1])
        epochs = int(args[0])

    saves = saves_dict[train_size]

    lr = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()

    config = AutoConfig.from_pretrained("one-layer-openai-gpt")
    config.vocab_size = mod_size + 5
    config.n_embd = n_embd
    config.loss_fn = torch.nn.CrossEntropyLoss()

    info = f"seed{seed}@{train_size}@{config.n_embd}&{symbol}"

    save_prefix = f"{save_dir}{info}/"
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)

    model = AutoModelForCausalLM.from_config(config).to(device)
    attention_hook = AttentionMatrixHook(model, save_dir=f"{save_dir}attention_matrices/{info}/")
    attn_module = model.transformer.h[0].attn
    qkv_hook = QKVHook(attn_module)


    optm = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    dataset = ModData(mod=mod_size)
    all_inputs = dataset.get_inputs()
    all_labels = dataset.get_labels()
    indices = torch.randperm(all_inputs.shape[0])
    train_indices = indices[:train_size]
    train_inputs = all_inputs[train_indices].to(device)
    train_labels = all_labels[train_indices].to(device)


    if diff_symbol:
        symbol = "-"
        val_dataset = ModData(mod=val_mod, symbol="-")
        val_inputs = val_dataset.get_inputs()[:val_size].to(device)
        val_labels = val_dataset.get_labels()[:val_size].to(device)
    else:
        val_indices = indices[train_size: train_size + val_size]
        val_inputs = all_inputs[val_indices].to(device)
        val_labels = all_labels[val_indices].to(device)

        val_inputs_cpu = val_inputs.cpu().detach().numpy()

    for epoch in range(epochs):
        model.train()
        total = train_size
        outputs = model(train_inputs, labels=train_labels, output_attentions=True)

        optm.zero_grad()

        train_probs = F.softmax(outputs.logits[:, -2, :], dim=-1)
        train_preds = torch.multinomial(train_probs, 1).flatten()
        correct = (train_preds == train_labels[:, -1]).sum()

        outputs.loss.backward()
        total_loss = outputs.loss.item()
        optm.step()

        train_acc = correct / total
        train_loss = total_loss

        q_train = qkv_hook.q
        k_train = qkv_hook.k
        v_train = qkv_hook.v
        qkv_hook.clear()

        model.eval()
        with torch.no_grad():
            outputs = model(val_inputs, labels=val_labels, output_attentions=True)
            val_loss = outputs.loss.item()
            val_probs = F.softmax(outputs.logits[:,-2,:], dim=-1)
            val_preds = torch.multinomial(val_probs, 1).flatten()
            correct = (val_preds == val_labels[:, -1]).sum()
            total = val_labels.shape[0]

            val_acc = correct / total

            q_val = qkv_hook.q
            k_val = qkv_hook.k
            v_val = qkv_hook.v
            qkv_hook.clear()


        if epoch in saves:
            checkpoint = {
                "epoch": epoch,
                # "state_dict": model.state_dict(),
                # "optimizer": optm.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                # "train_inputs": train_inputs.cpu(),
                # "val_inputs": val_inputs.cpu(),
            }
            save_path = f"{save_prefix}{epoch}.pt"
            torch.save(checkpoint, save_path)

            torch.save(q_train, f"{save_prefix}train_query_{epoch}.pt")
            torch.save(k_train, f"{save_prefix}train_key_{epoch}.pt")
            torch.save(v_train, f"{save_prefix}train_value_{epoch}.pt")
            torch.save(q_val, f"{save_prefix}val_query_{epoch}.pt")
            torch.save(k_val, f"{save_prefix}val_key_{epoch}.pt")
            torch.save(v_val, f"{save_prefix}val_value_{epoch}.pt")

            attention_hook.save_matrices(epoch)
            attention_hook.clear()

        if (epoch + 1)  % print_seq == 0:
            epoch_info = f"Info: {info}, Epoch:{epoch + 1},"
            train_info = f"Train Loss: {train_loss:.4f}, Train Acc:{train_acc:.4f},"
            val_info = f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            print(f"{epoch_info}{train_info}{val_info}")

    attention_hook.remove_hooks()
    qkv_hook.remove()
