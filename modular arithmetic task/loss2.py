import torch
import sys
# import wandb
import os
import random
from transformers import AutoModelForCausalLM, AutoConfig
from torch.nn import functional as F
from dataset import ModData
import pandas as pd


#############
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


def load_tensor(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"not file: {file_path}")
    tensor = torch.load(file_path)
    if isinstance(tensor, dict):
        tensor = list(tensor.values())[0]
    return tensor


def calculate_distances(points):
    dist_matrix = torch.cdist(points, points, p=2.0) ** 2
    return dist_matrix


def calculate_avg_and_max_distances(points):
    dist_matrix = calculate_distances(points)
    upper_triangle = dist_matrix.triu(diagonal=1)
    avg_distance = torch.mean(upper_triangle[upper_triangle > 0]).sqrt().item()
    max_distance = torch.max(upper_triangle).sqrt().item()
    return avg_distance, max_distance

def get_max_main(train_size, pos):
    dir = fr"E:/grokking/checkpoints/{train_size}@48&+"

    files = sorted([f for f in os.listdir(dir) if f.endswith('.pt')])

    epoch = 18000

    q_files = [f for f in files if f'_query_{epoch}.pt' in f]
    k_files = [f for f in files if f'_key_{epoch}.pt' in f]
    v_files = [f for f in files if f'_value_{epoch}.pt' in f]

    epoch_file = [f for f in files if f == f"{epoch}.pt"]
    qkv_path = os.path.join(dir, q_files[1])
    ckpt_path = os.path.join(dir, epoch_file[0])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    val_inputs = ckpt["val_inputs"]
    v = load_tensor(os.path.join(dir, v_files[1]))

    pos_val = v[:, pos, :]

    avg_distance, max_distance = calculate_avg_and_max_distances(pos_val)


    return avg_distance, max_distance
for seed in [42]:#random.sample(range(1, 1001), 10):
    print(seed)
    mod_size = 500
    epochs = 18001
    group = "save_info"
    save_dir = "TEST_loss2_checkpoints/"

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


    alpha = 0.3

    for epoch in range(epochs):
        model.train()
        total = train_size
        outputs = model(train_inputs, labels=train_labels, output_attentions=True)

        q_train = qkv_hook.q#
        k_train = qkv_hook.k#
        v_train = qkv_hook.v#
        qkv_hook.clear()#


        pos0_train = v_train[:, 0, :]
        pos2_train = v_train[:, 2, :]
        pos6_train = v_train[:, 6, :]

        total_d_train = 0

        random_indices_train = torch.randperm(pos0_train.size(0))
        dis0_train = torch.norm(pos0_train - pos0_train[random_indices_train], dim=1)
        dis2_train = torch.norm(pos2_train - pos2_train[random_indices_train], dim=1)
        dis6_train = torch.norm(pos6_train - pos6_train[random_indices_train], dim=1)

        total_d_train += torch.sum(dis0_train) + torch.sum(dis2_train) + torch.sum(dis6_train)

        K = 0

        for _ in range(K):
            random1 = torch.randperm(pos0_train.size(0), device=pos0_train.device)
            random2 = torch.randperm(pos0_train.size(0), device=pos0_train.device)
            dis0_train = torch.norm(pos0_train[random1] - pos0_train[random2], dim=1)
            dis2_train = torch.norm(pos2_train[random1] - pos2_train[random2], dim=1)

            total_d_train += torch.sum(dis0_train) + torch.sum(dis2_train) + torch.sum(dis6_train)

        d_train = total_d_train / (1+K)

        dis_loss_train = 10000 / d_train


        optm.zero_grad()

        train_probs = F.softmax(outputs.logits[:, -2, :], dim=-1)
        train_preds = torch.multinomial(train_probs, 1).flatten()
        correct = (train_preds == train_labels[:, -1]).sum()

        train_loss = outputs.loss

        final_loss_train = train_loss + alpha * dis_loss_train
        final_loss_train.backward()

        total_loss_train = final_loss_train.item()
        optm.step()


        train_acc = correct / total


        model.eval()
        with torch.no_grad():
            outputs = model(val_inputs, labels=val_labels, output_attentions=True)
            q_val = qkv_hook.q
            k_val = qkv_hook.k
            v_val = qkv_hook.v
            qkv_hook.clear()

            pos0_val = v_val[:, 0, :]
            pos2_val = v_val[:, 2, :]
            pos6_val = v_val[:, 6, :]
            total_d_val = 0

            random_indices_val = torch.randperm(pos0_val.size(0))  # size(0) is the number of rows.
            dis0_val = torch.norm(pos0_val - pos0_val[random_indices_val], dim=1)
            dis2_val = torch.norm(pos2_val - pos2_val[random_indices_val], dim=1)
            dis6_val = torch.norm(pos6_val - pos6_val[random_indices_val], dim=1)

            total_d_val += torch.sum(dis0_val).item() + torch.sum(dis2_val).item() + torch.sum(dis6_val).item()

            K_val = 0
            for _ in range(K_val):
                random1 = torch.randperm(pos0_val.size(0), device=pos0_val.device)
                random2 = torch.randperm(pos0_val.size(0), device=pos0_val.device)
                dis0_val = torch.norm(pos0_val[random1] - pos0_val[random2], dim=1)
                dis2_val = torch.norm(pos2_val[random1] - pos2_val[random2], dim=1)
                dis6_val = torch.norm(pos6_val[random1] - pos6_val[random2], dim=1)

                total_d_val += torch.sum(dis0_val).item() + torch.sum(dis2_val).item() + torch.sum(dis6_val).item()

            d_val = total_d_val / (1+K_val)
            dis_loss_val = 10000 / d_val

            device = outputs.loss.device
            dis_loss_val = torch.tensor(dis_loss_val, device=device, dtype=outputs.loss.dtype)

            val_loss = outputs.loss
            final_loss_val = val_loss + alpha * dis_loss_val
            total_loss_val = final_loss_val.item()

            val_probs = F.softmax(outputs.logits[:,-2,:], dim=-1)
            val_preds = torch.multinomial(val_probs, 1).flatten()
            if epoch==18000:
                df = pd.DataFrame(val_preds.cpu().numpy())
                df.to_excel('loss2_val_preds_output.xlsx', index=False, header=False)
                print("success")
            correct = (val_preds == val_labels[:, -1]).sum()
            total = val_labels.shape[0]

            val_acc = correct / total


        checkpoint = {
            "epoch": epoch,
            # "state_dict": model.state_dict(),
            # "optimizer": optm.state_dict(),
            "train_loss": train_loss.item(),
            "dis_train_loss": dis_loss_train.item(),
            "total_train_loss": total_loss_train,
            "train_acc": train_acc,
            "val_loss": val_loss.item(),
            "dis_val_loss": dis_loss_val.item(),
            "total_val_loss": total_loss_val,
            "val_acc": val_acc,
            # "train_inputs": train_inputs.cpu(),
            # "val_inputs": val_inputs.cpu(),
        }
        save_path = f"{save_prefix}{epoch}.pt"
        torch.save(checkpoint, save_path)


        if (epoch + 1)  % print_seq == 0:
            # epoch_info = f"Size: {data_size}, Epoch:{epoch + 1}/{epochs},"
            epoch_info = f"Info: {info}, Epoch:{epoch + 1},"
            train_info = f"total Train Loss: {total_loss_train:.4f}, dis Train Loss: {dis_loss_train.item():.4f}, Train Acc:{train_acc:.4f},"
            val_info = f"total Val Loss: {total_loss_val:.4f}, dis Val Loss: {dis_loss_val.item():.4f}, Val Acc: {val_acc:.4f}"
            print(f"{epoch_info}{train_info}{val_info}")

    attention_hook.remove_hooks()
    qkv_hook.remove()



