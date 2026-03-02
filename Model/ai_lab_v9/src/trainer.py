# ============================================================
# PyTorch Trainer — Optimized for BiLSTM
# ============================================================

import torch
from torch import amp
import os
import pickle
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt

SAVE_DIR = "deploy"
os.makedirs(SAVE_DIR, exist_ok=True)

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def train_model(
    model,
    X,
    y,
    vocab=None,
    epochs=100,
    batch_size=64,
    accum_steps=2,
    lr=2e-3,
    save_name="model_v7.pt",
    resume=False,
    full_data=False,
    pos_weight=None,       # ← MỚI: class weight cho dữ liệu mất cân bằng
):
    """
    Huấn luyện model.

    Có 2 chế độ:
    ─────────────────────────────────────────────────────────────
    full_data=False  — 10-fold: split 90/10, early stopping theo val_AUC
    full_data=True   — Final model: train 100% data, chạy đủ epoch
    ─────────────────────────────────────────────────────────────

    pos_weight: float hoặc None
        Nếu dataset mất cân bằng (vd: 72% Real, 28% Fake), truyền
        pos_weight = n_neg/n_pos để BCEWithLogitsLoss phạt nặng hơn
        khi model đoán sai class thiểu số (Fake).
    """

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"

    print("Su dung thiet bi:", device)

    model.to(device)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    if full_data:
        print("📌 Chế độ FULL DATA — train trên 100% dữ liệu")

        train_dataset = TensorDataset(
            torch.tensor(X).long(),
            torch.tensor(y).float()
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = None

    else:
        split = int(0.9 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(
            torch.tensor(X_train).long(),
            torch.tensor(y_train).float()
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val).long(),
            torch.tensor(y_val).float()
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, pin_memory=True, num_workers=0
        )

    # --------------------------------------------------------
    # OPTIMIZER + SCHEDULER
    # --------------------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # ── Loss function — CÓ CLASS WEIGHT ─────────────────────────
    if pos_weight is not None and pos_weight != 1.0:
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  Loss: BCEWithLogitsLoss(pos_weight={pos_weight:.4f})")
    else:
        crit = torch.nn.BCEWithLogitsLoss()
        print(f"  Loss: BCEWithLogitsLoss (không weight)")

    scaler = amp.GradScaler(device_type, enabled=use_amp)

    # --------------------------------------------------------
    # CHECKPOINT + EARLY STOPPING
    # --------------------------------------------------------
    ckpt_path = f"{SAVE_DIR}/{save_name}"

    best_auc = 0
    best_epoch = 0
    patience = 10
    wait = 0

    if resume and os.path.exists(ckpt_path):
        print("Tiep tuc tu checkpoint...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\nBat dau huan luyen...\n")

    train_losses = []
    val_losses = []

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0
        all_probs = []
        all_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        opt.zero_grad()

        for step, (xb, yb) in enumerate(pbar):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with amp.autocast(device_type, enabled=use_amp):
                logits = model(xb)
                loss = crit(logits, yb)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(yb.cpu().numpy())
            pbar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(train_loader)
        epoch_auc = roc_auc_score(all_targets, all_probs)
        train_losses.append(epoch_loss)

        # ============================================
        # VALIDATION
        # ============================================
        if val_loader is not None:
            model.eval()
            val_loss_total = 0
            val_probs = []
            val_targets = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    val_loss_total += loss.item()
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                    val_targets.extend(yb.cpu().numpy())

            val_epoch_loss = val_loss_total / len(val_loader)
            val_auc = roc_auc_score(val_targets, val_probs)
            val_losses.append(val_epoch_loss)

            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | val_loss={val_epoch_loss:.4f} | train_AUC={epoch_auc:.4f} | val_AUC={val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                wait += 1
                if wait >= patience:
                    print("⏹ Early stopping kích hoạt.")
                    break
        else:
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | train_AUC={epoch_auc:.4f}")
            torch.save(model.state_dict(), ckpt_path)
            best_auc = epoch_auc
            best_epoch = epoch

    # --------------------------------------------------------
    # KẾT THÚC
    # --------------------------------------------------------
    print("\n✅ Huan luyen thanh cong")

    if val_loader is not None:
        print(f"   Best val_AUC: {best_auc:.4f} tại epoch {best_epoch}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"   Final train_AUC: {best_auc:.4f} sau {best_epoch} epochs")

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------
    if vocab is not None:
        with open(f"{SAVE_DIR}/vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    config = {
        "model_type": model.__class__.__name__,
        "vocab_size": model.emb.num_embeddings,
        "trainer": "v7",
        "best_epoch": best_epoch,
    }
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # Plot learning curve
    # --------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(f"{REPORT_DIR}/train_losses.json", "w") as f:
        json.dump(train_losses, f)
    if val_losses:
        with open(f"{REPORT_DIR}/val_losses.json", "w") as f:
            json.dump(val_losses, f)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    if val_losses:
        plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{REPORT_DIR}/learning_curve.png", dpi=300)
    plt.close()

    print("Learning curve exported to reports/")

    return best_auc
