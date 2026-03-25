# ============================================================
# PyTorch Trainer — Optimized for BiLSTM
# ============================================================

import torch #Thư viện deep learning chính dùng để xây dựng và train model.
from torch import amp #amp = Automatic Mixed Precision: train nhanh hơn, tiết kiệm VRAM GPU
import os
import pickle #lưu vocab.pkl
import json #Lưu cấu hình model dạng JSON.
from torch.utils.data import DataLoader, TensorDataset #TensorDataset: Dùng để tạo dataset từ tensor vd X = tensor sentences, y = labels
                                                    #DataLoader: chia batch, shuffle, load dữ liệu
from sklearn.metrics import roc_auc_score #Tính AUC score: chỉ số đo lường khả năng phân biệt giữa các class của mô hình phân loại
from tqdm import tqdm #Thư viện hiển thị progress bar
                        #Epoch 1/100
                        #█████████████ 60%
import matplotlib.pyplot as plt #Dùng để vẽ learning curve: là biểu đồ thể hiện sự thay đổi hiệu suất của mô hình Machine Learning/Deep Learning (thường qua loss hoặc accuracy)

SAVE_DIR = "deploy"
os.makedirs(SAVE_DIR, exist_ok=True) #tạo thư mục lưu model chứa model, config, vocab

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True) #tạo thư mục chứa Report chứa learning curve, loss history


def train_model(
    model,                   #loại model
    X,                       #câu được mã hóa X(Batch_size, seq_len)
    y,
    vocab=None,
    epochs=100,              #Số lần train
    batch_size=64,           #Mỗi lần xử lý 64 sample
    accum_steps=2,           #Gradient accumulation: 2 batch mới update 1 lần
    lr=2e-3,                 #Learning rate = 0.002
    save_name="model_v7.pt", #Tên file model
    resume=False,            #Nếu True → load checkpoint để train tiếp
    full_data=False,         #False → train + validation ||True → train full dataset
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Nếu có GPU → dùng GPU.
    device_type = "cuda" if device.type == "cuda" else "cpu"    #Lấy loại device.
    use_amp = device.type == "cuda"       #AMP chỉ dùng khi có GPU

    print("Su dung thiet bi:", device)

    model.to(device) #Đưa model lên GPU.

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    if full_data:
        print("📌 Chế độ FULL DATA — train trên 100% dữ liệu")

        train_dataset = TensorDataset(
            torch.tensor(X).long(), #Tạo tensor(mảng đa chiều) kiểu số nguyên long() vì embedding cần int
            torch.tensor(y).float() #BCEWithLogitsLoss cần float
        )
        # 0 → (X[0], y[0])
        # 1 → (X[1], y[1])
        # 2 → (X[2], y[2])

        train_loader = DataLoader( #dataset -> chia batch -> đưa vào model
            train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=0 #shuffle: trộn, pin_memory: Tối ưu khi dùng GPU(RAM → GPU) pin_memory giúp copy nhanh hơn.
        )                                                #Số CPU thread load data.
        val_loader = None #trường hợp này không cần valid

    else:
        split = int(0.9 * len(X))               #Lấy 90% dữ liệu để train, 10% để validation
        X_train, X_val = X[:split], X[split:]   #
        y_train, y_val = y[:split], y[split:]

        train_dataset = TensorDataset(          #Tạo dataset train
            torch.tensor(X_train).long(),
            torch.tensor(y_train).float()
        )
        val_dataset = TensorDataset(            #Tạo dataset validation
            torch.tensor(X_val).long(),
            torch.tensor(y_val).float()
        )

        train_loader = DataLoader(           #dataset -> chia batch -> đưa vào model
            train_dataset, batch_size=batch_size,
            shuffle=True, pin_memory=True, num_workers=0
        )
        val_loader = DataLoader(            #Validation không cần trộn dữ liệu vì không train chỉ đánh giá model
            val_dataset, batch_size=batch_size,
            shuffle=False, pin_memory=True, num_workers=0
        )

    # --------------------------------------------------------
    # OPTIMIZER + SCHEDULER
    # --------------------------------------------------------

    #thiết lập cách model học (optimizer)
    #và cách thay đổi learning rate trong quá trình train (scheduler)

    #Optimizer là thuật toán dùng để cập nhật trọng số của model sau mỗi lần tính gradient.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        #torch.optim.AdamW: AdamW là phiên bản cải tiến của Adam
        #Điểm mạnh: hội tụ nhanh, ổn định, phù hợp model deep learning

        #model.parameters(): Lấy tất cả trọng số của model.
        #Ví dụ model có:Embedding weightsm, LSTM weights, Linear weights. Optimizer sẽ cập nhật tất cả các weight này.

        #lr = 0.002 => công thức update: weight = weight - 0.002 * gradient

        #weight_decay=1e-4 Đây là L2 regularization. Nó làm weight nhỏ lại để tránh overfitting.
        #Công thức: loss_total = loss + λ * ||weight||² ở đây λ = 0.0001

    #Scheduler dùng để tự động thay đổi learning rate theo từng step khi train.
    scheduler = torch.optim.lr_scheduler.OneCycleLR( #Scheduler này tên: OneCycleLR. Nó làm learning rate chạy theo hình chu kỳ.
                            #LR
                            # │
                            # │        /\       (tăng dần)
                            # │       /  \____  (giảm dần)
                            # │
                            # └────────────────────────
                            #           training steps

        opt, max_lr=lr,   #Scheduler sẽ điều chỉnh learning rate của optimizer này.
                            #max_lr=lr Learning rate lớn nhất. ví dụ lr = 0.002. Learning rate sẽ chạy kiểu: 0.0002 → 0.002 → 0.00002
                            # | giai đoạn | LR      |
                            # | --------- | ------- |
                            # | bắt đầu   | nhỏ     |
                            # | tăng      | dần     |
                            # | đỉnh      | `0.002` |
                            # | giảm      | dần     |
                            # | cuối      | cực nhỏ |

        epochs=epochs,    #số epochs train, Scheduler cần biết tổng thời gian training.
        steps_per_epoch=len(train_loader),   #Một epoch có bao nhiêu batch = tổng số mẫu/ batch_size
                                            # vd train_loader là một iterator(một đối tượng cho phép duyệt qua các phần tử của một tập hợp (như list, tuple, string)) chứa các batch
                                            # train_loader[
                                            #                 batch1,
                                            #                 batch2,
                                            #                 batch3,
                                            #                ...
                                            #                 batch100
                                            #             ]
                                            #len(train_loader) = 100

        pct_start=0.1,              #10% step đầu → tăng learning rate, đạt đỉnh ở step thứ 10%, 90% step sau → giảm learning rate
                                    #vd: total_step = 1000
                                    # | step | LR              | mỗi step là 1 lần train trên batch
                                    # | ---- | --------------- | gồm 64 sample và có thể ít hơn
                                    # | 0    | 0.0002          | do được làm tròn ở step cuối
                                    # | 50   | 0.0008          |
                                    # | 75   | 0.0015          |
                                    # | 100  | **0.002 (max)** |
                                    # | 500  | 0.001           |
                                    # | 750  | 0.0002          |
                                    # | 1000 | 0.00002         |

                                    #vd
                                    #epochs = 10
                                    #dataset = 10000 samples
                                    #batch_size = 100
                                    #batch = steps_per_epoch = 10000/100 = 100 //1 step = 1 batch mang đi train
                                    #total_step = epochs x steps_per_epoch = 100x10 = 1000
                                    #qua mỗi step LR sẽ thay đổi, tổng là 1000 lần thay đổi khi training 1000 batch
                                    # step 0      → 0.0002
                                    # step 50     → 0.001
                                    # step 100    → 0.002
                                    # step 300    → 0.0015
                                    # step 700    → 0.0004
                                    # step 1000   → 0.00002

        anneal_strategy='cos',      #Cách giảm learning rate. cos = cosine decay, Giảm mượt hơn so với linear.
    )

    # ── Loss function — CÓ CLASS WEIGHT ─────────────────────────
    if pos_weight is not None and pos_weight != 1.0:
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        print(f"  Loss: BCEWithLogitsLoss(pos_weight={pos_weight:.4f})")
    else:
        crit = torch.nn.BCEWithLogitsLoss() #BCEWithLogitsLoss: 2 trong 1: Sigmoid, Binary Cross Entropy
        print(f"  Loss: BCEWithLogitsLoss (không weight)")

    scaler = amp.GradScaler(device_type, enabled=use_amp) #GradScaler là công cụ giúp training bằng FP16 (Mixed Precision) không bị lỗi gradient quá nhỏ
                                                        #Nó phóng to loss lên trước khi backpropagation, để tránh gradient bị = 0.
                                                        #Loss → Backprop → Gradient → Optimizer update weight
                                                        #gradient = 0.00000012 FP16 không đủ độ chính xác, nên nó thành: 0
                                                        #GradScaler giải quyết thế nào?
                                                        # Bước 1: nhân loss lên
                                                        # Ví dụ:
                                                        # loss = 0.0002
                                                        # GradScaler làm:
                                                        # loss_scaled = loss × 1024 (gradient = dLoss / dWeight)
                                                        # Bước 2: backprop
                                                        # loss_scaled.backward() -> tính gradient
                                                        # Gradient lúc này không bị quá nhỏ.
                                                        # Bước 3: trước khi update weight
                                                        # PyTorch tự chia lại cho 1024 để đúng giá trị thật.

    # --------------------------------------------------------
    # CHECKPOINT + EARLY STOPPING
    # --------------------------------------------------------
    ckpt_path = f"{SAVE_DIR}/{save_name}"

    best_auc = 0        #AUC tốt nhất đã đạt được
    best_epoch = 0      #Lưu epoch tốt nhất
    patience = 10       #Đây là Early Stopping parameter(thông số dừng sớm).
                        #Nếu 10 epoch liên tiếp model không cải thiện -> dừng training
    wait = 0            #Đếm bao nhiêu epoch chưa cải thiện

    if resume and os.path.exists(ckpt_path): #nếu resume == true thì train tiếp từ model cũ
        print("Tiep tuc tu checkpoint...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\nBat dau huan luyen...\n")

    train_losses = [] #lưu loss mỗi epoch của train
    val_losses = []   #loss mỗi epoch của validation

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------
    for epoch in range(1, epochs + 1):

        model.train() #chuyển sang chế độ train
        total_loss = 0 #cộng loss cả epoch
        all_probs = [] #lưu prediction
        all_targets = [] #lưu label thật

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") #tạo progress bar khi train.
        opt.zero_grad()  #reset gradient, nếu không thì sẽ cộng dồn từ epoch trước->sai

        #forward
        for step, (xb, yb) in enumerate(pbar):  #số step = số bacth = tổng sample/batch_size
            xb = xb.to(device, non_blocking=True) #batch dữ liệu x
            yb = yb.to(device, non_blocking=True) #batch dữ liệu y
                                                #device: Chuyển tensor sang thiết bị tính toán
                                                #non_blocking=True: cho phép chạy song song
                                                #bình thường:CPU copy → GPU, training phải đợi copy xong
                                                #CPU copy → GPU (song song), GPU có thể tính batch trước-> làm tăng tốc trainning
                                                #chỉ hiệu quả khi có DataLoader(pin_memory=True) phía trên

            with amp.autocast(device_type, enabled=use_amp): #bật amp, autocast quyết định layer nào dùng float16(nhanh, ít vram), layer nào dùng float32(chậm, chính xác)
                logits = model(xb)      #Output trước sigmoid
                loss = crit(logits, yb) #loss function, gom chung sigmoid, binary cross entropy trả về loss trung bình của cả batch
                loss = loss / accum_steps #Gradient Accumulation: 1 step chỉ xử lý 1 batch 64 mẫu
                                          #Nhưng ta muốn đến mẫu 128 mới cập nhật weight
                                          #gradient = (g1)/2 + (g2)/2
                                          #loss1 = (l1+l2+..+l64)/64
                                          #loss = loss / 2
                                          #loss1 = (l1+l2+..+l64)/128
                                          #grad1 = (g1+g2+..+g64)/128
                                          #grad1 = (g1+g2+..+g64)/128
                                          #Train 2 step nhưng gradient chỉ bằng 1 step

            #backward: để tránh gradient bị quá nhỏ (underflow) khi dùng float16.
            scaler.scale(loss).backward() #Thực hiện backpropagation, tạo ra gradient nhưng trước đó nhân loss lên một hệ số lớn (scale).
                                        #Sau này PyTorch sẽ chia lại cho 1024 khi update weight.
                                        #(loss * scale).backward() Nhưng PyTorch tự động quản lý scale factor.
                                        #Mục đích để lừa gradient chứ không ảnh hưởng tới loss thật


            if (step + 1) % accum_steps == 0: #Chỉ update weight sau 2 batch.
                scaler.unscale_(opt)            #Chia gradient lại cho scale factor.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Giới hạn độ lớn của gradient về 1.0
                                                                        #Gradient norm là gì
                                                                        #Gradient vector:
                                                                        #g = [g1, g2, g3, ...]
                                                                        #Norm:
                                                                        #||g|| = sqrt(g1² + g2² + ...)
                                                                        #Nếu gradient quá lớn ||g|| = 50
                                                                        #Weight update: w = w - lr * 50 → model nổ gradient.
                                                                        #||g|| > 1
                                                                        #g = g × (1 / ||g||)
                scaler.step(opt) #update weight nhưng có kiểm tra gradient có NaN/Inf không.
                                    #Nếu gradient bình thường: opt.step(): w = w - lr * gradient
                                    #Nếu gradient bị lỗi: PyTorch bỏ qua step này, Không update weight.
                scaler.update() #Điều chỉnh scale factor, nếu train ổn định thì tăng scale để tận dụng float16 tối đa
                                #Nếu gradient overflow:gradient = inf -> scale giảm: scale = scale / 2
                opt.zero_grad() #reset gradient
                scheduler.step() #Sau mỗi 2 batch thì update lr sẽ thay đổi theo OneCycleLR

            total_loss += loss.item() #loss.item(): chuyển từ tensor(0.4321, device='cuda:0') thành số
            probs = torch.sigmoid(logits).detach().cpu().numpy() #biến output của model thành xác suất để tính AUC
                                                                #.detach() tách tensor ra khỏi computational graph (đồ thị tính gradient)
                                                                    #Computational graph là gì? Khi dùng PyTorch để train model, mọi phép toán sẽ được lưu vào graph để sau này tính gradient.
                                                                    #gradient không tính về probs vì graph đã bị cắt.
                                                                #.cpu() tensor đang ở  GPU, numpy chỉ chạy trên CPU
                                                                #.numpy() Chuyển tensor → numpy array
            all_probs.extend(probs)         #Lưu tất cả prediction của epoch
            all_targets.extend(yb.cpu().numpy()) #Lưu nhãn nhãn thật của batch convert yb = tensor([0,1,1]) -> [0,1,1]
            pbar.set_postfix(loss=loss.item())  #set_postfix() giúp hiển thị thêm thông tin.
                                                #loss=loss.item() hiển thị loss của batch hiện tại.
                                                #Epoch 1/10:  45%|█████████      | loss=0.432

        epoch_loss = total_loss / len(train_loader)  #Tính loss của 1 epoch = trung bình cộng loss của các batch
        epoch_auc = roc_auc_score(all_targets, all_probs)   #roc_auc_score sẽ đánh giá model phân biệt 0 và 1 tốt tới mức nào.
                                                            #Giá trị AUC (nhãn thật/nhãn dự đoán)
                                                            # AUC	        Ý nghĩa
                                                            # 0.5	    model đoán random
                                                            # 0.7	          khá
                                                            # 0.8	          tốt
                                                            # 0.9	        rất tốt
                                                            # 1.0	        perfect

        train_losses.append(epoch_loss) #train_losses là list dùng để lưu loss từng epoch.

        # ============================================
        # VALIDATION
        # ============================================
        if val_loader is not None:
            model.eval() #chuyển sang chế độ validation
            val_loss_total = 0  #tổng loss của epoch
            val_probs = []      #prediction probability
            val_targets = []    #label thật

            with torch.no_grad(): #không cần tính gradient
                for xb, yb in val_loader:   #xb = batch input, yb = batch label
                    xb = xb.to(device)  #CPU → GPU
                    yb = yb.to(device)  #CPU → GPU
                    logits = model(xb)  #logits, output trước sigmoid vd tensor([-1.2, 0.8, 2.1])
                    loss = crit(logits, yb) #tính loss
                    val_loss_total += loss.item() #cộng tổng loss
                    val_probs.extend(torch.sigmoid(logits).cpu().numpy())  #lưu prediction: logits → sigmoid → probability
                                                                            #vd logits = [-1, 2, 0] -> sigmoid: [0.27, 0.88, 0.50] sau đó GPU tensor → CPU → numpy
                    val_targets.extend(yb.cpu().numpy()) #lưu nhãn thật

            val_epoch_loss = val_loss_total / len(val_loader) #tính loss trung bình của epoch
            val_auc = roc_auc_score(val_targets, val_probs) #đo xem mô hình hoạt động tốt đến mức nào
            val_losses.append(val_epoch_loss) #Lưu validation loss

            #In kết quả epoch
            # vd Epoch 5:
            # train_loss=0.41
            # val_loss=0.39
            # train_AUC=0.91
            # val_AUC=0.88
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | val_loss={val_epoch_loss:.4f} | train_AUC={epoch_auc:.4f} | val_AUC={val_auc:.4f}")

            if val_auc > best_auc: #có valid sẽ đánh giá dựa trên val, nếu model cải thiện thì cập nhập best
                best_auc = val_auc
                best_epoch = epoch
                wait = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                wait += 1       #nếu model không cải thiện, biến đếm tăng 1
                if wait >= patience: #sau 10 lần không cải thiện => ngắt, tránh overfitting
                    print("⏹ Early stopping kích hoạt.")
                    break
        else: #trường hợp không có validation (trường hợp train 100% data)
            print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} | train_AUC={epoch_auc:.4f}")
            torch.save(model.state_dict(), ckpt_path) #Lưu model mỗi epoch
            best_auc = epoch_auc   #không có valid, Cập nhật best dựa trên train AUC
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
        "model_type": model.__class__.__name__, #Lấy tên class model. ở đây là class BiLSTM(nn.Module): => model_type = "BiLSTM"
        "vocab_size": model.emb.num_embeddings, #Lưu số token trong vocab
        "trainer": "v7",                        #phiên bản
        "best_epoch": best_epoch,               #Epoch có val_auc tốt nhất
    }
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --------------------------------------------------------
    # Plot learning curve
    # --------------------------------------------------------
    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(f"{REPORT_DIR}/train_losses.json", "w") as f:
        json.dump(train_losses, f)          #Lưu train loss
    if val_losses:
        with open(f"{REPORT_DIR}/val_losses.json", "w") as f:
            json.dump(val_losses, f)        #Chỉ lưu khi có valid loss

    plt.figure(figsize=(8, 5))                      #tạo figure width = 8 inch, height = 5 inch
    plt.plot(train_losses, label="Train loss")      #Vẽ train loss, chỉ vẽ đường không hiển thị tên
                                                    # vd: train_losses
                                                    # epoc(X-axis)	    loss(Y-axis)
                                                    # 0	                    0.65
                                                    # 1	                    0.52
                                                    # 2	                    0.44
    if val_losses:
        plt.plot(val_losses, label="Validation loss")   #vẽ validation loss
    plt.xlabel("Epoch")                             #X-axis → Epoch
    plt.ylabel("Loss")                              #Y-axis → Loss
    plt.title("Learning Curve")                     #Tiêu đề: Learning curve = biểu đồ quá trình học của model
    plt.legend()                                    #Hiển thị tên:  train loss, validation loss
                                                    # ┌─────────────────────┐
                                                    # │ Train loss          │
                                                    # │ Validation loss     │
                                                    # └─────────────────────┘
    plt.grid(True)                                  #Thêm lưới cho dễ nhìn
    plt.savefig(f"{REPORT_DIR}/learning_curve.png", dpi=300) # Lưu biểu đồ, dpi=300 = độ phân giải cao (thường dùng cho report hoặc paper)
    plt.close()                             #Đóng figure, Nếu không close, khi train nhiều lần: Memory leak

    print("Learning curve exported to reports/")

    return best_auc
