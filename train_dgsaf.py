import os
import json
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from collections import defaultdict
import torch.nn.functional as F


class KGDataset(Dataset):
    """知识图谱数据集"""
    def __init__(self, triples, entity_descriptions, entity2id, relation2id, num_entities,
                 negative_sample_size=256, mode='train'):
        self.triples = triples
        self.entity_descriptions = entity_descriptions
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.num_entities = num_entities
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        # 创建实体-关系-尾实体映射 (使用ID)
        self.hr2t = defaultdict(set)
        for h, r, t in triples:
            self.hr2t[(h, r)].add(t)
        # 创建头实体-关系-实体映射 (使用ID)
        self.tr2h = defaultdict(set)
        for h, r, t in triples:
            self.tr2h[(t, r)].add(h)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        # 生成负样本
        negative_samples = []
        if self.mode == 'train':
            # 头实体负采样
            neg_h_samples = set()
            while len(neg_h_samples) < self.negative_sample_size // 2:
                neg_h = random.randint(0, self.num_entities - 1)
                if neg_h not in self.tr2h[(t, r)] and neg_h != h:
                    neg_h_samples.add(neg_h)
            # 尾实体负采样
            neg_t_samples = set()
            while len(neg_t_samples) < self.negative_sample_size // 2:
                neg_t = random.randint(0, self.num_entities - 1)
                if neg_t not in self.hr2t[(h, r)] and neg_t != t:
                    neg_t_samples.add(neg_t)
            negative_samples = list(neg_h_samples) + list(neg_t_samples)
        # 获取关系文本描述
        try:
            # 从 relation2id 的 key (MID) 获取描述
            r_mid = list(self.relation2id.keys())[list(self.relation2id.values()).index(r)]
            r_text = f"Relation: {r_mid.split('/')[-1].replace('_', ' ')}"
        except (ValueError, IndexError):
            r_text = "Unknown relation"

        # 获取实体描述
        h_desc = self.entity_descriptions.get(str(h), "Unknown entity")
        t_desc = self.entity_descriptions.get(str(t), "Unknown entity")

        # 确保 negative_samples 总是返回一个 tensor，即使为空
        neg_tensor = torch.tensor(negative_samples, dtype=torch.long) if negative_samples else torch.empty(0, dtype=torch.long)

        # 返回 r_id 用于 evaluate 过滤
        return (
            torch.tensor(h, dtype=torch.long), # 0
            r_text,                           # 1
            torch.tensor(t, dtype=torch.long), # 2
            neg_tensor,                       # 3 (Always a tensor)
            h_desc,                           # 4
            t_desc,                           # 5
            torch.tensor(r, dtype=torch.long) # 6 - Add r_id for filtering
        )


class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, bert_model="bert-base-uncased", projection_dim=200):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert = BertModel.from_pretrained(bert_model)
        self.projection = nn.Linear(self.bert.config.hidden_size, projection_dim)
        # 冻结BERT底层
        for param in self.bert.parameters():
            param.requires_grad = False
        # 只解冻最后两层
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

    def forward(self, texts):
        """编码文本描述
        Args:
            texts: list[str] or str
        Returns:
            normalized_embeddings: [batch_size, projection_dim]
        """
        if isinstance(texts, str):
             texts = [texts] # Handle single string input

        # Ensure tokenizer and model are on the same device as input tensors will be later
        device = next(self.parameters()).device

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=32,
            return_tensors="pt"
            # return_token_type_ids 不是所有模型都需要，且 BertTokenizer 默认会处理
        ).to(device) # Move inputs to the same device as the model

        with torch.set_grad_enabled(self.training): # Respect training mode for BERT layers
             outputs = self.bert(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        projected = self.projection(cls_embeddings)  # [batch_size, projection_dim]
        return torch.nn.functional.normalize(projected, p=2, dim=-1) # L2 Normalize


class GeometryParameterizer(nn.Module):
    """几何参数化模块"""
    def __init__(self, text_dim, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, text_emb):
        """生成几何操作参数 [theta, scale]"""
        params = self.mlp(text_emb)  # [batch, 2]
        theta = torch.tanh(params[:, 0]) * np.pi  # 旋转角 ∈ [-π, π]
        scale = torch.nn.functional.softplus(params[:, 1]) + 0.1  # 缩放因子 > 0
        return theta, scale


class ComplexGeometryOperator(nn.Module):
    """复数空间几何操作"""
    def __init__(self):
        super().__init__()

    def forward(self, h_real, h_imag, theta, scale):
        """复数空间几何操作: t = s * (h_real + i*h_imag) * e^{iθ}"""
        # 扩展维度以进行广播
        cos_theta = torch.cos(theta).unsqueeze(-1) # [B, 1]
        sin_theta = torch.sin(theta).unsqueeze(-1) # [B, 1]
        scale_expanded = scale.unsqueeze(-1)       # [B, 1]

        # 旋转操作: (h_real + i*h_imag) * (cosθ + i*sinθ)
        t_real_rot = h_real * cos_theta - h_imag * sin_theta # [B, D/2]
        t_imag_rot = h_real * sin_theta + h_imag * cos_theta # [B, D/2]

        # 缩放操作
        t_pred_real = scale_expanded * t_real_rot # [B, D/2]
        t_pred_imag = scale_expanded * t_imag_rot # [B, D/2]
        return t_pred_real, t_pred_imag


class DGSAF(nn.Module):
    """Dynamic Geometry-Semantic Adaptive Fusion (DG-SAF)"""
    def __init__(self, num_entities, text_dim=200, complex_dim=200,
                 bert_model="bert-base-uncased", lambda_text=0.5, gamma_consist=0.3, margin=6.0):
        super().__init__()
        self.complex_dim = complex_dim
        self.lambda_text = lambda_text
        self.gamma_consist = gamma_consist
        self.margin = margin

        # 文本编码器
        self.text_encoder = TextEncoder(bert_model, text_dim)

        # 结构编码器 (复数空间)
        self.entity_real = nn.Embedding(num_entities, complex_dim // 2)
        self.entity_imag = nn.Embedding(num_entities, complex_dim // 2)
        nn.init.xavier_uniform_(self.entity_real.weight)
        nn.init.xavier_uniform_(self.entity_imag.weight)

        # 几何参数化模块
        self.geometry_param = GeometryParameterizer(text_dim)

        # 文本-结构映射器 (用于一致性损失)
        self.text_to_struct = nn.Linear(text_dim, complex_dim)

        # 几何操作器
        self.geometry_op = ComplexGeometryOperator()

    def forward(self, h_ids, r_texts, t_ids, t_neg_ids=None, h_descs=None, t_descs=None):
        """
        Args:
            h_ids: 头实体ID [batch]
            r_texts: 关系文本描述 list[str] (e.g., "Relation: /film/actor/film")
            t_ids: 尾实体ID [batch]
            t_neg_ids: 负例尾实体ID [batch, neg_samples] or empty tensor
            h_descs: 头实体的文本描述 list[str] (来自KGDataset)
            t_descs: 尾实体的文本描述 list[str] (来自KGDataset)
        Returns:
            total_loss, (struct_loss, text_loss, consist_loss)
        """
        device = h_ids.device
        batch_size = h_ids.shape[0]

        # ===== 1. 文本编码 (使用修正后的 TextEncoder) =====
        # 编码关系、头实体和尾实体的文本描述
        r_text_emb = self.text_encoder(r_texts)      # [batch, text_dim]
        # 使用 KGDataset 提供的真实描述
        h_text_emb = self.text_encoder(h_descs)      # [batch, text_dim]
        t_text_emb = self.text_encoder(t_descs)      # [batch, text_dim]

        # ===== 2. 生成动态几何参数 =====
        theta, scale = self.geometry_param(r_text_emb)  # theta [B], scale [B]

        # ===== 3. 结构编码 =====
        h_real = self.entity_real(h_ids)  # [batch, complex_dim // 2]
        h_imag = self.entity_imag(h_ids)
        t_real = self.entity_real(t_ids)
        t_imag = self.entity_imag(t_ids)

        # ===== 4. 几何操作预测 =====
        # 使用几何操作器
        t_pred_real, t_pred_imag = self.geometry_op(h_real, h_imag, theta, scale)
        # t_pred_real, t_pred_imag: [batch, complex_dim // 2]

        # ===== 5. 计算结构损失 (CompoundE) =====
        # 正例得分: Re(<h, r, t>) = dot(t_pred, t)
        pos_scores = (t_pred_real * t_real).sum(-1) + (t_pred_imag * t_imag).sum(-1) # [B]
        struct_loss = torch.tensor(0.0, device=device)
        if t_neg_ids is not None and t_neg_ids.numel() > 0: # Check if tensor is not empty
            t_neg_real = self.entity_real(t_neg_ids)  # [B, neg, complex_dim/2]
            t_neg_imag = self.entity_imag(t_neg_ids)
            # 扩展预测值以匹配负例维度
            t_pred_real_exp = t_pred_real.unsqueeze(1)  # [B, 1, complex_dim/2]
            t_pred_imag_exp = t_pred_imag.unsqueeze(1)
            # 负例得分
            neg_scores = (
                    (t_pred_real_exp * t_neg_real).sum(-1) +
                    (t_pred_imag_exp * t_neg_imag).sum(-1)
            )  # [B, neg]
            # 结构损失: max(0, margin - pos_score + neg_score)
            struct_loss = torch.relu(
                self.margin - pos_scores.unsqueeze(1) + neg_scores
            ).mean()

        # ===== 6. 计算文本损失 (SimKGC) =====
        # 将结构嵌入映射到文本空间（或反之）进行比较。
        # 这里选择将文本嵌入映射到结构空间 (复数维度)
        # 注意：h_text_emb + r_text_emb 是在文本空间操作，然后映射到结构空间
        combined_text_emb = h_text_emb + r_text_emb # [B, text_dim]
        combined_text_mapped = self.text_to_struct(combined_text_emb) # [B, complex_dim]

        # 将尾实体文本也映射到结构空间
        t_text_mapped = self.text_to_struct(t_text_emb) # [B, complex_dim]

        # 将结构预测的复数嵌入转为实数向量 [实部; 虚部] 用于比较
        t_pred_struct = torch.cat([t_pred_real, t_pred_imag], dim=-1) # [B, complex_dim]

        # 计算相似度得分 (例如 cosine similarity)
        # text_scores = F.cosine_similarity(combined_text_mapped, t_text_mapped, dim=-1) # 如果比较 t_text 和 h+r_text
        text_scores = F.cosine_similarity(combined_text_mapped, t_pred_struct, dim=-1) # 如果比较 t_pred 和 h+r_text 映射

        # SimKGC对比损失 (这里简化处理，实际可能需要负采样)
        # 假设目标是让正样本得分高，可以使用 margin loss 或 infoNCE
        # 使用 margin loss 示例:
        # 假设负样本是随机采样的其他 t_text (简化，实际应与结构负样本一致或另采样)
        # 这里为了简化，我们只计算正样本的损失，鼓励正样本得分高。
        # 更严格的实现需要负样本。
        text_loss = F.relu(0.1 - text_scores).mean() # Margin loss encouraging score > 0.1


        # ===== 7. 自适应一致性正则化 =====
        # 比较结构预测 t_pred_struct 和 文本预测 t_text_mapped
        similarity = torch.cosine_similarity(t_pred_struct, t_text_mapped, dim=-1) # [B]
        w_r = torch.sigmoid(similarity)  # w_r ∈ [0,1] [B]

        # 一致性损失: w_r * ||t_pred - t_text||^2
        mse_loss = torch.nn.functional.mse_loss(
            t_pred_struct, t_text_mapped, reduction='none'
        ).mean(-1) # [B]
        consist_loss = (w_r * mse_loss).mean()

        # ===== 8. 总损失 =====
        total_loss = (
                struct_loss +
                self.lambda_text * text_loss +
                self.gamma_consist * consist_loss
        )
        return total_loss, (struct_loss.item(), text_loss.item(), consist_loss.item())


def load_data(data_dir):
    """加载数据集"""
    # 加载ID映射
    with open(os.path.join(data_dir, 'entity2id.json'), 'r') as f:
        entity2id = json.load(f)
    with open(os.path.join(data_dir, 'relation2id.json'), 'r') as f:
        relation2id = json.load(f)
    # 加载三元组
    def load_triples(file_path):
        triples = []
        with open(os.path.join(data_dir, file_path), 'r') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((int(h), int(r), int(t)))
        return triples
    train_triples = load_triples('train.txt')
    valid_triples = load_triples('valid.txt')
    test_triples = load_triples('test.txt')
    # 加载实体描述
    with open(os.path.join(data_dir, 'entity_descriptions.json'), 'r') as f:
        entity_descriptions = json.load(f)
    return {
        'train': train_triples,
        'valid': valid_triples,
        'test': test_triples,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'entity_descriptions': entity_descriptions
    }


def evaluate(model, dataset, device, num_entities, hits_at_k=[1, 3, 10]):
    """评估模型性能 (Filter Setting)"""
    model.eval()
    total_ranks = []
    total_mrr = 0.0

    with torch.no_grad():
        # 使用 DataLoader 可以简化批次处理和设备移动
        eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # h_ids, r_texts, t_ids, _, h_descs, t_descs, r_ids = batch # Unpack all 7 elements
            h_ids, r_texts, t_ids, _, h_descs, t_descs, r_ids = [b.to(device) if isinstance(b, torch.Tensor) and b.dtype != torch.long else b for b in batch]
            # Move tensors to device, keep lists of strings as they are
            # h_ids, t_ids, r_ids are tensors, r_texts, h_descs, t_descs are lists
            h_ids = h_ids.to(device)
            t_ids = t_ids.to(device)
            r_ids = r_ids.to(device) # Needed for filtering

            # 获取所有实体的嵌入 (这部分逻辑正确)
            all_real = model.entity_real.weight # [num_entities, complex_dim/2]
            all_imag = model.entity_imag.weight # [num_entities, complex_dim/2]

            # 预测
            r_text_emb = model.text_encoder(r_texts) # [B, text_dim]
            theta, scale = model.geometry_param(r_text_emb) # [B], [B]
            h_real = model.entity_real(h_ids) # [B, D/2]
            h_imag = model.entity_imag(h_ids) # [B, D/2]
            t_pred_real, t_pred_imag = model.geometry_op(
                h_real, h_imag, theta, scale
            ) # [B, D/2], [B, D/2]

            # 计算所有实体的得分 [B, N]
            # 使用 einsum 或 bmm
            # t_pred: [B, D/2], all_real: [N, D/2] -> scores: [B, N]
            scores_real = torch.matmul(t_pred_real, all_real.transpose(0, 1)) # [B, N]
            scores_imag = torch.matmul(t_pred_imag, all_imag.transpose(0, 1)) # [B, N]
            scores = scores_real + scores_imag # [B, N]

            # 排序并计算排名 (这部分逻辑基本正确，但需要处理过滤)
            for j in range(scores.shape[0]):
                target = t_ids[j].item()
                # 过滤已知的正例 (filter setting)
                # 使用 dataset.hr2t 中存储的 ID 映射
                h_id_j = h_ids[j].item()
                r_id_j = r_ids[j].item()
                # 获取所有已知的 (h_id_j, r_id_j, ?) 的尾实体 ID
                filter_out = dataset.hr2t.get((h_id_j, r_id_j), set())

                # 创建过滤后的得分
                filtered_scores = scores[j].clone().cpu().numpy() # [N]
                for t in filter_out:
                    if t != target: # Don't filter the correct answer
                        filtered_scores[t] = -1e10  # 将已知正例(除目标外)设为极低分

                # 计算排名
                sorted_indices = np.argsort(-filtered_scores) # Descending order
                rank = np.where(sorted_indices == target)[0][0] + 1
                total_ranks.append(rank)
                total_mrr += 1.0 / rank

        # 计算评估指标
        if not total_ranks:
             print("Warning: No ranks calculated in evaluation.")
             return 0.0, {k: 0.0 for k in hits_at_k}

        total_ranks = np.array(total_ranks)
        mrr = total_mrr / len(total_ranks)
        hits = {k: np.mean(total_ranks <= k) for k in hits_at_k}
        return mrr, hits


def train(args):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    print("Loading dataset...")
    data = load_data(args.data_dir)

    # 创建数据集
    print("Creating datasets...")
    train_dataset = KGDataset(
        data['train'], data['entity_descriptions'],
        data['entity2id'], data['relation2id'],
        len(data['entity2id']), args.negative_sample_size, 'train'
    )
    valid_dataset = KGDataset(
        data['valid'], data['entity_descriptions'],
        data['entity2id'], data['relation2id'],
        len(data['entity2id']), 0, 'valid' # Negative samples not needed for eval
    )
    test_dataset = KGDataset( # Add test dataset
        data['test'], data['entity_descriptions'],
        data['entity2id'], data['relation2id'],
        len(data['entity2id']), 0, 'test' # Negative samples not needed for eval
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True if torch.cuda.is_available() else False
    )

    # 初始化模型
    print("Initializing model...")
    model = DGSAF(
        num_entities=len(data['entity2id']),
        text_dim=args.text_dim,
        complex_dim=args.complex_dim,
        bert_model=args.bert_model,
        lambda_text=args.lambda_text,
        gamma_consist=args.gamma_consist,
        margin=args.margin
    ).to(device)

    # 优化器
    optimizer = optim.AdamW([
        {'params': model.text_encoder.projection.parameters(), 'lr': args.text_lr},
        {'params': model.text_encoder.bert.encoder.layer[-2:].parameters(), 'lr': args.text_lr}, # Add BERT fine-tuning params
        {'params': model.geometry_param.parameters(), 'lr': args.param_lr},
        {'params': model.entity_real.parameters(), 'lr': args.struct_lr},
        {'params': model.entity_imag.parameters(), 'lr': args.struct_lr},
        {'params': model.text_to_struct.parameters(), 'lr': args.text_lr},
    ], weight_decay=args.weight_decay)

    # 学习率调度器
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # 训练循环
    print("Starting training...")
    best_mrr = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        struct_loss_total = 0
        text_loss_total = 0
        consist_loss_total = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in progress_bar:
            # h_ids, r_texts, t_ids, t_neg_ids = batch # OLD
            h_ids, r_texts, t_ids, t_neg_ids, h_descs, t_descs, _ = batch # NEW: Unpack all 7 elements, ignore r_ids for training

            # 移动到设备 (h_ids, t_ids, t_neg_ids are tensors)
            h_ids = h_ids.to(device)
            t_ids = t_ids.to(device)
            # t_neg_ids might be an empty tensor, that's okay
            if t_neg_ids.numel() > 0: # Only move if it has elements
                 t_neg_ids = t_neg_ids.to(device)
            # r_texts, h_descs, t_descs are lists of strings, handled by TextEncoder

            # 前向传播 - 传递所有参数
            optimizer.zero_grad()
            # OLD: loss, (struct_loss, text_loss, consist_loss) = model(h_ids, r_texts, t_ids, t_neg_ids)
            # NEW: Pass h_descs and t_descs
            try:
                loss, (struct_loss, text_loss, consist_loss) = model(
                    h_ids, r_texts, t_ids, t_neg_ids, h_descs, t_descs
                )
            except Exception as e:
                print(f"Error in forward pass: {e}")
                print(f"Batch shapes: h_ids={h_ids.shape}, t_ids={t_ids.shape}, t_neg_ids={t_neg_ids.shape if t_neg_ids is not None else None}")
                print(f"r_texts length: {len(r_texts)}, h_descs length: {len(h_descs)}, t_descs length: {len(t_descs)}")
                raise e

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # 记录损失
            total_loss += loss.item()
            struct_loss_total += struct_loss
            text_loss_total += text_loss
            consist_loss_total += consist_loss

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'S': f'{struct_loss:.4f}',
                'T': f'{text_loss:.4f}',
                'C': f'{consist_loss:.4f}'
            })

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_struct_loss = struct_loss_total / len(train_loader)
        avg_text_loss = text_loss_total / len(train_loader)
        avg_consist_loss = consist_loss_total / len(train_loader)

        # 验证
        print("Starting validation...")
        mrr, hits = evaluate(
            model, valid_dataset, device,
            len(data['entity2id']), args.hits_at_k
        )

        # 打印结果
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}:")
        print(
            f"  Loss: {avg_loss:.4f} (Struct: {avg_struct_loss:.4f}, Text: {avg_text_loss:.4f}, Consist: {avg_consist_loss:.4f})")
        print(f"  MRR: {mrr:.4f}")
        print(f"  Hits@{args.hits_at_k}: {[hits[k] for k in args.hits_at_k]}")
        print(f"  Time: {epoch_time:.2f}s")

        # 保存最佳模型
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  New best model saved with MRR: {mrr:.4f}")

    # 加载最佳模型并测试
    print("Evaluating on test set...")
    # Load test dataset
    # test_dataset = KGDataset( # Already created above
    #     data['test'], data['entity_descriptions'],
    #     data['entity2id'], data['relation2id'],
    #     len(data['entity2id']), 0, 'test' # Negative samples not needed for eval
    # )
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    # Evaluate
    test_mrr, test_hits = evaluate(
        model, test_dataset, device, # Use test_dataset here
        len(data['entity2id']), args.hits_at_k
    )
    print("\nTest Results:")
    print(f"  MRR: {test_mrr:.4f}")
    print(f"  Hits@{args.hits_at_k}: {[test_hits[k] for k in args.hits_at_k]}")

    # 保存结果
    results = {
        'mrr': test_mrr,
        'hits': {k: test_hits[k] for k in args.hits_at_k},
        'args': vars(args)
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DG-SAF for Knowledge Graph Completion')
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data/FB15k-237',
                        help='Directory containing preprocessed data')
    # 模型参数
    parser.add_argument('--text_dim', type=int, default=200,
                        help='Dimension of text embeddings')
    parser.add_argument('--complex_dim', type=int, default=400,
                        help='Dimension of complex embeddings (real + imag)')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model')
    parser.add_argument('--lambda_text', type=float, default=0.5,
                        help='Weight for text loss')
    parser.add_argument('--gamma_consist', type=float, default=0.3,
                        help='Weight for consistency loss')
    parser.add_argument('--margin', type=float, default=6.0,
                        help='Margin for structure loss')
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--negative_sample_size', type=int, default=256,
                        help='Number of negative samples per positive')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--text_lr', type=float, default=1e-4,
                        help='Learning rate for text components')
    parser.add_argument('--param_lr', type=float, default=5e-4,
                        help='Learning rate for geometry parameters')
    parser.add_argument('--struct_lr', type=float, default=1e-3,
                        help='Learning rate for structure embeddings')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Warmup steps for scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    # 评估参数
    parser.add_argument('--hits_at_k', nargs='+', type=int,
                        default=[1, 3, 10], help='Hits@k values')
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs/dgsaf',
                        help='Output directory')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存参数
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    train(args)
