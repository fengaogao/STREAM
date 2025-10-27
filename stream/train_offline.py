"""Offline training pipeline for STREAM."""  # STREAM 的离线训练流程入口说明
from __future__ import annotations  # 允许使用未来的注解特性

import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 以读取项目文本信息
import pickle  # 导入 pickle 用于持久化路由器
from collections import Counter  # 导入 Counter 统计项目类别频次
from pathlib import Path  # 导入 Path 方便文件路径操作
from typing import Dict, List, Tuple  # 导入类型提示

import torch  # 导入 PyTorch 深度学习库
from sklearn.cluster import KMeans  # 导入 KMeans 聚类用于构建路由器中心
from torch.optim import AdamW  # 导入 AdamW 优化器
from tqdm import tqdm  # 导入 tqdm 以展示训练进度

from config import StreamConfig  # 导入 STREAM 配置结构
from dataio import ItemVocab, build_dataloader, load_all_splits  # 导入数据加载相关工具
from models.causal_lm_stream import CausalLMStreamModel  # 导入因果语言模型封装
from models.bert_stream import BertStreamModel  # 导入 BERT 序列模型封装
from state_adapter import ItemHead, ItemHeadInit  # 导入低秩项目头初始化工具
from subspace import SubspaceResult, compute_subspace  # 导入子空间结果结构和原始计算函数
from utils import get_logger, set_seed  # 导入日志与随机种子工具

LOGGER = get_logger(__name__)  # 初始化模块级日志记录器


def parse_args() -> argparse.Namespace:  # 解析命令行参数的函数定义
    parser = argparse.ArgumentParser(description="Offline training for STREAM")  # 创建解析器并附上描述
    parser.add_argument("--data_dir", type=Path, default="/home/zj/code/STREAM/Yelp")  # 数据目录参数
    parser.add_argument("--out_dir", type=Path, default="/home/zj/code/STREAM/Yelp/bert")  # 输出目录参数
    parser.add_argument("--model_type", choices=["causal", "bert"], default="bert")  # 模型类型参数
    parser.add_argument("--pretrained_name_or_path", type=str, default="/home/zj/model/Llama-2-7b-hf")  # 预训练模型路径
    parser.add_argument("--rank_r", type=int, default=32)  # 方向子空间维度参数
    parser.add_argument("--router_k", type=int, default=16)  # 路由器聚类中心数量
    parser.add_argument("--subspace_mode", choices=["gradcov", "pca"], default="gradcov")  # 保留原有模式选项用于回退
    parser.add_argument("--epochs", type=int, default=10)  # 训练轮数
    parser.add_argument("--batch_size", type=int, default=64)  # 批大小
    parser.add_argument("--lr", type=float, default=2e-4)  # 学习率
    parser.add_argument("--seed", type=int, default=17)  # 随机种子
    parser.add_argument("--device", type=str, default="cuda")  # 训练设备
    return parser.parse_args()  # 返回解析后的参数


def train_epoch(model, dataloader, optimizer, device, model_type: str) -> float:  # 单轮训练函数定义
    model.train()  # 切换到训练模式
    total_loss = 0.0  # 初始化累计损失
    steps = 0  # 初始化步数
    for batch in tqdm(dataloader, desc="train", leave=False):  # 遍历训练数据并显示进度
        optimizer.zero_grad()  # 清零梯度
        if model_type == "causal":  # 根据模型类型准备输入
            inputs = {
                "input_ids": batch["input_ids"].to(device),  # 将输入 ID 移动到设备
                "attention_mask": batch["attention_mask"].to(device),  # 将注意力掩码移动到设备
                "labels": batch["labels"].to(device),  # 将标签移动到设备
            }
            outputs = model.model(**inputs)  # 前向计算获得输出
        else:  # 针对 BERT 模型的输入准备
            inputs = {
                "input_ids": batch["input_ids"].to(device),  # 将输入 ID 移动到设备
                "attention_mask": batch["attention_mask"].to(device),  # 将注意力掩码移动到设备
                "labels": batch["labels"].to(device),  # 将标签移动到设备
            }
            outputs = model.model(**inputs)  # 前向计算获得输出
        loss = outputs.loss  # 取出损失
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪保证稳定
        optimizer.step()  # 更新参数
        total_loss += float(loss.item())  # 累加损失
        steps += 1  # 更新步数
    return total_loss / max(steps, 1)  # 返回平均损失


def build_router(model, dataloader, router_k: int, device) -> Dict:  # 构建路由器中心的函数
    hidden_vectors = []  # 用于收集隐藏向量的列表
    for batch in dataloader:  # 遍历评估数据
        with torch.no_grad():  # 推理阶段无需梯度
            hidden = model.stream_hidden_states({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})  # 获取标准化前的隐藏状态
        hidden_vectors.append(torch.nn.functional.normalize(hidden, dim=-1).cpu())  # 将归一化后的隐藏向量移动到 CPU 收集
        if len(hidden_vectors) * hidden.shape[0] > 5000:  # 控制样本数量以节省计算
            break  # 满足数量后提前停止
    hidden_cat = torch.cat(hidden_vectors, dim=0)  # 拼接所有隐藏向量
    if router_k <= 1 or hidden_cat.size(0) < router_k:  # 若聚类数量不合理则退化为均值
        centers = hidden_cat.mean(dim=0, keepdim=True)  # 取均值作为中心
    else:  # 正常执行 KMeans 聚类
        kmeans = KMeans(n_clusters=router_k, random_state=0, n_init=10)  # 构造聚类模型
        kmeans.fit(hidden_cat.numpy())  # 在 CPU 上训练聚类
        centers = torch.from_numpy(kmeans.cluster_centers_)  # 将中心转为张量
    return {"centers": centers}  # 返回中心字典


def build_item_name_map(item_vocab: ItemVocab) -> dict[int, str]:  # 构造项目 ID 与名称映射
    m = {}  # 初始化映射字典
    for i in range(item_vocab.num_items):  # 遍历所有项目
        meta = item_vocab.meta_of(i) if hasattr(item_vocab, "meta_of") else {}  # 尝试读取项目元信息
        name = meta.get("title") or meta.get("name") or ""  # 取标题或名称
        m[i] = name  # 写入映射
    return m  # 返回映射


def load_item_categories(data_dir: Path, item_vocab: ItemVocab) -> Tuple[Dict[int, List[str]], List[str]]:  # 加载项目类别信息
    item_text_path = data_dir / "item_text.json"  # 指定文本信息文件路径
    category_map: Dict[int, List[str]] = {i: [] for i in range(item_vocab.num_items)}  # 预先为所有项目填空类别
    category_counter: Counter[str] = Counter()  # 初始化类别频次统计
    if not item_text_path.exists():  # 若没有文本文件
        LOGGER.warning("Item text metadata not found at %s", item_text_path)  # 记录警告便于定位
        return category_map, []  # 返回空类别信息
    with item_text_path.open("r", encoding="utf-8") as f:  # 打开文本信息文件
        item_text = json.load(f)  # 读取 JSON 内容
    marker = "Genres:"  # 指定用于提取类别的标记
    for idx_str, text in item_text.items():  # 遍历每个项目文本
        try:
            item_idx = int(idx_str)  # 将键转换为整数索引
        except ValueError:  # 如转换失败则跳过
            continue
        if item_idx >= item_vocab.num_items:  # 如果索引超出词表范围
            continue
        categories: List[str] = []  # 初始化项目类别列表
        if marker in text:  # 若文本中包含类别标记
            raw = text.split(marker, 1)[1]  # 截取类别部分
            categories = [c.strip() for c in raw.split(",") if c.strip()]  # 按逗号拆分并清理空格
        category_map[item_idx] = categories  # 记录该项目的类别列表
        category_counter.update(categories)  # 累计类别频次
    ordered_categories = [cat for cat, _ in category_counter.most_common()]  # 按频次由高到低排序类别
    return category_map, ordered_categories  # 返回项目到类别映射以及类别排序


def extract_targets_from_batch(batch: Dict[str, torch.Tensor], model_type: str) -> torch.Tensor:  # 从批次中提取目标项目 ID
    if model_type == "causal":  # 对于因果语言模型
        targets = batch["target_item"]  # 直接取目标项目张量
        if targets.is_cuda:  # 若在 GPU 上则转回 CPU
            targets = targets.detach().cpu()
        return targets.long()  # 返回长整型张量
    labels = batch["labels"]  # 对于 BERT 模型使用标签序列
    if labels.is_cuda:  # 若标签在 GPU 上
        labels = labels.detach().cpu()  # 转回 CPU 以便处理
    batch_targets = torch.full((labels.size(0),), -1, dtype=torch.long)  # 初始化目标张量
    for idx, row in enumerate(labels):  # 遍历每一行标签
        positives = row[row >= 0]  # 取出被遮蔽的真实项目
        if positives.numel() > 0:  # 若存在正样本
            batch_targets[idx] = int(positives[0].item())  # 记录第一个正样本的项目 ID
    return batch_targets  # 返回目标项目张量


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:  # 将批次移动到指定设备
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}  # 逐个张量搬移


def compute_category_semantic_subspace(
    model,
    dataloader,
    category_map: Dict[int, List[str]],
    category_order: List[str],
    max_rank: int,
    device: torch.device,
    model_type: str,
    fallback_mode: str,
    min_samples_per_category: int = 20,
) -> SubspaceResult:  # 基于类别语义构建子空间
    if not category_order:  # 若没有类别信息
        LOGGER.warning("No category metadata detected, falling back to %s", fallback_mode)  # 记录警告
        return compute_subspace(model, dataloader, rank=max_rank, mode=fallback_mode, device=device)  # 回退到原始方法
    model.eval()  # 切换模型到评估模式
    category_sums: Dict[str, torch.Tensor] = {}  # 存储每个类别的梯度累计向量
    category_counts: Dict[str, int] = {}  # 存储每个类别的样本数量
    total_sum: torch.Tensor | None = None  # 存储所有样本的梯度和
    total_count = 0  # 统计总样本数
    feature_dim: int | None = None  # 记录梯度向量维度
    for batch in tqdm(dataloader, desc="collect-directions", leave=False):  # 遍历数据构建方向
        batch_device = move_batch_to_device(batch, device)  # 将批次移动到目标设备
        grads = model.stream_positive_gradients(batch_device)  # 计算正样本梯度
        targets = extract_targets_from_batch(batch_device, model_type)  # 提取目标项目 ID
        grads = grads.detach()  # 确保梯度张量不保留计算图
        if grads.device != device:  # 若梯度不在目标设备
            grads = grads.to(device)  # 调整设备
        if feature_dim is None:  # 初始化累加器
            feature_dim = grads.size(-1)
            total_sum = torch.zeros(feature_dim, device=device)
        assert total_sum is not None  # 类型检查辅助
        for grad, target in zip(grads, targets.tolist()):  # 遍历批次样本
            total_sum += grad  # 累计整体梯度
            total_count += 1  # 更新总体计数
            if target < 0 or target not in category_map:  # 跳过没有类别的样本
                continue
            categories = category_map[target]  # 取出项目对应的类别列表
            if not categories:  # 若没有类别则跳过
                continue
            for category in categories:  # 遍历该项目所属的所有类别
                if category not in category_sums:  # 如首次出现则初始化
                    category_sums[category] = torch.zeros_like(grad)
                    category_counts[category] = 0
                category_sums[category] += grad  # 累加类别梯度
                category_counts[category] += 1  # 累计类别计数
    if total_count == 0 or feature_dim is None or total_sum is None:  # 若没有成功收集梯度
        LOGGER.warning("No gradients collected, falling back to %s", fallback_mode)  # 记录警告
        return compute_subspace(model, dataloader, rank=max_rank, mode=fallback_mode, device=device)  # 回退
    global_mean = total_sum / float(total_count)  # 计算整体平均梯度
    selected_categories: List[str] = []  # 准备选择的类别列表
    for category in category_order:  # 按频次排序遍历类别
        if category_counts.get(category, 0) < min_samples_per_category:  # 样本不足的类别跳过
            continue
        selected_categories.append(category)  # 加入候选列表
        if len(selected_categories) >= max_rank:  # 达到需求数量则停止
            break
    if not selected_categories:  # 若仍没有合适类别
        LOGGER.warning("No category passed the minimum sample threshold, fallback to %s", fallback_mode)  # 记录警告
        return compute_subspace(model, dataloader, rank=max_rank, mode=fallback_mode, device=device)  # 回退
    orthogonal_vectors: List[torch.Tensor] = []  # 存放正交化后的方向
    meta_categories: List[Dict] = []  # 记录用于解释的元信息
    for category in selected_categories:  # 遍历选中的类别
        category_mean = category_sums[category] / float(category_counts[category])  # 计算类别平均梯度
        direction = category_mean - global_mean  # 通过对比全局平均得到语义方向
        for existing in orthogonal_vectors:  # 进行 Gram-Schmidt 正交化
            projection = torch.dot(existing, direction) * existing
            direction = direction - projection
        norm = direction.norm()  # 计算方向向量范数
        if norm < 1e-6:  # 若方向过小则跳过
            continue
        direction = direction / norm  # 单位化方向向量
        orthogonal_vectors.append(direction)  # 保存方向
        meta_categories.append(
            {
                "category": category,
                "count": int(category_counts[category]),
                "share": float(category_counts[category] / total_count),
            }
        )  # 记录类别信息
        if len(orthogonal_vectors) >= max_rank:  # 若已达到上限则退出
            break
    if not orthogonal_vectors:  # 若正交化后没有保留方向
        LOGGER.warning("All semantic directions were filtered out, fallback to %s", fallback_mode)  # 记录警告
        return compute_subspace(model, dataloader, rank=max_rank, mode=fallback_mode, device=device)  # 回退
    basis = torch.stack(orthogonal_vectors, dim=1)  # 将方向向量按列堆叠为基矩阵
    meta = {
        "method": "category_contrast",  # 记录采用的语义差异方法
        "feature_dim": feature_dim,
        "total_samples": total_count,
        "categories": meta_categories,
    }  # 构造元信息
    return SubspaceResult(basis=basis.detach().cpu(), mode="gradcov", meta=meta)  # 返回子空间结果


def main() -> None:  # 主函数定义
    args = parse_args()  # 解析命令行参数
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 根据参数选择设备
    set_seed(args.seed)  # 设置随机种子确保可复现

    out_dir = args.out_dir  # 读取输出目录
    out_dir.mkdir(parents=True, exist_ok=True)  # 若目录不存在则创建

    item_vocab = ItemVocab.from_metadata(args.data_dir)  # 构建项目词表
    splits = load_all_splits(args.data_dir)  # 读取全部数据划分

    if args.model_type == "causal":  # 根据模型类型实例化模型
        item_name_map = build_item_name_map(item_vocab)  # 构建项目名称映射用于提示
        model = CausalLMStreamModel(
            args.pretrained_name_or_path,
            item_vocab,
            device,
            tokenizer_name_or_path=None,
        )  # 实例化因果语言模型
        tokenizer = model.tokenizer  # 保存分词器
    else:  # BERT 模型分支
        model = BertStreamModel(item_vocab, device)  # 实例化 BERT 模型
        tokenizer = None  # BERT 不需要额外分词器

    _, train_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=True,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )  # 构建训练数据加载器
    _, eval_loader = build_dataloader(
        splits["original"],
        model_type=args.model_type,
        batch_size=args.batch_size,
        shuffle=False,
        item_vocab=item_vocab,
        tokenizer=tokenizer,
    )  # 构建评估数据加载器

    optimizer = AdamW(model.parameters(), lr=args.lr)  # 创建优化器
    for epoch in range(args.epochs):  # 按轮次进行训练
        loss = train_epoch(model, train_loader, optimizer, device, args.model_type)  # 执行单轮训练
        LOGGER.info("Epoch %d loss %.4f", epoch + 1, loss)  # 记录训练损失

    category_map, category_order = load_item_categories(args.data_dir, item_vocab)  # 加载项目类别信息
    if args.subspace_mode == "pca":  # 若用户指定了 PCA 则沿用原始逻辑
        subspace = compute_subspace(model, eval_loader, rank=args.rank_r, mode="pca", device=device)  # 直接使用 PCA
    else:  # 默认使用语义方向构建子空间
        subspace = compute_category_semantic_subspace(
            model=model,
            dataloader=eval_loader,
            category_map=category_map,
            category_order=category_order,
            max_rank=args.rank_r,
            device=device,
            model_type=args.model_type,
            fallback_mode="gradcov",
        )  # 构建具备语义的子空间

    effective_rank = subspace.basis.size(1)  # 计算实际得到的方向数量
    config = StreamConfig(rank_r=effective_rank, router_k=args.router_k)  # 根据实际维度生成配置
    config.to_json(out_dir / "config.json")  # 保存配置文件

    subspace.save(out_dir)  # 保存子空间基

    U = subspace.basis.to(device)  # 将子空间基矩阵移到设备

    item_head = ItemHead(rank=effective_rank, num_items=item_vocab.num_items, device=device)  # 创建项目头
    if args.model_type == "causal":  # 针对因果模型取输出嵌入
        embeddings = model.lm_head_weight[model.item_token_ids].detach().t().to(device)  # 取出项目嵌入并转置
    else:  # 针对 BERT 模型取解码权重
        embeddings = model.decoder_weight.detach().t().to(device)  # 取出解码层权重
    item_head.initialise(ItemHeadInit(U=U.cpu(), item_embeddings=embeddings.cpu(), lambda_l2=config.lambda_l2))  # 初始化项目头权重
    torch.save({"W": item_head.state_dict(), "rank": effective_rank, "num_items": item_vocab.num_items}, out_dir / "item_head.pt")  # 持久化项目头参数

    router = build_router(model, eval_loader, args.router_k, device)  # 构建路由器中心
    with (out_dir / "router.pkl").open("wb") as f:  # 打开路由器文件
        pickle.dump(router, f)  # 保存路由器信息

    model_dir = out_dir / "model"  # 模型保存目录
    model_dir.mkdir(exist_ok=True)  # 若不存在则创建
    model.model.save_pretrained(model_dir)  # 保存基座模型参数
    if args.model_type == "causal":  # 如为因果模型则额外保存分词器
        tokenizer_dir = out_dir / "tokenizer"  # 指定分词器目录
        tokenizer.save_pretrained(tokenizer_dir)  # 保存分词器

    LOGGER.info("Training complete. Artifacts saved to %s", out_dir)  # 记录完成日志


if __name__ == "__main__":  # 当作为脚本执行时
    main()  # 调用主函数
