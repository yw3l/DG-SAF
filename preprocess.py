import os
import json
import argparse


def load_triples(file_path):
    """加载三元组文件 - 修正：处理每行多个三元组的情况"""
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # 处理每行包含多个三元组的情况 (每3个元素一组)
            for i in range(0, len(parts), 3):
                if i + 2 < len(parts):
                    h, r, t = parts[i], parts[i + 1], parts[i + 2]
                    triples.append((h, r, t))
    return triples


def load_entity_descriptions(file_path):
    """加载实体描述映射"""
    mid2desc = {}
    full_path=os.path.join('data/FB15k237', file_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                mid, desc = line.strip().split('\t', 1)
                mid2desc[mid] = desc
    return mid2desc


def load_entity_names(file_path):
    """加载实体名称映射"""
    mid2name = {}
    full_path=os.path.join('data/FB15k237', file_path)
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' in line:
                mid, name = line.strip().split('\t', 1)
                mid2name[mid] = name
    return mid2name


def create_mappings(triples):
    """创建实体和关系的ID映射"""
    entity_set = set()
    relation_set = set()

    for h, r, t in triples:
        entity_set.add(h)
        entity_set.add(t)
        relation_set.add(r)

    entities = sorted(entity_set)
    relations = sorted(relation_set)

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    return entity2id, relation2id, entities, relations


def save_mappings(entity2id, relation2id, output_dir):
    """保存ID映射到文件"""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'entities.txt'), 'w') as f:
        for e, id in entity2id.items():
            f.write(f"{id}\t{e}\n")

    with open(os.path.join(output_dir, 'relations.txt'), 'w') as f:
        for r, id in relation2id.items():
            f.write(f"{id}\t{r}\n")

    with open(os.path.join(output_dir, 'entity2id.json'), 'w') as f:
        json.dump(entity2id, f, indent=2)

    with open(os.path.join(output_dir, 'relation2id.json'), 'w') as f:
        json.dump(relation2id, f, indent=2)


def save_triples(triples, entity2id, relation2id, output_path):
    """保存三元组到文件 - 修正：确保只保存有效的三元组"""
    with open(output_path, 'w') as f:
        for h, r, t in triples:
            # 确保实体和关系在映射中
            if h in entity2id and t in entity2id and r in relation2id:
                f.write(f"{entity2id[h]}\t{relation2id[r]}\t{entity2id[t]}\n")


def create_text_descriptions(mid2desc, mid2name, entity2id, output_path):
    """创建实体文本描述文件"""
    entity_desc = {}
    for mid, id in entity2id.items():
        desc = mid2desc.get(mid, "")
        name = mid2name.get(mid, mid.split('/')[-1].replace('_', ' '))
        # 组合名称和描述作为完整文本
        full_text = f"{name}. {desc}" if desc else name
        entity_desc[str(id)] = full_text

    with open(output_path, 'w') as f:
        json.dump(entity_desc, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Preprocess FB15k-237 dataset for KGC')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train.txt, valid.txt, test.txt')
    parser.add_argument('--desc_path', type=str, required=True,
                        help='Path to FB15k_mid2description.txt')
    parser.add_argument('--name_path', type=str, required=True,
                        help='Path to FB15k_mid2name.txt')
    parser.add_argument('--output_dir', type=str, default='data/FB15k237',
                        help='Output directory for processed data')

    args = parser.parse_args()

    # 构建完整的文件路径
    train_path = os.path.join(args.data_dir, 'train.txt')
    valid_path = os.path.join(args.data_dir, 'valid.txt')
    test_path = os.path.join(args.data_dir, 'test.txt')

    # 加载数据
    print("Loading train triples...")
    train_triples = load_triples(train_path)

    print("Loading valid triples...")
    valid_triples = load_triples(valid_path)

    print("Loading test triples...")
    test_triples = load_triples(test_path)

    print("Loading entity descriptions...")
    mid2desc = load_entity_descriptions(args.desc_path)

    print("Loading entity names...")
    mid2name = load_entity_names(args.name_path)

    # 创建ID映射 - 使用所有三元组来创建完整的映射
    print("Creating entity and relation mappings...")
    all_triples = train_triples + valid_triples + test_triples
    entity2id, relation2id, entities, relations = create_mappings(all_triples)

    # 保存映射
    print("Saving mappings...")
    save_mappings(entity2id, relation2id, args.output_dir)

    # 保存所有三元组
    print("Saving train triples...")
    save_triples(train_triples, entity2id, relation2id, os.path.join(args.output_dir, 'train.txt'))

    print("Saving valid triples...")
    save_triples(valid_triples, entity2id, relation2id, os.path.join(args.output_dir, 'valid.txt'))

    print("Saving test triples...")
    save_triples(test_triples, entity2id, relation2id, os.path.join(args.output_dir, 'test.txt'))

    # 创建文本描述
    print("Creating entity text descriptions...")
    create_text_descriptions(mid2desc, mid2name, entity2id,
                             os.path.join(args.output_dir, 'entity_descriptions.json'))

    print(f"\nPreprocessing completed! Data saved to {args.output_dir}")
    print(f"Total entities: {len(entities)}")
    print(f"Total relations: {len(relations)}")
    print(f"Total train triples: {len(train_triples)}")
    print(f"Total valid triples: {len(valid_triples)}")
    print(f"Total test triples: {len(test_triples)}")


if __name__ == "__main__":
    main()
