import logging, numpy as np, random, re
from datasets import load_dataset, Dataset
from src.globals import SEED
 
logger = logging.getLogger(__name__)
 
def load_split(num_clients: int, alpha: float = 0.1, val_ratio: float = 0.2):
    random.seed(SEED); np.random.seed(SEED)
    humaneval = load_dataset("openai_humaneval", split="test", download_mode="reuse_dataset_if_exists")
    humaneval = humaneval.shuffle(SEED)
    mbpp = load_dataset("mbpp", split="test", download_mode="reuse_dataset_if_exists").select(range(300)).shuffle(SEED)
    mbpp_std = mbpp.map(lambda ex: {
        'task_id': f'MBPP/{ex["task_id"]}', 
        'prompt': ex['text'] + '\n' + ex['code'], 
        'test': ex['test_list'][0] if ex['test_list'] else '', 
        'canonical_solution': ex['code'],
        'entry_point': 'solution'  # Default entry point for MBPP
    })
    full_ds = Dataset.from_list(humaneval.to_list() + mbpp_std.to_list()).shuffle(SEED)
    tv = full_ds.train_test_split(test_size=0.25, seed=SEED)
    global_train, global_test = tv['train'], tv['test']
    probs = np.random.dirichlet([alpha] * num_clients, len(global_train))
    buckets = [[] for _ in range(num_clients)]
    for ex, p in zip(global_train, probs):
        buckets[int(np.argmax(p))].append(ex)
    for idx, b in enumerate(buckets):
        if not b:
            donor = max(range(num_clients), key=lambda j: len(buckets[j]))
            buckets[idx].append(buckets[donor].pop())
        if len(b) < 5:
            for _ in range(5 - len(b)):
                orig = random.choice(b)
                mutated_prompt = re.sub(r'var(\d)', lambda m: f'var{int(m.group(1))+1}', orig['prompt'])
                b.append({'prompt': mutated_prompt, 'test': orig['test'], 'canonical_solution': orig['canonical_solution']})
    client_splits = []
    for cid, bucket in enumerate(buckets):
        ds = Dataset.from_list(bucket)
        tr_val = ds.train_test_split(test_size=val_ratio, seed=SEED)
        client_splits.append((tr_val['train'], tr_val['test']))
        logger.info(f"[Split] CID{cid} train={len(tr_val['train'])} val={len(tr_val['test'])}")
    return client_splits, global_test
