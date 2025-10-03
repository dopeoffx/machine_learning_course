# apriori_trie_with_progress.py
from __future__ import annotations
from dataclasses import dataclass
from bisect import bisect_left
from typing import List, Tuple, Dict, Iterable
import sys, time
from collections import Counter

# ---------------- progress bar ----------------

class Progress:
    def __init__(self, total: int, desc: str = "") -> None:
        self.total = max(1, total)
        self.desc = desc
        self.start = time.time()
        self.last = self.start
        self.count = 0

    def update(self, n: int = 1) -> None:
        self.count += n
        now = time.time()
        
        if (now - self.last) >= 0.1 or self.count >= self.total:
            pct = self.count / self.total * 100.0
            rate = self.count / max(1e-9, now - self.start)
            eta = (self.total - self.count) / rate if rate > 0 else float("inf")
            sys.stdout.write(
                f"\r{self.desc}: {self.count}/{self.total} ({pct:5.1f}%)  "
                f"{rate:,.0f}/s  ETA {eta:5.1f}s"
            )
            sys.stdout.flush()
            self.last = now

    def finish(self) -> None:
        self.update(0)
        sys.stdout.write("\n")
        sys.stdout.flush()

# ---------------- dataset ----------------

@dataclass
class Dataset:
    txs: List[List[int]]

    @staticmethod
    def from_lines(lines: Iterable[str]) -> "Dataset":
        txs: List[List[int]] = []
        for line in lines:
            if not line.strip():
                continue
            nums = list(map(int, line.split()))
            nums.sort()  # jistota vzestupně
            txs.append(nums)
        return Dataset(txs)

    def __len__(self) -> int:
        return len(self.txs)

    def items_universe(self) -> List[int]:
        s = set()
        for t in self.txs:
            s.update(t)
        return sorted(s)

# ---------------- transaction trie ----------------

@dataclass
class _TrieNode:
    item: int | None
    pass_count: int
    children: List[Tuple[int,int]] 

class TxTrie:
    def __init__(self) -> None:
        self.nodes: List[_TrieNode] = [_TrieNode(item=None, pass_count=0, children=[])]
        self.root = 0

    @staticmethod
    def build(ds: Dataset, progress: bool = False) -> "TxTrie":
        t = TxTrie()
        N = len(ds)
        pb = Progress(N, "Building trie") if progress else None
        for tx in ds.txs:
            t._insert(tx)
            if pb: pb.update()
        if pb: pb.finish()
        return t

    def _insert(self, tx: List[int]) -> None:
        cur = self.root
        self.nodes[cur].pass_count += 1
        for it in tx:
            cur = self._child_or_create(cur, it)
            self.nodes[cur].pass_count += 1

    def _child_or_create(self, parent: int, item: int) -> int:
        ch = self.nodes[parent].children
        keys = [k for k,_ in ch]
        pos = bisect_left(keys, item)
        if pos < len(ch) and ch[pos][0] == item:
            return ch[pos][1]
        nid = len(self.nodes)
        self.nodes.append(_TrieNode(item=item, pass_count=0, children=[]))
        ch.insert(pos, (item, nid))
        return nid

    def support_of(self, items: List[int]) -> int:
        def dfs(node: int, j: int) -> int:
            if j == len(items):
                return self.nodes[node].pass_count
            target = items[j]
            total = 0
            for it, child in self.nodes[node].children:
                if it < target:
                    total += dfs(child, j)
                elif it == target:
                    total += dfs(child, j+1)
                    break
                else:
                    break
            return total
        if not items:
            return self.nodes[self.root].pass_count
        return dfs(self.root, 0)

# ---------------- apriori s trie + progress ----------------

def _threshold_count(min_sup_rel: float, n: int) -> int:
    from math import ceil
    assert 0.0 <= min_sup_rel <= 1.0
    return int(ceil(min_sup_rel * n))

def _generate_candidates(prev: List[Tuple[int,...]]) -> List[Tuple[int,...]]:
    if not prev: return []
    k_1 = len(prev[0])
    prev = sorted(prev)
    prev_set = set(prev)
    out: set[Tuple[int,...]] = set()
    for i in range(len(prev)):
        for j in range(i+1, len(prev)):
            a, b = prev[i], prev[j]
            if k_1 > 1 and a[:k_1-1] != b[:k_1-1]:
                break
            cand = a + (b[-1],)
            if any(cand[p] >= cand[p+1] for p in range(len(cand)-1)):
                continue
            ok = True
            for drop in range(len(cand)):
                sub = cand[:drop] + cand[drop+1:]
                if sub not in prev_set:
                    ok = False; break
            if ok: out.add(cand)
    return sorted(out)

def apriori_with_trie(
    ds: Dataset,
    min_sup_rel: float,
    progress: bool = False,
    l1_prune: bool = True,   
) -> Tuple[List[Tuple[Tuple[int,...], float]], Dict[Tuple[int,...], float]]:
    N = len(ds)
    if N == 0:
        return [], {}
    min_cnt = _threshold_count(min_sup_rel, N)

    # --- L₁ counting ---
    cnt = Counter()
    for t in ds.txs:
        cnt.update(t)

    # --- L₁ pruning transakcí ---
    if l1_prune:
        F1 = {it for it, c in cnt.items() if c >= min_cnt}
        pruned_txs = [[it for it in t if it in F1] for t in ds.txs]
        ds_pruned = Dataset(pruned_txs)
    else:
        ds_pruned = ds

    # postav trie z (případně) ořezaných transakcí
    trie = TxTrie.build(ds_pruned, progress=progress)

    frequent: List[Tuple[Tuple[int,...], float]] = []
    supp_map: Dict[Tuple[int,...], float] = {}

    # --- L₁  ---
    items = sorted(cnt.keys())  
    pb = Progress(len(items), "L1 supports") if progress else None
    last_level: List[Tuple[int,...]] = []
    for it in items:
        c = cnt[it]            
        if c >= min_cnt:
            s_rel = c / N
            iset = (it,)
            frequent.append((iset, s_rel))
            supp_map[iset] = s_rel
            last_level.append(iset)
        if pb: pb.update()
    if pb: pb.finish()

    # --- vyšší úrovně k ≥ 2 ---
    level = 1
    while last_level:
        Ck = _generate_candidates(last_level)
        if not Ck:
            break
        if progress:
            print(f"\nLevel L{level+1}: candidates = {len(Ck)}")
            pb = Progress(len(Ck), f"Counting supports L{level+1}")
        next_level: List[Tuple[int,...]] = []
        for cand in Ck:
            c = trie.support_of(list(cand))
            if c >= min_cnt:
                s_rel = c / N
                frequent.append((cand, s_rel))
                supp_map[cand] = s_rel
                next_level.append(cand)
            if progress: pb.update()
        if progress: pb.finish()
        last_level = next_level
        level += 1

    frequent.sort(key=lambda x: (len(x[0]), x[0]))
    return frequent, supp_map

# ---------------- pravidla----------------

@dataclass
class Rule:
    x: Tuple[int,...]
    y: Tuple[int,...]
    support: float
    confidence: float
    

def generate_rules(
    frequent: List[Tuple[Tuple[int,...], float]],
    supp_map: Dict[Tuple[int,...], float],
    min_conf: float,
    progress: bool = False
) -> List[Rule]:
    assert 0.0 <= min_conf <= 1.0

    total_masks = sum(max(0, (1 << len(I)) - 2) for I, _ in frequent if len(I) >= 2)
    pb = Progress(total_masks, "Generating rules") if progress else None

    rules: List[Rule] = []
    for I, sI in frequent:
        k = len(I)
        if k < 2:
            continue
        full = (1 << k) - 1
        for mask in range(1, full):
            if mask == full:
                continue
            X = tuple(I[i] for i in range(k) if (mask >> i) & 1)
            Y = tuple(I[i] for i in range(k) if (mask >> i) & 1 == 0)
            if not X or not Y:
                if pb: pb.update()
                continue
            sX = supp_map.get(X)
            sY = supp_map.get(Y)
            if sX is None or sX == 0:
                if pb: pb.update()
                continue
            conf = sI / sX
            if conf + 1e-12 >= min_conf:
                lift = conf / sY if sY and sY > 0 else float("inf")
                rules.append(Rule(x=X, y=Y, support=sI, confidence=conf))
            if pb: pb.update()
    if pb: pb.finish()

    rules.sort(key=lambda r: (-r.confidence, -r.support, len(r.y)))
    return rules

# ---------------- demo ----------------

if __name__ == "__main__":
    lines = open("small_items.txt")
    ds = Dataset.from_lines(lines)
    min_sup_rel = 0.25
    frequent, supp_map = apriori_with_trie(ds, min_sup_rel, progress=True)
    print("Frequent itemsets:")
    for iset, s in frequent:
        print(f"{list(iset)}  supp={s:.3f}")

    min_conf = 0.5
    rules = generate_rules(frequent, supp_map, min_conf, progress=True)
    print("\nRules (min_conf={min_conf}):")
    for r in rules[:10]:
        print(f"{list(r.x)} -> {list(r.y)}   supp={r.support:.3f}  conf={r.confidence:.3f}")
