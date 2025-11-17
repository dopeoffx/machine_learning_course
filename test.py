from sklearn.datasets import load_iris
import numpy as np

# ---------- Uzlová struktura stromu ----------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 class_counts=None, prediction=None, depth=0):
        self.feature = feature          # index sloupce pro split (int) nebo None v listu
        self.threshold = threshold      # hranice pro split (float) nebo None v listu
        self.left = left                # levý potomek (Node)
        self.right = right              # pravý potomek (Node)
        self.class_counts = class_counts  # počty tříd v uzlu (np.array shape (k,))
        self.prediction = prediction    # int - většinová třída v listu
        self.depth = depth              # hloubka uzlu (pro info / debug)

    @property
    def is_leaf(self):
        return self.feature is None


class decision_tree:
    def __init__(self, metric="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0):
        """
        metric: "gini" nebo "entropy" (tady implementujeme Gini; entropy můžeš doplnit)
        max_depth: maximální hloubka (None = bez omezení)
        min_samples_split: min. počet vzorků v uzlu, aby se zkoušel split
        min_samples_leaf: min. počet vzorků v každém listu po splitu
        min_impurity_decrease: minimální zlepšení impurity, jinak se nesplítne
        """
        self.metric = metric
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None

    # --------- veřejné API ---------
    def train(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = int(y.max() + 1)
        self.n_features_ = X.shape[1]
        idxs = np.arange(len(y))
        self.root = self._build(X, y, idxs, depth=0)
        return self

    def test(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(self.root, x) for x in X], dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probs = []
        for x in X:
            node = self._descend(self.root, x)
            counts = node.class_counts.astype(float)
            s = counts.sum()
            if s == 0:
                # fallback – uniformní
                probs.append(np.ones(self.n_classes_) / self.n_classes_)
            else:
                probs.append(counts / s)
        return np.vstack(probs)

    # --------- jádro trénování ---------
    def _build(self, X, y, idxs, depth):
        # spočti počty tříd v uzlu
        counts = np.bincount(y[idxs], minlength=self.n_classes_)
        prediction = int(np.argmax(counts))

        # zastavovací podmínky (pre-pruning)
        if self._should_stop(X, y, idxs, counts, depth):
            return Node(class_counts=counts, prediction=prediction, depth=depth)

        # impurity rodiče
        parent_imp = self._impurity_from_counts(counts)

        # najdi nejlepší split přes všechny rysy
        best_feat, best_thr, best_imp_after, left_idx, right_idx = self._best_split(X, y, idxs, parent_imp)
        if best_feat is None:
            # žádné smysluplné dělení
            return Node(class_counts=counts, prediction=prediction, depth=depth)

        # vytvoř potomky
        left_child = self._build(X, y, left_idx, depth+1)
        right_child = self._build(X, y, right_idx, depth+1)
        return Node(feature=best_feat, threshold=best_thr,
                    left=left_child, right=right_child,
                    class_counts=counts, prediction=prediction, depth=depth)

    def _should_stop(self, X, y, idxs, counts, depth):
        # čistý uzel
        if np.count_nonzero(counts) == 1:
            return True
        # min. počet vzorků pro split
        if idxs.size < self.min_samples_split:
            return True
        # max. hloubka
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _best_split(self, X, y, idxs, parent_imp):
        """
        Pro numerické rysy:
        - seřadí vzorky podle X[:, j]
        - kandidátní prahy vezme mezi sousední odlišné hodnoty
        - spočte váženou impurity po splitu a vybere minimum
        """
        n = idxs.size
        best_feat = None
        best_thr = None
        best_imp_after = np.inf
        best_left_idx = None
        best_right_idx = None

        # celkový počet tříd v uzlu (pro rychlé prefixy)
        y_node = y[idxs]

        for j in range(self.n_features_):
            # hodnoty ve sloupci j pro aktuální uzel
            xj = X[idxs, j]
            # setřídit podle hodnot; zároveň přeskup y
            order = np.argsort(xj, kind="mergesort")  # stabilní řazení, determinismus
            xj_sorted = xj[order]
            y_sorted = y_node[order]

            # prefix counts vlevo, suffix counts vpravo
            left_counts = np.zeros(self.n_classes_, dtype=int)
            right_counts = np.bincount(y_sorted, minlength=self.n_classes_)

            # procházej mezi pozicemi a zvažuj hranici mezi různými hodnotami
            for i in range(0, n - 1):
                cls = y_sorted[i]
                left_counts[cls] += 1
                right_counts[cls] -= 1

                # pokud jsou sousední hodnoty stejné, nemá smysl dávat práh mezi ně
                if xj_sorted[i] == xj_sorted[i + 1]:
                    continue

                left_n = i + 1
                right_n = n - left_n

                # min_samples_leaf podmínka
                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                # impurity dětí
                imp_left = self._impurity_from_counts(left_counts)
                imp_right = self._impurity_from_counts(right_counts)
                imp_after = (left_n / n) * imp_left + (right_n / n) * imp_right

                # požadovaný minimální pokles impurity
                if parent_imp - imp_after < self.min_impurity_decrease:
                    continue

                if imp_after < best_imp_after:
                    best_imp_after = imp_after
                    best_feat = j
                    # práh dáme mezi dvě hodnoty (průměr)
                    best_thr = (xj_sorted[i] + xj_sorted[i + 1]) / 2.0

                    # ulož si i indexy větví pro rekurzi (abychom je nemuseli znovu hledat)
                    mask_left = X[idxs, j] <= best_thr
                    best_left_idx = idxs[mask_left]
                    best_right_idx = idxs[~mask_left]

        return best_feat, best_thr, best_imp_after, best_left_idx, best_right_idx

    # --------- impurity ----------
    def _impurity_from_counts(self, counts):
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        if self.metric == "gini":
            return 1.0 - np.sum(p * p)
        elif self.metric == "entropy":
            # volitelně: jednoduchá entropie (bez log(0))
            p_nonzero = p[p > 0]
            return -np.sum(p_nonzero * np.log2(p_nonzero))
        else:
            # fallback: misclassification error
            return 1.0 - np.max(p)

    # --------- predikce ----------
    def _descend(self, node, x):
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _predict_one(self, node, x):
        leaf = self._descend(node, x)
        return leaf.prediction
    
    def print_tree(self, feature_names=None, class_names=None, show_impurity=True):
        if self.root is None:
            print("Tree is empty. Call train() first.")
            return
        if feature_names is None:
            feature_names = [f"X[{j}]" for j in range(self.n_features_)]
        self._print_node(self.root, feature_names, class_names, show_impurity, prefix="")

    def _print_node(self, node, feature_names, class_names, show_impurity, prefix):
        # hlavička uzlu
        if node.is_leaf:
            # label a počty tříd
            total = int(node.class_counts.sum())
            pred = node.prediction
            label = class_names[pred] if class_names is not None else str(pred)
            info = f"LEAF: predict={label}, samples={total}, counts={node.class_counts.tolist()}"
            if show_impurity:
                imp = self._impurity_from_counts(node.class_counts)
                info += f", impurity={imp:.3f}"
            print(prefix + info)
            return

        name = feature_names[node.feature]
        info = f"[{name} <= {node.threshold:.4f}]"
        if show_impurity:
            imp = self._impurity_from_counts(node.class_counts)
            info += f"  (impurity={imp:.3f}, samples={int(node.class_counts.sum())})"
        print(prefix + info)

        # levá větev
        print(prefix + "├─ yes")
        self._print_node(node.left, feature_names, class_names, show_impurity, prefix + "│   ")
        # pravá větev
        print(prefix + "└─ no ")
        self._print_node(node.right, feature_names, class_names, show_impurity, prefix + "    ")


if __name__ == "__main__":
    data = load_iris()
    X = data.data.astype(float)
    Y = data.target.astype(int)
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    split = int(0.8 * len(Y))
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], Y[train_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    clf = decision_tree(
        metric="gini",
        max_depth=3,            
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0
    ).train(X_train, y_train)
    
    print("\n=== TREE STRUCTURE ===")
    clf.print_tree(feature_names=feature_names, class_names=class_names, show_impurity=True)

    y_pred = clf.test(X_test)
    acc = (y_pred == y_test).mean()
    print(f"Accuracy: {acc:.3f}")

    probs = clf.predict_proba(X_test[:5])
    for i in range(5):
        print(f"sample {i}: true={class_names[y_test[i]]}, pred={class_names[y_pred[i]]}, proba={np.round(probs[i], 3)}")
