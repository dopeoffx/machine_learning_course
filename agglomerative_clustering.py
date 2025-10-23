import pandas as pd
import math
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

class AgglomerativeClustering():
    def __init__(self, dataframe, metrics = "Manhattan", linkage="single",  target_number_of_clusters=0) -> None:
        self.df = dataframe.select_dtypes(include=[np.number])
        
        self.rows = list(dataframe.itertuples(index=False, name=None))
        self.n = len(self.rows)
        self.alive = [True for i in range(self.n)]
        self.function = self.euclidean_metric if metrics=="Euclidean" else self.manhattan_metric
        self.linkage = "complete" if linkage=="complete" else "single"
        self.table = self.create_table()
        self.target_number_of_clusters = target_number_of_clusters 
        
        
        self.cluster_id = list(range(self.n))
        self.next_cluster_id = self.n
        
        self.cluster_size = {i: 1 for i in range(self.n)}
        self.members = {i: [i] for i in range(self.n)}
        
        self.linkage_rows = []
        
        
            
        
    def create_table(self):
        table = [0.0] * (self.n*(self.n-1)//2)
        k=0        
        for j in range(1, len(self.rows)):
            a = self.rows[j]
            for i in range(j):
                b = self.rows[i]
                table[k]= self.function(a, b)
                k += 1
        return table
    
    def _idx(self, i, j):
        if j==i:
            raise ValueError("diagonal not stored")
        if i>j:
            i,j = j, i
        return j*(j-1)//2 + i
    
    def set(self, i, j, val):
        if i == j:
            raise ValueError("no diagonal in condensed form")
        self.table[self._idx(i, j)] = val
        
    def get(self, i, j):
        if i == j:
            raise ValueError("no diagonal in condensed form")
        return self.table[self._idx(i, j)]
    
    def manhattan_metric(self, a, b):
        return sum(abs(x - y) for x, y in zip(a, b))
    
    def euclidean_metric(self, a, b):
        return  math.sqrt(sum(pow(x - y, 2) for x, y in zip(a, b)))
    
    def smallest_distance(self):
        best = (float("inf"), -1, -1)
        alive_idx = [i for i, a in enumerate(self.alive) if a]
        for a, i in enumerate(alive_idx):
            for j in alive_idx[a+1:]:
                d = self.get(i, j)
                if d < best[0]:
                    best = (d, i, j)
        return best
    
    def merge_clusters(self, A, B, dist):
        LINK = min if self.linkage == "single" else max
        
        idA = self.cluster_id[A]
        idB = self.cluster_id[B]
        new_size = self.cluster_size[idA] + self.cluster_size[idB]
        self.linkage_rows.append([idA, idB, float(dist), int(new_size)])
        
        new_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.cluster_size[new_id] = new_size
        self.members[new_id] = self.members[idA] + self.members[idB]
        self.cluster_id[A] = new_id
            
        for X in range(self.n):
            if not self.alive[X] or X == A or X == B:
                continue
            self.set(A, X, LINK(self.get(A, X), self.get(B, X)))

        self.alive[B] = False
        
    def fit_until_k(self, k):
        while sum(self.alive) > k:
            dist, i, j = self.smallest_distance()
            self.merge_clusters(i, j, dist)
        return [idx for idx,a in enumerate(self.alive) if a]
    

        
     # --- Výstupy pro vizualizace ---
    def get_linkage_matrix(self):
        """
        Vrátí linkage matici (numpy array shape (m,4)).
        Pozn.: U hierarchického slučování z n bodů vznikne n-1 řádků,
        ale pokud zastavíš dříve (na k klastrech), bude mít m = n - k.
        """
        if not self.linkage_rows:
            raise RuntimeError("Nejdřív zavolej fit_until_k(k), aby vznikla historie slučování.")
        return np.array(self.linkage_rows, dtype=float)

    def get_labels_at_k(self, k):
        """
        Po fit_until_k(k) vrátí pole délky n s čísly 0..k-1 (štítky klastrů).
        """
        if sum(self.alive) != k:
            raise RuntimeError("Nejdřív zavolej fit_until_k(k).")

        alive_ids = [self.cluster_id[i] for i, alive in enumerate(self.alive) if alive]
        alive_ids = list(dict.fromkeys(alive_ids))  # unikátní ve správném pořadí

        # mapování 'id klastru' -> 'pořadový label 0..k-1'
        id_to_label = {cid: lbl for lbl, cid in enumerate(alive_ids)}
        
        labels = [-1]*self.n
        for cid in alive_ids:
            for pt in self.members[cid]:
                labels[pt] = id_to_label[cid]
        return np.array(labels, dtype=int)

    def plot_dendrogram(self, truncate_mode=None, p=12, leaf_rotation=90):
        """
        Vykreslí dendrogram z aktuální linkage historie (po fit_until_k).
        truncate_mode např. 'lastp' pro zkrácení dlouhých dendrogramů.
        """
        Z = self.get_linkage_matrix()
        plt.figure()
        dendrogram(Z, truncate_mode=truncate_mode, p=p, leaf_rotation=leaf_rotation)
        plt.title("Dendrogram")
        plt.xlabel("Vzorky nebo sloučené klastry")
        plt.ylabel("Vzdálenost")
        plt.show()

    def plot_scatter(self, labels=None):
        """
        Scatter plot:
        - pokud máš přesně 2 numerické sloupce, použije je,
        - jinak provede PCA do 2D.
        """
        X = self.df.values
        if X.shape[1] == 2:
            X2 = X
            xlab, ylab = self.df.columns[:2]
        else:
            pca = PCA(n_components=2, random_state=0)
            X2 = pca.fit_transform(X)
            xlab, ylab = "PC1", "PC2"

        plt.figure()
        if labels is None:
            plt.scatter(X2[:, 0], X2[:, 1])
        else:
            for lbl in np.unique(labels):
                idx = labels == lbl
                plt.scatter(X2[idx, 0], X2[idx, 1], label=f"Cluster {lbl}")
            plt.legend()
        plt.title("Scatter plot (2D)")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
        
    
if __name__ == "__main__":
    df_full = pd.read_csv('clusters3.csv', sep=";")
    
    clust = AgglomerativeClustering(df_full, target_number_of_clusters=2, linkage="complete")
    clust.fit_until_k(3)
    
    labels = clust.get_labels_at_k(3)

    clust.fit_until_k(1)
    
    clust.plot_dendrogram()
    
    clust.plot_scatter(labels=labels)
    
