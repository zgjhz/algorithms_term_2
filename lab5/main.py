import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QFileDialog
)

# === Вспомогательные функции для кластеризации ===

def pearson_distance_to_center(points: pd.DataFrame, center: pd.Series) -> pd.Series:
    X = points.values
    c = center.values
    X_centered = X - X.mean(axis=1, keepdims=True)
    c_centered = c - c.mean()
    row_norms = np.linalg.norm(X_centered, axis=1)
    center_norm = np.linalg.norm(c_centered)
    dots = X_centered.dot(c_centered)
    denom = row_norms * center_norm
    corr = np.zeros_like(dots)
    nonzero = denom > 0
    corr[nonzero] = dots[nonzero] / denom[nonzero]
    corr = np.clip(corr, -1.0, 1.0)
    return pd.Series(1 - corr, index=points.index)


def forel(data: pd.DataFrame, radius: float) -> pd.Series:
    clusters = []
    remaining = data.copy()
    while not remaining.empty:
        center = remaining.sample(1).iloc[0]
        moved = True
        while moved:
            dists = pearson_distance_to_center(remaining, center)
            in_sphere = remaining[dists <= radius]
            new_center = in_sphere.mean()
            moved = not new_center.equals(center)
            center = new_center
        members = remaining.index[dists <= radius].tolist()
        clusters.append(members)
        remaining = remaining.drop(index=members)
    labels = pd.Series(index=data.index, dtype=int)
    for i, members in enumerate(clusters):
        labels.loc[members] = i
    return labels


def compactness(data: pd.DataFrame, labels: pd.Series) -> float:
    total = 0.0
    for lbl in labels.unique():
        pts = data[labels == lbl]
        center = pts.mean()
        total += pearson_distance_to_center(pts, center).sum()
    return total / len(data)


def sequential_reduction(df: pd.DataFrame, features: list[str], radius: float, min_feats: int):
    history = []
    current = features.copy()
    iteration = 1
    while len(current) > min_feats:
        comp_scores = {}
        for feat in current:
            temp_feats = [f for f in current if f != feat] + ['ID1','ID2']
            lbls = forel(df[temp_feats], radius)
            comp_scores[feat] = compactness(df[temp_feats], lbls)
        drop = min(comp_scores, key=comp_scores.get)
        remaining = len(current) - 1
        comp_value = comp_scores[drop]
        history.append({'Итерация': iteration,
                        'Удалён признак': drop,
                        'Осталось признаков': remaining,
                        'Compactness': comp_value})
        current.remove(drop)
        iteration += 1
    return current, pd.DataFrame(history)


def visualize_clusters(data: pd.DataFrame, labels: pd.Series, title: str):
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)
    plt.figure(figsize=(5, 4))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(label='Cluster')
    plt.show()

class ClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Кластеризация FOREL с отбором признаков и обезличиванием")
        self.resize(900, 800)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        btn_load = QPushButton("Загрузить CSV...")
        btn_load.clicked.connect(self.load_csv)
        layout.addWidget(btn_load)

        hl1 = QHBoxLayout()
        hl1.addWidget(QLabel("Радиус:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1.0)
        self.radius_spin.setValue(0.5)
        hl1.addWidget(self.radius_spin)
        layout.addLayout(hl1)

        hl2 = QHBoxLayout()
        hl2.addWidget(QLabel("Оставить ≥ признаков:"))
        self.min_feat_spin = QSpinBox()
        self.min_feat_spin.setRange(2, 100)
        self.min_feat_spin.setValue(15)
        hl2.addWidget(self.min_feat_spin)
        layout.addLayout(hl2)

        btn_run = QPushButton("Запустить кластеризацию и обезличивание")
        btn_run.clicked.connect(self.on_run)
        layout.addWidget(btn_run)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        self.setLayout(layout)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "", "CSV файлы (*.csv)")
        if path:
            self.csv_path = path
            self.output.append(f"Загружен набор данных: {path}")

    def on_run(self):
        df_orig = pd.read_csv(self.csv_path)
        df_orig = df_orig.head(1000)
        np.random.seed(0)
        df = df_orig.copy()
        df['ID1'] = np.random.rand(len(df))
        df['ID2'] = np.random.rand(len(df))
        feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ['ID1','ID2']]
        radius = self.radius_spin.value()
        min_feats = self.min_feat_spin.value()

        # Кластеризация до обезличивания
        data_before = df[feats + ['ID1','ID2']]
        labels_before = forel(data_before, radius)
        visualize_clusters(df[feats], labels_before, 'Кластеры до обезличивания')
        sizes_before = labels_before.value_counts().to_dict()
        comp_before = compactness(data_before, labels_before)
        self.output.append(f"До обезличивания: размеры кластеров {sizes_before}, compactness = {comp_before:.6e}")

        # Отбор признаков
        selected_feats, history = sequential_reduction(df, feats, radius, min_feats)
        self.output.append(f"Отобраны признаки ({len(selected_feats)}): {selected_feats}")

        # История удаления признаков
        hist_df = history.copy()
        hist_df['Compactness'] = hist_df['Compactness'].apply(lambda x: f"{x:.6e}")
        self.output.append("\nИстория удаления признаков:")
        self.output.append(hist_df.to_string(index=False))

        # Полное обезличивание
        df_anon = df[selected_feats].copy()
        for col in selected_feats:
            try:
                df_anon[col] = pd.qcut(df_anon[col], 4, labels=False, duplicates='drop')
            except:
                pass
        for col in selected_feats:
            df_anon[col] = pd.cut(df_anon[col], 4, labels=False, include_lowest=True)
        for col in selected_feats:
            std = df_anon[col].std()
            df_anon[col] += np.random.normal(0, std * 0.05, len(df_anon))
        idx = df_anon.index.tolist()
        np.random.shuffle(idx)
        for i in range(0, len(idx), 5):
            grp = idx[i:i+5]
            mean_vals = df_anon.loc[grp].mean()
            for col in selected_feats:
                df_anon.loc[grp, col] = mean_vals[col]

        # Кластеризация после обезличивания
        labels_after = forel(df_anon, radius)
        visualize_clusters(df_anon, labels_after, 'Кластеры после обезличивания')
        sizes_after = labels_after.value_counts().to_dict()
        comp_after = compactness(df_anon, labels_after)
        self.output.append(f"После обезличивания: размеры кластеров {sizes_after}, compactness = {comp_after:.6e}")
        self.output.append("Готово.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClusteringApp()
    win.show()
    sys.exit(app.exec_())