import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.datasets import load_breast_cancer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QTextEdit, QComboBox
)

# === Clustering utilities ===

def pearson_distance_to_center(points: pd.DataFrame, center: pd.Series) -> pd.Series:
    return points.apply(lambda row: 1 - pearsonr(row, center)[0], axis=1)

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

def sequential_reduction(df: pd.DataFrame, features: list[str],
                         radius: float, min_feats: int) -> tuple[list[str], pd.DataFrame]:
    history = []
    current = features.copy()
    while len(current) > min_feats:
        comp_scores = {}
        for feat in current:
            temp = [f for f in current if f != feat] + ['ID1', 'ID2']
            lbls = forel(df[temp], radius)
            comp_scores[feat] = compactness(df[temp], lbls)
        drop = min(comp_scores, key=comp_scores.get)
        history.append({'removed': drop,
                        'remaining_count': len(current)-1 + 2,
                        'compactness': comp_scores[drop]})
        current.remove(drop)
    return current, pd.DataFrame(history)

# === PyQt5 GUI ===

class ClusteringApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOREL Clustering Interface")
        self.resize(800, 600)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        # Dataset selector
        ds_layout = QHBoxLayout()
        ds_layout.addWidget(QLabel("Dataset:"))
        self.ds_combo = QComboBox()
        self.ds_combo.addItem("Breast Cancer (sklearn)")
        ds_layout.addWidget(self.ds_combo)
        layout.addLayout(ds_layout)

        # Radius input
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("FOREL Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1.0)
        self.radius_spin.setSingleStep(0.01)
        self.radius_spin.setValue(0.5)
        r_layout.addWidget(self.radius_spin)
        layout.addLayout(r_layout)

        # Feature count input
        f_layout = QHBoxLayout()
        f_layout.addWidget(QLabel("Keep ≥ features:"))
        self.min_feat_spin = QSpinBox()
        self.min_feat_spin.setRange(2, 30)
        self.min_feat_spin.setValue(15)
        f_layout.addWidget(self.min_feat_spin)
        layout.addLayout(f_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Clustering")
        self.run_btn.clicked.connect(self.on_run)
        btn_layout.addWidget(self.run_btn)
        self.anon_btn = QPushButton("Advanced Anonymization + Clustering")
        self.anon_btn.clicked.connect(self.on_anon)
        btn_layout.addWidget(self.anon_btn)
        layout.addLayout(btn_layout)

        # Output area
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def on_run(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        np.random.seed(0)
        df['ID1'] = np.random.rand(len(df))
        df['ID2'] = np.random.rand(len(df))

        radius = self.radius_spin.value()
        min_feats = self.min_feat_spin.value()
        feats = data.feature_names.tolist()

        labels1 = forel(df[feats + ['ID1','ID2']], radius)
        comp1 = compactness(df[feats + ['ID1','ID2']], labels1)

        selected, hist_df = sequential_reduction(df, feats, radius, min_feats)
        labels2 = forel(df[selected + ['ID1','ID2']], radius)
        comp2 = compactness(df[selected + ['ID1','ID2']], labels2)

        df_deid = df[selected]
        labels3 = forel(df_deid, radius)
        comp3 = compactness(df_deid, labels3)

        out = []
        out.append(f"Initial compactness: {comp1:.6e}")
        out.append(f"After feature selection ({len(selected)} feats): {comp2:.6e}")
        out.append(f"After de-identification: {comp3:.6e}")
        out.append("\nFeature removal history:")
        out.append(hist_df.to_string(index=False))
        self.output.setPlainText("\n".join(out))

    def on_anon(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        np.random.seed(0)
        df['ID1'] = np.random.rand(len(df))
        df['ID2'] = np.random.rand(len(df))

        radius = self.radius_spin.value()
        min_feats = self.min_feat_spin.value()
        feats = data.feature_names.tolist()

        # Feature selection
        selected, _ = sequential_reduction(df, feats, radius, min_feats)
        df_sel = df[selected]

        # 3 anonymization techniques:
        results = []
        # 1) Drop IDs (already dropped)
        df_drop = df_sel.copy()
        lbl_drop = forel(df_drop, radius)
        comp_drop = compactness(df_drop, lbl_drop)
        results.append(("Drop IDs", comp_drop))

        # 2) Generalization (quartile binning)
        df_gen = df_sel.copy()
        for col in df_gen.columns:
            try:
                df_gen[col] = pd.qcut(df_gen[col], q=4, labels=False, duplicates='drop')
            except Exception:
                pass
        lbl_gen = forel(df_gen, radius)
        comp_gen = compactness(df_gen, lbl_gen)
        results.append(("Generalization (quartiles)", comp_gen))

        # 3) Noise addition (5% of std)
        df_noise = df_sel.copy()
        for col in df_noise.columns:
            std = df_noise[col].std()
            df_noise[col] += np.random.normal(0, std*0.05, size=len(df_noise))
        lbl_noise = forel(df_noise, radius)
        comp_noise = compactness(df_noise, lbl_noise)
        results.append(("Noise addition (5% std)", comp_noise))

        out = ["Advanced Anonymization Results:"]
        for name, comp in results:
            out.append(f"{name}: compactness = {comp:.6e}")
        self.output.setPlainText("\n".join(out))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClusteringApp()
    win.show()
    sys.exit(app.exec_())
