import sys
import os
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QPushButton,
    QMessageBox, QTableWidget, QTableWidgetItem
)

# === Imputation methods ===

def hot_deck_impute(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    for col in df_out.columns:
        mask = df_out[col].isna()
        if not mask.any():
            continue
        pool = df_out[col].dropna().to_numpy()
        df_out.loc[mask, col] = rng.choice(pool, size=mask.sum())
    return df_out


def locf_impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(method='ffill')


def spline_impute(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    numeric_cols = df_out.select_dtypes(include='number').columns
    for col in numeric_cols:
        arr = df_out[col].to_numpy(dtype=float)
        x = np.arange(len(arr))
        mask_known = ~np.isnan(arr)
        if mask_known.sum() < 4:
            df_out[col] = pd.Series(arr).interpolate().to_numpy()
            continue
        cs = CubicSpline(x[mask_known], arr[mask_known], axis=0)
        arr[~mask_known] = cs(x[~mask_known])
        df_out[col] = arr
    return df_out


def compute_delta_metric(
    df_complete: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for col in df_complete.columns:
        idx = df_missing.index[df_missing[col].isna()]
        if len(idx) == 0:
            continue
        orig = df_complete.loc[idx, col]
        imp = df_imputed.loc[idx, col]
        if pd.api.types.is_numeric_dtype(orig):
            rel = (orig - imp).abs() / orig.abs()
        else:
            codes, uniques = pd.factorize(df_complete[col].astype(str))
            mapping = {u: c+1 for c, u in enumerate(uniques)}
            orig_num = orig.astype(str).map(mapping)
            imp_num  = imp.astype(str).map(mapping)
            rel = (orig_num - imp_num).abs() / orig_num
        delta = rel.sum() * 100
        rows.append({'Столбец': col, 'ΔM': delta})
    total = sum(r['ΔM'] for r in rows)
    rows.append({'Столбец': 'TOTAL', 'ΔM': total})
    return pd.DataFrame(rows)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {'Среднее': {}, 'Медиана': {}, 'Мода': {}}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            stats['Среднее'][col] = series.mean()
            stats['Медиана'][col] = series.median()
        else:
            stats['Среднее'][col] = np.nan
            stats['Медиана'][col] = np.nan
        mode_vals = series.mode(dropna=True)
        stats['Мода'][col] = mode_vals.iloc[0] if not mode_vals.empty else np.nan
    return pd.DataFrame(stats)

# === Debug runner saving into one Excel file ===

def debug_run_all_detailed(
    csv_dir: str = 'lab4',
    pct_list: list = [3, 5, 10, 20, 30],
    output_file: str = 'debug_results.xlsx'
):
    """
    Для всех small/medium/large и pct из pct_list прогоняет методы
    hot_deck, locf, spline,
    собирает:
      - base_stats: эталонная статистика по полным наблюдениям
      - missing_stats: статистика по датасету с пропусками
      - imp_stats: статистика после импутации
      - metrics: ΔM метрики
    и сохраняет всё в один Excel-файл, каждый набор на отдельном листе.
    """
    methods = {'hot_deck': hot_deck_impute,
               'locf': locf_impute,
               'spline': spline_impute}

    # Списки для объединения
    list_base = []
    list_missing = []
    list_imp = []
    list_metric = []

    datasets = ['small.csv', 'medium.csv', 'large.csv']
    for fname in datasets:
        path = os.path.join(csv_dir, fname)
        if not os.path.exists(path):
            print(f"Warning: file {path} not found, skipping.")
            continue
        ds = os.path.splitext(fname)[0]
        df = pd.read_csv(path, encoding='utf-8-sig')
        complete = df.dropna()

        # Эталонная статистика
        base = descriptive_stats(complete).reset_index().rename(columns={'index': 'Столбец'})
        base['dataset'] = ds
        list_base.append(base)

        for pct in pct_list:
            rng = np.random.default_rng(42)
            mask = rng.random(complete.shape) < pct/100
            df_missing = complete.mask(mask)

            miss = descriptive_stats(df_missing).reset_index().rename(columns={'index': 'Столбец'})
            miss['dataset'] = ds
            miss['pct_missing'] = pct
            list_missing.append(miss)

            for m_name, m_fn in methods.items():
                df_imp = m_fn(df_missing.copy())
                imp = descriptive_stats(df_imp).reset_index().rename(columns={'index': 'Столбец'})
                imp['dataset'] = ds
                imp['pct_missing'] = pct
                imp['method'] = m_name
                list_imp.append(imp)

                metric = compute_delta_metric(complete, df_missing, df_imp)
                metric['dataset'] = ds
                metric['pct_missing'] = pct
                metric['method'] = m_name
                list_metric.append(metric)

    # Объединение
    df_base = pd.concat(list_base, ignore_index=True)
    df_missing = pd.concat(list_missing, ignore_index=True)
    df_imp = pd.concat(list_imp, ignore_index=True)
    df_metric = pd.concat(list_metric, ignore_index=True)

    # Сохранение в один Excel-файл
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_base.to_excel(writer, sheet_name='base_stats', index=False)
        df_missing.to_excel(writer, sheet_name='missing_stats', index=False)
        df_imp.to_excel(writer, sheet_name='imputed_stats', index=False)
        df_metric.to_excel(writer, sheet_name='metrics', index=False)

    print(f"All results saved to {output_file}")

# === GUI ===

class ImputationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Импутация данных — Лабораторная работа 4")
        self.resize(900, 700)

        lbl_file = QLabel("Выберите датасет:")
        self.csv_combo = QComboBox()
        self.csv_combo.addItems(['small.csv','medium.csv','large.csv'])

        lbl_pct = QLabel("Процент пропусков:")
        self.pct_spin = QSpinBox(); self.pct_spin.setRange(0,100); self.pct_spin.setValue(10); self.pct_spin.setSuffix(" %")

        lbl_method = QLabel("Метод импутации:")
        self.method_combo = QComboBox(); self.method_combo.addItems(['hot_deck','locf','spline'])

        self.degrade_btn = QPushButton("Испортить датасет")
        self.impute_btn = QPushButton("Применить метод и оценить")
        self.debug_btn = QPushButton("Запустить debug все")

        self.table_stats = QTableWidget(); self.table_stats.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_metric = QTableWidget(); self.table_metric.setEditTriggers(QTableWidget.NoEditTriggers)

        top = QHBoxLayout()
        top.addWidget(lbl_file); top.addWidget(self.csv_combo); top.addSpacing(20)
        top.addWidget(lbl_pct); top.addWidget(self.pct_spin); top.addSpacing(20)
        top.addWidget(lbl_method); top.addWidget(self.method_combo); top.addStretch()
        top.addWidget(self.degrade_btn); top.addWidget(self.impute_btn); top.addWidget(self.debug_btn)

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(QLabel("Статистика до/после импутации:")); main.addWidget(self.table_stats)
        main.addWidget(QLabel("Метрика ΔM:")); main.addWidget(self.table_metric)

        self.degrade_btn.clicked.connect(self.degrade_dataset)
        self.impute_btn.clicked.connect(self.impute_dataset)
        self.debug_btn.clicked.connect(self.run_debug_all)

        self.df_orig = None; self.df_miss = None

    def degrade_dataset(self):
        fname = f"lab4/{self.csv_combo.currentText()}"
        if not os.path.exists(fname): QMessageBox.warning(self, "Ошибка", f"Не найден файл {fname}"); return
        df = pd.read_csv(fname, encoding='utf-8-sig'); pct = self.pct_spin.value()
        rng = np.random.default_rng(42); mask = rng.random(df.shape) < pct/100
        self.df_orig = df; self.df_miss = df.mask(mask)

        stats_before = descriptive_stats(self.df_miss).add_suffix(' (до)')
        self._show_stats(stats_before)
        self.table_metric.clear()

    def impute_dataset(self):
        if self.df_miss is None or self.df_orig is None:
            QMessageBox.warning(self, "Ошибка", "Сначала испортьте датасет"); return
        m = self.method_combo.currentText()
        fn = {'hot_deck':hot_deck_impute,'locf':locf_impute,'spline':spline_impute}[m]
        imp = fn(self.df_miss.copy())
        before = descriptive_stats(self.df_miss).add_suffix(' (до)')
        after  = descriptive_stats(imp).add_suffix(' (после)')
        stats = pd.concat([before, after], axis=1); self._show_stats(stats)
        metric = compute_delta_metric(self.df_orig, self.df_miss, imp); self._show_metric(metric)

    def run_debug_all(self):
        debug_run_all_detailed()
        QMessageBox.information(self, "Debug Complete", "Results saved to debug_results.xlsx")

    def _show_stats(self, df: pd.DataFrame):
        self.table_stats.clear(); dfr = df.reset_index().rename(columns={'index':'Столбец'})
        hdr = [str(c) for c in dfr.columns]; self.table_stats.setColumnCount(len(hdr)); self.table_stats.setRowCount(len(dfr))
        self.table_stats.setHorizontalHeaderLabels(hdr)
        for i, row in dfr.iterrows():
            for j, val in enumerate(row):
                self.table_stats.setItem(i,j,QTableWidgetItem('' if pd.isna(val) else str(val)))
        self.table_stats.resizeColumnsToContents()

    def _show_metric(self, df: pd.DataFrame):
        self.table_metric.clear(); hdr = list(df.columns); self.table_metric.setColumnCount(len(hdr)); self.table_metric.setRowCount(len(df))
        self.table_metric.setHorizontalHeaderLabels(hdr)
        for i, row in df.iterrows():
            for j, val in enumerate(row):
                self.table_metric.setItem(i,j,QTableWidgetItem('' if pd.isna(val) else str(val)))
        self.table_metric.resizeColumnsToContents()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImputationApp()
    win.show()
    sys.exit(app.exec_())
