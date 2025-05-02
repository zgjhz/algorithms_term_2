import sys
import os
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QPushButton,
    QMessageBox, QTableWidget, QTableWidgetItem
)

# === здесь ваши функции из исходника ===

def introduce_missing_df(df: pd.DataFrame, pct: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_miss = df.copy()
    for col in df_miss.columns:
        mask = rng.random(len(df_miss)) < (pct / 100)
        df_miss.loc[mask, col] = np.nan
    return df_miss

def impute_hot_deck_strat(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    # обработка цены
    price_missing = df_out.index[df_out['Цена приёма (руб)'].isna()]
    for i in price_missing:
        doc = df_out.at[i, 'Врач']
        sym = df_out.at[i, 'Симптомы']
        mask = (
            (df_out['Врач'] == doc) &
            (df_out['Симптомы'] == sym) &
            df_out['Цена приёма (руб)'].notna()
        )
        pool = df_out.loc[mask, 'Цена приёма (руб)'].values
        if len(pool) == 0:
            pool = df_out.loc[
                (df_out['Врач'] == doc) & df_out['Цена приёма (руб)'].notna(),
                'Цена приёма (руб)'
            ].values
        if len(pool) == 0:
            pool = df_out['Цена приёма (руб)'].dropna().values
        df_out.at[i, 'Цена приёма (руб)'] = rng.choice(pool)

    stratify = {
        'Симптомы':                'Врач',
        'Врач':                    None,
        'Дата посещения врача':    'Врач',
        'Анализы':                 'Врач',
        'Дата получения анализов': 'Врач',
        'ФИО':                     'Страна',
        'Паспортые_данные':        'Страна',
        'СНИЛС':                   'Страна',
    }
    for col in df_out.columns:
        if col == 'Цена приёма (руб)':
            continue
        miss_idx = df_out.index[df_out[col].isna()]
        if miss_idx.empty:
            continue
        group_col = stratify.get(col, None)
        if group_col:
            for i in miss_idx:
                key = df_out.at[i, group_col]
                pool = df_out.loc[
                    (df_out[group_col] == key) &
                    df_out[col].notna(),
                    col
                ].values
                if len(pool) == 0:
                    pool = df_out[col].dropna().values
                df_out.at[i, col] = rng.choice(pool)
        else:
            pool = df_out[col].dropna().values
            fills = rng.choice(pool, size=len(miss_idx))
            df_out.loc[miss_idx, col] = fills

    return df_out

def impute_spline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Восстанавливает пропуски в числовых столбцах
    сплайн-интерполяцией 3-го порядка по индексу.
    """
    df_out = df.copy()
    # Все числовые поля
    numeric = df_out.select_dtypes(include='number').columns
    # Применяем interpolate с методом 'spline'
    df_out[numeric] = (
        df_out[numeric]
        .interpolate(method='spline', order=3, limit_direction='both')
    )
    return df_out

def impute_locf_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    df_out['__dt_visit'] = pd.to_datetime(
        df_out['Дата посещения врача'],
        format="%Y-%m-%dT%H:%M:%S%z",
        errors='coerce'
    )
    df_out.sort_values(['ФИО', '__dt_visit'], inplace=True)
    df_out = (
        df_out
        .groupby('ФИО', group_keys=False)
        .apply(lambda g: g.ffill())
    )
    df_out['Дата посещения врача'] = (
        df_out['__dt_visit']
        .dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
    )
    df_out.drop(columns='__dt_visit', inplace=True)
    return df_out

def evaluate_imputation(
    df_orig: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame
) -> pd.DataFrame:
    records = []
    numeric_cols = df_orig.select_dtypes(include='number').columns
    for col in df_orig.columns:
        miss_idx = df_missing.index[df_missing[col].isna()]
        if miss_idx.empty:
            continue
        valid_idx = miss_idx.intersection(df_imputed.index)
        if valid_idx.empty:
            continue
        tv = df_orig.loc[valid_idx, col]
        pv = df_imputed.loc[valid_idx, col]
        if col in numeric_cols:
            sum_rel = (tv.subtract(pv).abs().divide(tv.abs()).sum())
            records.append({
                'column':      col,
                'sum_rel_err': sum_rel,
                'error_rate':  np.nan
            })
        else:
            mismatches = (tv.astype(str) != pv.astype(str)).sum()
            rate = mismatches / len(valid_idx)
            records.append({
                'column':      col,
                'sum_rel_err': np.nan,
                'error_rate':  rate
            })
    total = sum(r['sum_rel_err'] for r in records if pd.notna(r['sum_rel_err']))
    records.append({
        'column':      'TOTAL',
        'sum_rel_err': total,
        'error_rate':  np.nan
    })
    return pd.DataFrame(records)

# === GUI ===

class ImputationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Импутация данных — Лабораторная работа 4")
        self.resize(900, 600)

        # --- виджеты ---
        lbl_file = QLabel("Выберите датасет:")
        self.csv_combo = QComboBox()
        self.csv_combo.addItems(["small.csv", "medium.csv", "large.csv"])

        lbl_pct = QLabel("Процент пропусков:")
        self.pct_spin = QSpinBox()
        self.pct_spin.setRange(0, 100)
        self.pct_spin.setValue(10)
        self.pct_spin.setSuffix(" %")

        self.degrade_btn = QPushButton("Испортить датасет")
        self.impute_btn  = QPushButton("Восстановить датасет")

        # Табличка для вывода результата
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # --- раскладка ---
        top_layout = QHBoxLayout()
        top_layout.addWidget(lbl_file)
        top_layout.addWidget(self.csv_combo)
        top_layout.addSpacing(20)
        top_layout.addWidget(lbl_pct)
        top_layout.addWidget(self.pct_spin)
        top_layout.addStretch()
        top_layout.addWidget(self.degrade_btn)
        top_layout.addWidget(self.impute_btn)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.table)

        # --- сигналы ---
        self.degrade_btn.clicked.connect(self.degrade_dataset)
        self.impute_btn.clicked.connect(self.impute_dataset)

        # в памяти
        self.df_orig = None
        self.df_miss = None

    def degrade_dataset(self):
        fname = f"lab4/{self.csv_combo.currentText()}"
        if not os.path.exists(fname):
            QMessageBox.warning(self, "Ошибка", f"Не найден файл {fname}")
            return
        df = pd.read_csv(fname, encoding='utf-8-sig')
        pct = self.pct_spin.value()
        df_miss = introduce_missing_df(df, pct, seed=42)
        self.df_orig = df
        self.df_miss = df_miss

        # Очистим таблицу
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Столбец", "Пропущено ячеек"])

        na_counts = df_miss.isna().sum()
        # заполняем
        for i, (col, cnt) in enumerate(na_counts.items()):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(col))
            self.table.setItem(i, 1, QTableWidgetItem(str(int(cnt))))

    def impute_dataset(self):
        if self.df_orig is None or self.df_miss is None:
            QMessageBox.warning(self, "Ошибка", "Сначала испорьте датасет")
            return

        # Раньше было 2 метода — добавляем третий 'spline'
        methods = {
            'hot_deck': impute_hot_deck_strat,
            'locf':     impute_locf_by_patient,
            'spline':   impute_spline
        }

        reports = []
        for name, fn in methods.items():
            df_imp = fn(self.df_miss.copy())
            rep = evaluate_imputation(self.df_orig, self.df_miss, df_imp)
            rep['method'] = name
            reports.append(rep)

        df_all = pd.concat(reports, ignore_index=True)
        pivot = df_all.pivot_table(
            index='column',
            columns='method',
            values=['sum_rel_err','error_rate']
        ).round(3)

        # Флэттеним MultiIndex колонок: ('sum_rel_err','hot_deck') → 'sum_rel_err_hot_deck'
        pivot.columns = [
            f"{metric}_{method}"
            for metric, method in pivot.columns
        ]
        pivot.reset_index(inplace=True)

        # заполняем QTableWidget
        self.table.clear()
        self.table.setRowCount(len(pivot))
        self.table.setColumnCount(len(pivot.columns))
        self.table.setHorizontalHeaderLabels(pivot.columns.tolist())

        for i, row in pivot.iterrows():
            for j, value in enumerate(row):
                self.table.setItem(
                    i, j,
                    QTableWidgetItem("" if pd.isna(value) else str(value))
                )
        self.table.resizeColumnsToContents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ImputationApp()
    win.show()
    sys.exit(app.exec_())
