import pandas as pd
import numpy as np

# === 1) Введение пропусков ===

def introduce_missing_df(
    df: pd.DataFrame,
    pct: float,
    seed: int = 42
) -> pd.DataFrame:
    """
    Случайно превращает pct% значений в каждом столбце df в NaN.
    """
    rng = np.random.default_rng(seed)
    df_miss = df.copy()
    for col in df_miss.columns:
        mask = rng.random(len(df_miss)) < (pct / 100)
        df_miss.loc[mask, col] = np.nan
    return df_miss

# === 2) Улучшенный Stratified Hot-Deck ===

def impute_hot_deck_strat(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Stratified Hot-deck по «логическим» группам:
     – для «Цена приёма (руб)» сначала по (Врач, Симптомы), затем по Врач, затем глобально
     – для визит-полей по Врач
     – для identity-полей (ФИО, Паспорт, СНИЛС) по Страна
     – для остальных — глобально
    """
    rng = np.random.default_rng(seed)
    df_out = df.copy()

    # список индексов, где было NaN в цене
    price_missing = df_out.index[df_out['Цена приёма (руб)'].isna()]

    # сначала заполним цену отдельно
    for i in price_missing:
        doc = df_out.at[i, 'Врач']
        sym = df_out.at[i, 'Симптомы']
        # 1) попытка (Врач, Симптомы)
        mask = (
            (df_out['Врач'] == doc) &
            (df_out['Симптомы'] == sym) &
            df_out['Цена приёма (руб)'].notna()
        )
        pool = df_out.loc[mask, 'Цена приёма (руб)'].values
        # 2) fallback: по Врачу
        if len(pool) == 0:
            mask2 = (df_out['Врач'] == doc) & df_out['Цена приёма (руб)'].notna()
            pool = df_out.loc[mask2, 'Цена приёма (руб)'].values
        # 3) fallback: глобально
        if len(pool) == 0:
            pool = df_out['Цена приёма (руб)'].dropna().values
        df_out.at[i, 'Цена приёма (руб)'] = rng.choice(pool)

    # mapping других столбцов на группу
    stratify = {
        'Симптомы':                'Врач',
        'Врач':                    None,
        'Дата посещения врача':    'Врач',
        'Анализы':                 'Врач',
        'Дата получения анализов': 'Врач',
        'ФИО':                     'Страна',
        'Паспортые данные':        'Страна',
        'СНИЛС':                   'Страна',
        # остальные — None (глобально)
    }

    for col in df_out.columns:
        if col == 'Цена приёма (руб)':
            continue  # уже обработали
        miss_idx = df_out.index[df_out[col].isna()]
        if len(miss_idx) == 0:
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

# === 3) LOCF по пациенту ===

def impute_locf_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """
    Last-Observation-Carried-Forward и Backfill внутри каждой группы 'ФИО',
    чтобы не «тянуть» данные одного пациента к другому.
    """
    df_out = df.copy()
    # временная колонка для сортировки
    df_out['__dt_visit'] = pd.to_datetime(
        df_out['Дата посещения врача'],
        format="%Y-%m-%dT%H:%M:%S%z",
        errors='coerce'
    )
    df_out.sort_values(['ФИО', '__dt_visit'], inplace=True)

    df_out = (
        df_out
        .groupby('ФИО', group_keys=False)
        .apply(lambda g: g.ffill().bfill())
    )

    # вернуть дату визита в строковый формат
    df_out['Дата посещения врача'] = (
        df_out['__dt_visit']
        .dt.strftime("%Y-%m-%dT%H:%M:%S+03:00")
    )
    df_out.drop(columns='__dt_visit', inplace=True)
    return df_out

# === 4) Оценка для числовых и категориальных ===

def evaluate_imputation(
    df_orig: pd.DataFrame,
    df_missing: pd.DataFrame,
    df_imputed: pd.DataFrame
) -> pd.DataFrame:
    """
    Для каждого столбца:
      – если числовой: sum_rel_err = Σ|ai - âi|/|ai|
      – иначе:        error_rate = (# mismatches)/(# recovered)
    Использует только те индексы, где пропуск восстановлен (т. е. присутствует в df_imputed).
    Возвращает DataFrame с колонками: column, sum_rel_err, error_rate.
    """
    records = []
    numeric_cols = df_orig.select_dtypes(include='number').columns

    for col in df_orig.columns:
        # все индексы, где в df_missing стоит NaN
        miss_idx = df_missing.index[df_missing[col].isna()]
        if len(miss_idx) == 0:
            continue

        # оставляем только те индексы, что есть и в df_imputed
        valid_idx = miss_idx.intersection(df_imputed.index)
        if len(valid_idx) == 0:
            # ничего не восстановлено для этого столбца
            continue

        # Истинные и восстановленные значения по этим индексам
        tv = df_orig.loc[valid_idx, col]
        pv = df_imputed.loc[valid_idx, col]

        if col in numeric_cols:
            # Σ |ai - âi| / |ai|
            sum_rel = (tv.subtract(pv).abs()
                            .divide(tv.abs())
                            .sum())
            records.append({
                'column':      col,
                'sum_rel_err': sum_rel,
                'error_rate':  np.nan
            })
        else:
            # доля mismatches среди тех, что реально восстановлены
            tv_s = tv.astype(str)
            pv_s = pv.astype(str)
            mismatches = (tv_s != pv_s).sum()
            rate = mismatches / len(valid_idx)
            records.append({
                'column':      col,
                'sum_rel_err': np.nan,
                'error_rate':  rate
            })

    # Общий TOTAL по числовым sum_rel_err
    total = sum(r['sum_rel_err'] for r in records if pd.notna(r['sum_rel_err']))
    records.append({
        'column':      'TOTAL',
        'sum_rel_err': total,
        'error_rate':  np.nan
    })

    return pd.DataFrame(records)

# === 5) Интеграция в main ===

imputers = {
    'hot_deck': impute_hot_deck_strat,
    'locf':     impute_locf_by_patient
}

def main(csv_files, percentages=[3,5,10,20,30], seed=42):
    import os
    os.makedirs("lab4/imputed_variants", exist_ok=True)
    all_records = []

    for path in csv_files:
        df_orig = pd.read_csv(path, encoding='utf-8-sig')
        name = os.path.splitext(os.path.basename(path))[0]

        for pct in percentages:
            df_miss = introduce_missing_df(df_orig, pct, seed)

            for method, fn in imputers.items():
                # 1) применяем метод
                df_imp = fn(df_miss)

                # 2) сохраняем восстановленный датасет
                out_path = f"lab4/imputed_variants/{name}_missing_{pct}_{method}.csv"
                df_imp.to_csv(out_path, index=False, encoding='utf-8-sig')

                # 3) сразу перезагружаем, чтобы индексы были RangeIndex
                df_imp = pd.read_csv(out_path, encoding='utf-8-sig')

                # 4) оцениваем
                report = evaluate_imputation(df_orig, df_miss, df_imp)
                for _, row in report.iterrows():
                    all_records.append({
                        'dataset':     name,
                        'pct_missing': pct,
                        'method':      method,
                        'column':      row['column'],
                        'sum_rel_err': row['sum_rel_err'],
                        'error_rate':  row['error_rate']
                    })

    results = pd.DataFrame(all_records)
    pivot = results.pivot_table(
        index=['dataset','pct_missing','column'],
        columns='method',
        values=['sum_rel_err','error_rate']
    )
    print(pivot.round(3))
    return results


# === Пример запуска ===
if __name__ == "__main__":
    csv_files = ["lab4/medium.csv"]
    all_results = main(csv_files)
    all_results.to_csv("lab4/imputation_quality_summary.csv", index=False, encoding='utf-8-sig')
