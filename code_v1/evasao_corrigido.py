#!/usr/bin/env python3
"""
Análise de evasão escolar (IFCE Campus Acaraú) - Regressão Logística.

Convertido do notebook Colab original para script standalone
(sem dependência de google.colab / upload interativo).

Versão corrigida (evasao_corrigido.py). Diferenças em relação a evasao.py:
  1. Renda com valores faltantes agora vira NaN de verdade (não a string "nan").
  2. Variáveis categóricas nominais passam por One-Hot Encoding (não LabelEncoder),
     tornando os coeficientes / odds ratios interpretáveis.
  3. Pré-processamento (imputação, encoding, scaling) fica dentro de um Pipeline
     ajustado SOMENTE no treino -> elimina o data leakage.
  4. Remoção automática de variáveis irrelevantes (constantes, quase todas nulas,
     alta cardinalidade tipo identificador).
  5. Os "novos alunos" do exemplo agora vêm do conjunto de TESTE, não do treino.
  6. Avisos de não-convergência do modelo deixam de ser silenciados.
  7. Abordagem híbrida: regressão logística balanceada para INTERPRETAR (odds ratios)
     + HistGradientBoosting balanceado para PREVER (melhor ROC-AUC/F1). As distâncias
     via geopy usam cache em disco (geocode_cache.json) -> 1ª execução lenta, demais rápidas.

Uso:
    python evasao_corrigido.py caminho/para/arquivo.xlsx
    python evasao_corrigido.py caminho/para/arquivo.xlsx --skip-geocoding
"""

import argparse
import json
import os
import re
import time
import unicodedata
import warnings
from datetime import date

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Silencia apenas o ruído esperado do geopy/pandas, mas NÃO esconde
# avisos de convergência da regressão logística.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="geopy")

ENDERECO_CAMPUS = "Av. Des. Armando de Souza Louzada - Sítio - Buriti, Acaraú - CE, 62580-000"
COORD_CAMPUS_FALLBACK = (-2.885, -40.118)

# Cache em disco: endereço buscado -> coordenada (lat, lon). Mantém a precisão do
# endereço completo, mas só a 1ª execução paga o custo (~1.1s por endereço NOVO);
# execuções seguintes leem do cache e ficam praticamente instantâneas.
CACHE_DISTANCIAS = "geocode_cache.json"

COLS_DROP_INICIAL = [
    "matricula", "nome", "ultimo_evento_de_matricula", "sit_ult_per_letivo",
    "periodo_atual", "per_let_inigresso", "cpf", "documento_de_estrangeiro",
    "num_de_identidade", "orgao_exp", "data_expedicao", "estado_ident",
    "nome_do_pai", "nome_da_mae", "no_pasta",
]

COLS_DROP_FINAL = [
    "endereco", "numero", "complemento", "bairro", "cep", "cidade",
    "percentual_frequencia", "grupo_etnico", "ultima_presenca",
]

NAO_EVASAO = ["Concluído", "Concludente", "Matriculado", "Formado", "Projeto Final (Concludente)"]
EVASAO = [
    "Abandono", "Cancelado Compulsoriamente", "Cancelado Voluntariamente",
    "Trancado", "Transferido Externo", "Transferido Interno", "Falecido",
]

TARGET = "situacao_matricula"

# Limiares para remoção de variáveis irrelevantes.
MAX_FRAC_NULOS = 0.60        # colunas com mais de 60% de nulos são descartadas
MAX_FRAC_CARDINALIDADE = 0.50  # colunas-texto com >50% de valores únicos (tipo ID) são descartadas
MAX_CATEGORIAS = 50          # colunas-texto com mais de 50 categorias distintas são descartadas


def clean_col_name(col: str) -> str:
    """Remove acentos, cedilha, pontuação e normaliza nome de coluna."""
    col = str(col)
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("utf-8")
    col = col.lower().strip()
    col = col.replace(".", "").replace(",", "").replace(" ", "_")
    return col


def simplifica_cota(cota):
    if pd.isnull(cota):
        return cota
    cota = str(cota).strip()
    if cota == "Ampla Concorrência":
        return "AC"
    if cota.startswith("L"):
        match = re.match(r"(L\d+)", cota)
        return match.group(1) if match else cota
    return cota


def _limpa_simbolos(serie: pd.Series, mapa: dict) -> pd.Series:
    """Substitui símbolos (<, >, =, ...) preservando NaN como NaN (e não 'nan')."""
    def _f(v):
        if pd.isnull(v):
            return np.nan
        s = str(v)
        for k, r in mapa.items():
            s = s.replace(k, r)
        return s.strip()

    return serie.map(_f)


def _load_cache(path: str) -> dict:
    """Carrega o cache de geocoding do disco (endereço -> coordenada ou None)."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: (tuple(v) if v else None) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _save_cache(path: str, cache: dict) -> None:
    """Persiste o cache de geocoding no disco para reuso nas próximas execuções."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: (list(v) if v else None) for k, v in cache.items()}, f, ensure_ascii=False)
    except Exception:
        pass


def load_data(path: str) -> pd.DataFrame:
    print(f"Lendo arquivo: {path}")
    df = pd.read_excel(path)
    print("Colunas do dataset:")
    print(df.columns.tolist())
    print("\nDimensões (linhas, colunas):", df.shape)
    print("\nContagem de valores nulos por coluna:")
    print(df.isnull().sum())
    return df


def _make_get_distancia(coord_campus, cache_coords, geolocator):
    def get_distancia(row):
        rua = str(row.get("endereco", "")).strip()
        bairro = str(row.get("bairro", "")).strip()
        cep = str(row.get("cep", "")).strip().replace("-", "").replace(".", "")
        cidade = str(row.get("cidade", "")).strip()

        rua = "" if rua.lower() in ["nan", "none", ""] else rua
        bairro = "" if bairro.lower() in ["nan", "none", ""] else bairro
        cep = "" if cep.lower() in ["nan", "none", ""] else cep
        cidade = "" if cidade.lower() in ["nan", "none", ""] else cidade

        buscas = []
        if cep:
            buscas.append(f"{cep}, Brazil")
        if rua and cidade:
            buscas.append(f"{rua}, {cidade}, Ceará, Brazil")
        if bairro and cidade:
            buscas.append(f"{bairro}, {cidade}, Ceará, Brazil")
        if cidade:
            buscas.append(f"{cidade}, Ceará, Brazil")

        if not buscas:
            return np.nan

        for endereco_busca in buscas:
            if endereco_busca in cache_coords:
                if cache_coords[endereco_busca]:
                    return geodesic(coord_campus, cache_coords[endereco_busca]).km
                continue

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    location = geolocator.geocode(endereco_busca, timeout=5)
                    time.sleep(1.1)
                    if location:
                        coord_aluno = (location.latitude, location.longitude)
                        cache_coords[endereco_busca] = coord_aluno
                        return geodesic(coord_campus, coord_aluno).km
                    cache_coords[endereco_busca] = None
                    break
                except (GeocoderTimedOut, GeocoderServiceError):
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    cache_coords[endereco_busca] = None
                    break
                except Exception:
                    cache_coords[endereco_busca] = None
                    break
        return np.nan

    return get_distancia


def clean_data(df: pd.DataFrame, skip_geocoding: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    df.drop(columns=COLS_DROP_INICIAL, inplace=True, errors="ignore")

    if "nascimento" in df.columns:
        df["nascimento"] = pd.to_datetime(df["nascimento"], errors="coerce", dayfirst=True)
        hoje = date.today()
        df["idade"] = df["nascimento"].apply(
            lambda x: hoje.year - x.year - ((hoje.month, hoje.day) < (x.month, x.day))
            if pd.notnull(x) else np.nan
        )
        df.drop(columns=["nascimento"], inplace=True)

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].apply(
            lambda x: 0 if x in NAO_EVASAO else (1 if x in EVASAO else x) if pd.notnull(x) else np.nan
        )

    if "cota" in df.columns:
        df["cota"] = df["cota"].apply(simplifica_cota)

    # Renda: substitui símbolos MAS preserva faltantes como NaN (antes viravam a string "nan").
    if "renda_familiar_per_capita_sig" in df.columns:
        df["renda_familiar_per_capita_sig"] = _limpa_simbolos(
            df["renda_familiar_per_capita_sig"],
            {"<": "menor", ">": "maior", "=": ""},
        )
    if "renda_familiar" in df.columns:
        df["renda_familiar"] = _limpa_simbolos(
            df["renda_familiar"],
            {"<=": "menor_igual", ">=": "maior_igual", "<": "menor", ">": "maior", "=": ""},
        )

    tem_distancia = "distancia_campus_km" in df.columns and not df["distancia_campus_km"].isnull().all()
    if not skip_geocoding and not tem_distancia:
        geolocator = Nominatim(user_agent="ifce_acarau_pesquisa_evasao_v5")
        try:
            location_campus = geolocator.geocode(ENDERECO_CAMPUS, timeout=10)
            coord_campus = (
                (location_campus.latitude, location_campus.longitude)
                if location_campus else COORD_CAMPUS_FALLBACK
            )
        except Exception:
            coord_campus = COORD_CAMPUS_FALLBACK

        cache_coords = _load_cache(CACHE_DISTANCIAS)
        n_cache_inicial = len(cache_coords)
        get_distancia = _make_get_distancia(coord_campus, cache_coords, geolocator)
        print(
            f"Calculando distâncias por endereço completo... "
            f"(cache: {n_cache_inicial} endereços conhecidos; cada endereço NOVO ~1.1s)"
        )
        df["distancia_campus_km"] = df.apply(get_distancia, axis=1)
        _save_cache(CACHE_DISTANCIAS, cache_coords)
        print(f"Cache salvo em '{CACHE_DISTANCIAS}' ({len(cache_coords)} endereços). Próxima execução será quase instantânea.")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df.dropna(subset=[TARGET], inplace=True)

    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove variáveis irrelevantes: constantes, quase todas nulas e do tipo identificador.

    O alvo (situacao_matricula) nunca é removido.
    """
    df = df.copy()
    n = len(df)
    removidas = {}

    for col in list(df.columns):
        if col == TARGET:
            continue

        n_unicos = df[col].nunique(dropna=True)
        frac_nulos = df[col].isnull().mean()

        # Coluna constante (ou totalmente vazia) -> nenhum poder preditivo.
        if n_unicos <= 1:
            removidas[col] = "constante/vazia"
            df.drop(columns=[col], inplace=True)
            continue

        # Coluna majoritariamente ausente.
        if frac_nulos > MAX_FRAC_NULOS:
            removidas[col] = f"{frac_nulos:.0%} nulos"
            df.drop(columns=[col], inplace=True)
            continue

        # Coluna-texto com cardinalidade de identificador (quase um valor por linha).
        # (usa is_numeric_dtype: no pandas 3.0 texto vira dtype 'str', não 'object')
        if not pd.api.types.is_numeric_dtype(df[col]):
            if n > 0 and (n_unicos / n) > MAX_FRAC_CARDINALIDADE:
                removidas[col] = f"alta cardinalidade ({n_unicos} únicos)"
                df.drop(columns=[col], inplace=True)
                continue
            if n_unicos > MAX_CATEGORIAS:
                removidas[col] = f"muitas categorias ({n_unicos})"
                df.drop(columns=[col], inplace=True)
                continue

    if removidas:
        print("\n--- Variáveis irrelevantes removidas ---")
        for col, motivo in removidas.items():
            print(f"  - {col}: {motivo}")
    else:
        print("\nNenhuma variável irrelevante detectada.")

    return df


def prepare_ml_dataset(df: pd.DataFrame):
    """Seleciona X e y. NÃO faz imputação/encoding/scaling aqui:
    isso é responsabilidade do Pipeline, ajustado só no treino (evita leakage)."""
    df_ml = df.copy()
    df_ml.drop(columns=[c for c in COLS_DROP_FINAL if c in df_ml.columns], inplace=True, errors="ignore")
    df_ml = drop_irrelevant_columns(df_ml)

    X = df_ml.drop(columns=[TARGET])
    y = df_ml[TARGET].astype(int)
    return X, y


def _make_preprocessor(X: pd.DataFrame, dense: bool = False) -> ColumnTransformer:
    """Imputação + scaling (numéricas) e imputação + one-hot (categóricas).

    dense=True força a saída densa do one-hot (exigido pelo HistGradientBoosting).
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]  # robusto a dtype 'str' (pandas 3.0)

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconhecido")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])


def build_logreg_pipeline(X: pd.DataFrame) -> Pipeline:
    """Modelo INTERPRETÁVEL: regressão logística balanceada -> coeficientes / odds ratios.

    class_weight='balanced' compensa o desbalanceamento das classes (mais evasão do
    que não-evasão), equilibrando o recall entre as duas.
    """
    return Pipeline([
        ("pre", _make_preprocessor(X, dense=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])


def build_gb_pipeline(X: pd.DataFrame) -> Pipeline:
    """Modelo de PREVISÃO: HistGradientBoosting balanceado -> melhor desempenho (ROC-AUC/F1)."""
    return Pipeline([
        ("pre", _make_preprocessor(X, dense=True)),
        ("clf", HistGradientBoostingClassifier(random_state=42, class_weight="balanced")),
    ])


def _relatorio_odds_ratios(logreg: Pipeline):
    """Imprime coeficientes e odds ratios da regressão logística (interpretação)."""
    clf = logreg.named_steps["clf"]
    features = logreg.named_steps["pre"].get_feature_names_out()
    coef = clf.coef_[0]
    odds_ratio = np.exp(coef)

    if np.any(np.asarray(clf.n_iter_) >= 1000):
        print("\n[AVISO] A regressão logística pode não ter convergido (max_iter atingido).")

    print("\n=== INTERPRETAÇÃO — REGRESSÃO LOGÍSTICA (Odds Ratios) ===\n")
    resultados = pd.DataFrame({
        "Variável": features,
        "Coeficiente": coef,
        "Odds Ratio": odds_ratio,
    }).sort_values(by="Odds Ratio", ascending=False)
    for _, row in resultados.iterrows():
        print(f"{row['Variável']}: Coeficiente = {row['Coeficiente']:.4f}, Odds Ratio = {row['Odds Ratio']:.4f}")

    print("\n--- TOP 5 VARIÁVEIS MAIS IMPORTANTES (Maior impacto absoluto) ---")
    importante = np.argsort(np.abs(coef))[-5:][::-1]
    for idx in importante:
        print(f"{features[idx]}: Coeficiente = {coef[idx]:.4f}")


def _avalia(nome: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """Imprime relatório de classificação, matriz de confusão e ROC-AUC no teste."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n>>> {nome}")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")


def _importancia_permutacao(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n: int = 10):
    """Importância por permutação (queda no ROC-AUC) por variável ORIGINAL do HistGB."""
    print("\n--- HistGradientBoosting: TOP variáveis (importância por permutação, queda no ROC-AUC) ---")
    r = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    for idx in np.argsort(r.importances_mean)[::-1][:n]:
        print(f"{X_test.columns[idx]}: {r.importances_mean[idx]:.4f} +/- {r.importances_std[idx]:.4f}")


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1) Modelo interpretável (odds ratios) — todo pré-processamento ajustado SÓ no treino.
    logreg = build_logreg_pipeline(X)
    logreg.fit(X_train, y_train)
    _relatorio_odds_ratios(logreg)

    # 2) Modelo de previsão (melhor desempenho).
    gb = build_gb_pipeline(X)
    gb.fit(X_train, y_train)

    print("\n=== DESEMPENHO NO CONJUNTO DE TESTE ===")
    _avalia("Regressão Logística (interpretável)", logreg, X_test, y_test)
    _avalia("HistGradientBoosting (previsão)", gb, X_test, y_test)
    _importancia_permutacao(gb, X_test, y_test)

    # Retorna o modelo de previsão (melhor) para os exemplos.
    return gb, X_test


def predict_examples(model, X_test: pd.DataFrame, n: int = 3):
    print("\n--- Exemplo de previsão (HistGradientBoosting) para alunos do teste ---")
    novos_dados = X_test.iloc[:n].copy()
    previsoes = model.predict(novos_dados)
    probabilidades = model.predict_proba(novos_dados)[:, 1]

    for i, (prev, prob) in enumerate(zip(previsoes, probabilidades)):
        status = "Evasão" if prev == 1 else "Não evasão"
        print(f"Aluno {i + 1}: Previsão = {status}, Probabilidade de evasão = {prob:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Análise de evasão escolar - regressão logística")
    parser.add_argument("arquivo", help="Caminho do arquivo .xlsx com os dados dos alunos")
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Pula o cálculo de distância via geocoding (mais rápido, exige coluna já preenchida ou aceita NaN)",
    )
    args = parser.parse_args()

    df = load_data(args.arquivo)
    df = clean_data(df, skip_geocoding=args.skip_geocoding)
    X, y = prepare_ml_dataset(df)
    model, X_test = train_and_evaluate(X, y)
    predict_examples(model, X_test)


if __name__ == "__main__":
    main()
