#!/usr/bin/env python3
"""
Análise de evasão escolar (IFCE Campus Acaraú) - Regressão Logística.

Convertido do notebook Colab original para script standalone
(sem dependência de google.colab / upload interativo).

Uso:
    python evasao_analise.py caminho/para/arquivo.xlsx
    python evasao_analise.py caminho/para/arquivo.xlsx --skip-geocoding
"""

import argparse
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

ENDERECO_CAMPUS = "Av. Des. Armando de Souza Louzada - Sítio - Buriti, Acaraú - CE, 62580-000"
COORD_CAMPUS_FALLBACK = (-2.885, -40.118)

COLS_DROP_INICIAL = [
    "matricula", "nome", "ultimo_evento_de_matricula", "sit_ult_per_letivo",
    "periodo_atual", "per_let_inigresso", "cpf", "documento_de_estrangeiro",
    "num_de_identidade", "orgao_exp_", "data_expedicao", "estado_ident_",
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
        df["nascimento"] = pd.to_datetime(df["nascimento"], errors="coerce")
        hoje = date.today()
        df["idade"] = df["nascimento"].apply(
            lambda x: hoje.year - x.year - ((hoje.month, hoje.day) < (x.month, x.day))
            if pd.notnull(x) else np.nan
        )
        df.drop(columns=["nascimento"], inplace=True)

    if "situacao_matricula" in df.columns:
        df["situacao_matricula"] = df["situacao_matricula"].apply(
            lambda x: 0 if x in NAO_EVASAO else (1 if x in EVASAO else x) if pd.notnull(x) else np.nan
        )

    if "cota" in df.columns:
        df["cota"] = df["cota"].apply(simplifica_cota)

    if "renda_familiar_per_capita_sig" in df.columns:
        df["renda_familiar_per_capita_sig"] = (
            df["renda_familiar_per_capita_sig"]
            .astype(str)
            .replace({"<": "menor", ">": "maior", "=": ""}, regex=True)
        )
    if "renda_familiar" in df.columns:
        df["renda_familiar"] = (
            df["renda_familiar"]
            .astype(str)
            .replace(
                {"<=": "menor_igual", ">=": "maior_igual", "<": "menor", ">": "maior", "=": ""},
                regex=True,
            )
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

        cache_coords = {}
        get_distancia = _make_get_distancia(coord_campus, cache_coords, geolocator)
        print("Calculando distâncias... (~1.1s por endereço novo, cache acelera repetidos)")
        df["distancia_campus_km"] = df.apply(get_distancia, axis=1)

    if "situacao_matricula" in df.columns:
        df["situacao_matricula"] = pd.to_numeric(df["situacao_matricula"], errors="coerce")
        df.dropna(subset=["situacao_matricula"], inplace=True)

    return df


def prepare_ml_dataset(df: pd.DataFrame):
    df_ml = df.copy()
    df_ml.drop(columns=[c for c in COLS_DROP_FINAL if c in df_ml.columns], inplace=True, errors="ignore")

    if "idade" in df_ml.columns:
        df_ml["idade"] = df_ml["idade"].fillna(df_ml["idade"].median())
    if "distancia_campus_km" in df_ml.columns:
        df_ml["distancia_campus_km"] = df_ml["distancia_campus_km"].fillna(df_ml["distancia_campus_km"].median())

    categorical_cols = df_ml.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df_ml[col] = df_ml[col].fillna("Desconhecido")

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        le_dict[col] = le

    df_ml = df_ml.dropna()

    X = df_ml.drop(columns=["situacao_matricula"])
    y = df_ml["situacao_matricula"]
    return X, y, le_dict


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    coef = model.coef_[0]
    odds_ratio = np.exp(coef)
    features = X.columns

    print("\n--- RESULTADOS DA REGRESSÃO LOGÍSTICA ---\n")
    print("Coeficientes e Odds Ratios:")
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

    print("\n--- AVALIAÇÃO DO MODELO ---")
    y_pred = model.predict(X_test_scaled)
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusão:")
    print(confusion_matrix(y_test, y_pred))

    return model, scaler


def predict_examples(model, scaler, X: pd.DataFrame, n: int = 3):
    print("\n--- Exemplo de previsão para novos alunos ---")
    novos_dados = X.iloc[:n].copy()
    novos_dados_scaled = scaler.transform(novos_dados)
    previsoes = model.predict(novos_dados_scaled)
    probabilidades = model.predict_proba(novos_dados_scaled)[:, 1]

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
    X, y, _ = prepare_ml_dataset(df)
    model, scaler = train_and_evaluate(X, y)
    predict_examples(model, scaler, X)


if __name__ == "__main__":
    main()
