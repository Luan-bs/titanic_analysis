from pathlib import Path
import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHO_DADOS = RAIZ_PROJETO / "data" / "raw" / "titanic_raw.csv"


def carregar_dados():
    """Carrega o dataset bruto do Titanic a partir da raiz do projeto."""
    if not CAMINHO_DADOS.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado em {CAMINHO_DADOS}. "
            "Execute o notebook notebooks/extract_data.ipynb primeiro."
        )
    return pd.read_csv(CAMINHO_DADOS)
