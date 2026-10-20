from pathlib import Path
import sys
import pandas as pd

RAIZ_PROJETO = next(pasta for pasta in [Path.cwd(), *Path.cwd().parents] if (pasta / 'utils.py').exists())
sys.path.insert(0, str(RAIZ_PROJETO))
from utils import CAMINHO_DADOS

URL_DATASET = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
CAMINHO_DADOS.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(URL_DATASET)
df.to_csv(CAMINHO_DADOS, index=False)
print(f'Dados salvos em: {CAMINHO_DADOS}')
print(f'Dimensoes: {df.shape[0]} linhas e {df.shape[1]} colunas')
df.head()