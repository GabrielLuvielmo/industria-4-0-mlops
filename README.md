# MLOps - Detecção de Falhas Industriais

Projeto reprodutível para classificação desbalanceada (99,5% operacional / 0,5% falha), avaliação de métricas e ajuste financeiro de threshold.

## Estrutura

- `src/main.py`: geração dos dados, divisão estratificada, pipeline de pré-processamento + modelo, métricas, curva ROC e otimização do threshold.
- `data/sensor_data.csv`: dados sintéticos gerados pelo script.
- `figures/`: gráficos produzidos automaticamente.
- `outputs/`: resultados numéricos em CSV/JSON.
- `.vscode/`: recomendações/configurações básicas do VSCode.
- `requirements.txt`: dependências.

## Execução no VSCode

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/main.py
```

No macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Prevenção de Data Leakage

A divisão `train_test_split(..., stratify=y)` ocorre antes do `StandardScaler`. O scaler é ajustado somente no treinamento porque está dentro de um `Pipeline` que recebe apenas `X_train` no `fit`. O conjunto de teste só entra no fluxo em `predict_proba`/avaliação final.

## Cenário financeiro

O projeto usa `FN_COST = R$ 50.000` para uma falha grave não detectada e `FP_COST = R$ 5.000` para um falso alarme. O threshold ótimo é selecionado no conjunto de validação. O teste permanece intocado até a avaliação final. Em produção, a calibração deve ser repetida com dados históricos e uma função de custo validada pela operação.
