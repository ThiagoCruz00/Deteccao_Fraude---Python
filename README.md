# Detecção de fraude em transações

Este projeto transforma um notebook introdutório de classificação em uma solução mais próxima de um cenário real de prevenção a fraudes.

## Problema de negócio

Uma instituição financeira precisa identificar transações potencialmente fraudulentas para enviá-las à análise manual. O problema é desbalanceado: a grande maioria das transações é legítima e apenas uma pequena parcela é fraude.

O objetivo adotado neste projeto é **detectar pelo menos 80% das fraudes**, reduzindo, tanto quanto possível, o número de transações legítimas bloqueadas ou encaminhadas para análise. Esse objetivo é mais útil do que otimizar apenas a acurácia, porque um modelo que classifica tudo como “normal” pode ter acurácia alta e ainda assim não detectar fraudes.

> A solução não decide sozinha se uma transação é fraude. Ela cria uma fila de risco para investigação, o que é mais adequado para uma primeira etapa de operação financeira.

## O que havia de problemático no código original

O código original era um bom ponto de partida, mas apresentava problemas que poderiam produzir uma avaliação otimista ou uma decisão incorreta em produção:

1. O `StandardScaler` era ajustado antes da separação dos dados. Isso permite que informações estatísticas do teste influenciem o treinamento, caracterizando vazamento de dados.

1. O SMOTE era executado, mas os dados balanceados não eram usados no treinamento de nenhum modelo.

1. O vetor `y_probs` vinha do modelo de regressão logística treinado diretamente, mas era usado para criar previsões depois do treinamento de outro objeto chamado `pipeline`. A avaliação ficava inconsistente.

1. O limiar padrão de 0,50 era usado sem relação com o custo de perder uma fraude ou gerar falsos alertas.

1. A acurácia não era suficiente para avaliar o problema. Para fraude, recall, precisão, F1 e PR-AUC são métricas mais informativas.

1. A linha `from re import X` era desnecessária e deveria ser removida.

## Solução implementada

O arquivo [`deteccao_fraude_corrigida.py`](./deteccao_fraude_corrigida.py) implementa o seguinte fluxo:

1. Carrega o dataset público de transações com cartão de crédito.

1. Cria a variável `Amount_log` com `log1p`, reduzindo a assimetria do valor da transação.

1. Divide os dados de forma estratificada em treino, validação e teste.

1. Coloca a padronização e a regressão logística dentro de um `Pipeline`. Assim, o scaler é ajustado exclusivamente no conjunto de treino.

1. Usa `class_weight="balanced"` para dar maior importância à classe rara.

1. Escolhe o limiar de decisão no conjunto de validação. O critério é manter recall de pelo menos 80% e, entre os limiares possíveis, escolher a melhor precisão disponível.

1. Faz a avaliação final somente no conjunto de teste, que não participa do treinamento nem da escolha do limiar.

1. Exibe relatório de classificação, ROC-AUC, PR-AUC, matriz de confusão e curva Precisão-Recall.

1. Salva o pipeline e o limiar em `detector_fraude.joblib` para reutilização.

## Como executar no Google Colab

O arquivo principal para o Colab é [`deteccao_fraude_colab.ipynb`](./deteccao_fraude_colab.ipynb). Faça upload desse notebook em [Google Colab](https://colab.research.google.com/) e execute as células em ordem.

Crie uma célula e instale as dependências:

```python
!pip -q install pandas numpy scikit-learn matplotlib seaborn joblib
```

Se preferir executar o script diretamente, carregue o arquivo no Colab ou clone o repositório:

```python
from google.colab import files
files.upload()
```

Execute o script:

```python
%run deteccao_fraude_corrigida.py
```

O dataset será baixado automaticamente da URL pública usada no projeto. A execução gera gráficos, métricas no console e o arquivo `detector_fraude.joblib`.

Para baixar o modelo salvo:

```python
files.download("detector_fraude.joblib")
```

## Como usar o modelo em novas transações

As novas transações devem conter as mesmas colunas usadas no treinamento, exceto `Class`. O artefato salvo guarda o pipeline, o limiar escolhido e a lista de colunas:

```python
import joblib
import pandas as pd

artefato = joblib.load("detector_fraude.joblib")
novas_transacoes = pd.read_csv("novas_transacoes.csv")

probabilidade = artefato["modelo"].predict_proba(novas_transacoes)[:, 1]
decisao = (probabilidade >= artefato["limiar"]).astype(int)

resultado = novas_transacoes.copy()
resultado["probabilidade_fraude"] = probabilidade
resultado["enviar_para_analise"] = decisao
resultado.head()
```

Quando `enviar_para_analise` for igual a `1`, a transação deve ser encaminhada para revisão conforme as regras da instituição. O modelo não substitui regras antifraude, autenticação adicional ou investigação humana.

## Interpretação das métricas

**Recall de fraude** mede a proporção de fraudes reais que foram encontradas. Ele é priorizado neste projeto porque deixar uma fraude passar pode ser mais caro do que investigar uma transação legítima.

**Precisão** mede a proporção dos alertas que realmente são fraudes. Uma precisão baixa significa que a equipe de análise receberá muitos falsos positivos.

**PR-AUC** resume o desempenho na relação entre precisão e recall e costuma ser mais informativa do que acurácia em bases muito desbalanceadas.

**ROC-AUC** mede a capacidade geral de ordenar transações fraudulentas acima das normais em diferentes limiares. Ela não substitui a análise do limiar operacional escolhido.

## Limitações e próximos passos

O dataset é público e histórico. Portanto, o resultado não deve ser interpretado como desempenho de produção. Em um sistema real, seria necessário validar o modelo com divisão temporal, pois treinar com transações futuras para prever o passado não representa a operação real.

Também é recomendável calcular o limiar a partir de custos financeiros reais, acompanhar mudança no padrão de fraude, revisar o modelo periodicamente e proteger os dados sensíveis. Uma evolução natural seria comparar esta solução com modelos baseados em árvores, calibrar probabilidades e criar monitoramento de deriva das variáveis.

## Arquivos

| Arquivo | Descrição |
| --- | --- |
| `deteccao_fraude_colab.ipynb` | Notebook pronto para upload e execução no Google Colab |
| `deteccao_fraude_corrigida.py` | Script completo para execução no Google Colab |
| `detector_fraude.joblib` | Modelo salvo após a execução do script |
| `README.md` | Documentação do problema e da solução |

## Licença e fonte dos dados

Este projeto é educacional. O dataset utilizado é disponibilizado pelo TensorFlow em uma URL pública:

[1]: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv "Dataset público de transações com cartão de crédito"

[2]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html "Documentação do average precision do scikit-learn"

[3]: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html "Documentação de Pipeline do scikit-learn"

O uso em produção exige revisão de segurança, privacidade, governança e conformidade regulatória.
