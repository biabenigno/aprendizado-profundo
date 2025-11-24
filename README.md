# Deteção de Uso de Capacete em Motociclistas com CNN

## Sobre o Projeto

Este projeto tem como objetivo desenvolver um modelo de **Rede Neural Convolucional (CNN)** para a classificação automática de segurança no trânsito, especificamente para identificar se motociclistas estão utilizando o capacete de segurança, para a cadeira de Ciência dos Dados do professor Caio Moura.

A automação dessa tarefa é de alta relevância para sistemas de **cidades inteligentes** e fiscalização, contribuindo para a prevenção de acidentes fatais. O modelo é treinado para classificar imagens nas classes: **"Com Capacete"** e **"Sem Capacete"**.

## Arquitetura do Modelo (CNN)

A arquitetura da Rede Neural Convolucional (CNN) utilizada é do tipo **Sequencial** e foi desenhada para extrair características visuais relevantes das imagens.

| Etapa | Componentes Principais | Função |
| :--- | :--- | :--- |
| **Pré-Processamento** | `Rescaling(1./255)` | Normalização dos valores dos pixels de $0-255$ para $0-1$. |
| **Regularização** | `RandomFlip`, `RandomRotation`, `RandomZoom` | **Data Augmentation** (Aumento de Dados) para gerar novas variações de imagens e evitar o overfitting. |
| **Extração de Features** | 3 Camadas `Conv2D` com `ReLU` | Extrai características de diferentes níveis (bordas, texturas, formas). |
| **Redução de Dimensionalidade** | 3 Camadas `MaxPooling2D` | Reduz a dimensão espacial das *feature maps*. |
| **Classificação** | `Flatten`, `Dense` (128, `ReLU`), `Dropout(0.5)` | Camadas finais para classificação e regularização. |
| **Saída** | `Dense` (`softmax`) | Camada final que gera a probabilidade de cada uma das classes. |

## Dataset

* **Nome:** On Vehicle Helmet Detection Dataset.
* **Divisão:** O conjunto de dados foi dividido em $80\%$ para treino e $20\%$ para validação.
* **Dimensão da Imagem:** Todas as imagens foram padronizadas para **$128 \times 128$ pixels**.
* **Processamento:** As imagens são processadas em lotes (*batches*) de $32$.

## Resultados Obtidos

O treinamento foi realizado por **15 épocas**. O modelo demonstrou um desempenho satisfatório na base de validação, indicando boa capacidade de generalização.

### Desempenho Final (Validação)

| Métrica | Valor |
| :--- | :--- |
| **Acurácia** | Próxima a **87%** |
| **Convergência** | As curvas de treino e validação seguiram tendências similares. |

> O F1-Score equilibrado reforça a robustez do classificador. A matriz de confusão permitiu verificar que o modelo possui uma alta taxa de acerto na distinção entre as classes.
