# 🔥🫁 BreathAlert BR

**Queimadas e saúde respiratória no Brasil — uma investigação de dados (2016–2025)**

Global Solution 2026 · Ciência de Dados

Cruzamento entre os focos de queimada do **INPE/BDQueimadas** e as internações
respiratórias do **SUS (DATASUS/SIH)**, por estado e mês, para entender **se**, **onde**,
**quando** e **com que defasagem** o fogo se reflete na saúde da população brasileira.

> O "**BR**" no nome deixa explícito o recorte do estudo: dados públicos brasileiros,
> de todos os 27 estados, ao longo de uma década.

---

## 🔎 Principais achados

- **Os calendários não coincidem.** Queimadas picam na seca (ago–out); internações
  respiratórias picam no inverno (abr–jun). No mesmo mês, a correlação é nula.
- **O sinal existe, mas é defasado.** Removida a sazonalidade (anomalias) e aplicada
  defasagem temporal, surge uma associação positiva que cresce com o atraso — coerente
  com o efeito cumulativo da fumaça sobre a saúde.
- **O efeito tem endereço.** É mais nítido no **Sudeste**, onde a grande população permite
  medir as variações com menos ruído. Lá, o teste de **Mann-Whitney** confirma (p = 0,026)
  que meses de queimada intensa são seguidos, ~3 meses depois, por internações acima do normal.

## 📁 Estrutura do repositório

```
breathalert-br/
├── analise/
│   └── projeto.ipynb                   # notebook completo (análise + gráficos + conclusões)
├── dados/
│   ├── queimadas_consolidado.csv       # focos por estado/ano/mês (INPE)        · 3.240 linhas
│   ├── datasus_consolidado.csv         # internações respiratórias (SUS)        · 3.240 linhas
│   ├── populacao_consolidado.csv       # população por UF/ano (IBGE)            ·   270 linhas
│   └── queimadas_bioma_consolidado.csv # focos agregados por bioma (INPE)       ·     6 linhas
├── site/
│   └── index.html                      # site de apresentação (storytelling dos dados)
└── pitch/
    ├── BreathAlert_Pitch.pdf           # pitch completo (PDF)
    └── BreathAlert_Pitch.pptx          # apresentação editável
```

## ▶️ Como reproduzir

**Rodar a análise** (os dados consolidados já estão em `dados/`):

```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook analise/projeto.ipynb   # Run All
```

**Ver o site:** abra `site/index.html` no navegador (ou publique via GitHub Pages).
A última seção traz os links de **download** dos dados — ajuste os caminhos no início
do `<script>` (`DL_BASE`, `REPO_URL`) para apontar para este repositório.

## 📊 Fontes dos dados

| Base | Fonte |
|---|---|
| Focos de queimada | [INPE / BDQueimadas](https://terrabrasilis.dpi.inpe.br/queimadas/) |
| Internações respiratórias | [DATASUS / SIH-SUS](https://datasus.saude.gov.br/) |
| População estimada | [IBGE / SIDRA tabela 6579](https://sidra.ibge.gov.br/tabela/6579) |

## ⚠️ Limitações

Análise em escala mensal e estadual (grosseira para um fenômeno local); achados por
recorte são exploratórios; o ano de 2025 pode estar levemente subestimado pela
defasagem de processamento do SIH.

## 🚀 Próximos passos

Este estudo é o começo, não o ponto final. As próximas frentes miram em fechar a
lacuna de granularidade e transformar a análise em uma ferramenta de uso real:

1. **Dados municipais e diários** — descer da escala estadual/mensal para onde a fumaça de fato age.
2. **Modelo preditivo sazonal** — Prophet ou LSTM para antecipar os picos de risco a partir da seca.
3. **Qualidade do ar (PM2.5)** — integrar material particulado (sensor MERRA-2 da NASA e estações locais).
4. **Índice IRSQ por município** — Índice de Risco à Saúde por Queimadas, com níveis de alerta.
5. **API pública e dashboard** — disponibilizar o índice como serviço para secretarias de saúde e Defesa Civil.
6. **Validação em campo** — parceria com secretarias municipais para calibrar os alertas com dados reais.

---

*Projeto acadêmico desenvolvido para a Global Solution FIAP 2026.*
