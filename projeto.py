#%%
# Frameworks
import re
import os
import io
import unicodedata

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("default")
from scipy import stats

import urllib.request, json

#%%
# Mapeamento de UFs para estados e regiões
UFS = [
    ("11", "RO", "Rondônia",            "Norte"),
    ("12", "AC", "Acre",                "Norte"),
    ("13", "AM", "Amazonas",            "Norte"),
    ("14", "RR", "Roraima",             "Norte"),
    ("15", "PA", "Pará",                "Norte"),
    ("16", "AP", "Amapá",               "Norte"),
    ("17", "TO", "Tocantins",           "Norte"),
    ("21", "MA", "Maranhão",            "Nordeste"),
    ("22", "PI", "Piauí",               "Nordeste"),
    ("23", "CE", "Ceará",               "Nordeste"),
    ("24", "RN", "Rio Grande do Norte", "Nordeste"),
    ("25", "PB", "Paraíba",             "Nordeste"),
    ("26", "PE", "Pernambuco",          "Nordeste"),
    ("27", "AL", "Alagoas",             "Nordeste"),
    ("28", "SE", "Sergipe",             "Nordeste"),
    ("29", "BA", "Bahia",               "Nordeste"),
    ("31", "MG", "Minas Gerais",        "Sudeste"),
    ("32", "ES", "Espírito Santo",      "Sudeste"),
    ("33", "RJ", "Rio de Janeiro",      "Sudeste"),
    ("35", "SP", "São Paulo",           "Sudeste"),
    ("41", "PR", "Paraná",              "Sul"),
    ("42", "SC", "Santa Catarina",      "Sul"),
    ("43", "RS", "Rio Grande do Sul",   "Sul"),
    ("50", "MS", "Mato Grosso do Sul",  "Centro-Oeste"),
    ("51", "MT", "Mato Grosso",         "Centro-Oeste"),
    ("52", "GO", "Goiás",               "Centro-Oeste"),
    ("53", "DF", "Distrito Federal",    "Centro-Oeste"),
]

# Mapeamento de meses para converter os nomes em números
mapa_meses = {
        "Jan": 1,
        "Fev": 2,
        "Mar": 3,
        "Abr": 4,
        "Mai": 5,
        "Jun": 6,
        "Jul": 7,
        "Ago": 8,
        "Set": 9,
        "Out": 10,
        "Nov": 11,
        "Dez": 12
    }

# Sem acentos para facilitar buscas e comparações
def _sem_acento(txt):
    """'SÃO PAULO' -> 'sao paulo' (minúsculo, sem acento)."""
    txt = str(txt).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFKD", txt)
                   if not unicodedata.combining(c))


# --- Mapas derivados automaticamente da tabela UFS ---
mapa_uf         = {cod: sig for cod, sig, nome, reg in UFS}                 # IBGE -> sigla
nome_para_sigla = {_sem_acento(nome): sig for cod, sig, nome, reg in UFS}  # nome -> sigla
sigla_para_nome = {sig: nome for cod, sig, nome, reg in UFS}               # sigla -> nome
regiao_sigla    = {sig: reg for cod, sig, nome, reg in UFS}                 # sigla -> região (SUS)
regiao_map      = {nome.upper(): reg for cod, sig, nome, reg in UFS}        # NOME -> região (queimadas)
siglas          = set(mapa_uf.values())

# nome do mês derivado do mapa_meses (número -> "Jan")
nome_mes = {v: k for k, v in mapa_meses.items()}            


#%%
# Funcoes auxiliares
def estado_para_sigla(valor):
    """Aceita sigla ('SP') OU nome completo ('SÃO PAULO') -> sigla."""
    v = str(valor).strip()
    if v.upper() in siglas:
        return v.upper()
    return nome_para_sigla.get(_sem_acento(v))
 
def mes_para_numero(valor):
    """Aceita número (1..12) OU abreviação ('Jan') -> int 1..12."""
    v = str(valor).strip()
    if v.isdigit():
        n = int(v)
        return n if 1 <= n <= 12 else None
    return mapa_meses.get(v.capitalize())


# Função para processar os arquivos do SUS e extrair as internações por estado/mês
def processar_datasus(arquivo):

    print(f"Processando: {arquivo.name}")

    # Tentar ler o arquivo com diferentes codificações até encontrar uma que funcione
    linhas = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(arquivo, "r", encoding=enc, errors="replace") as f:
                linhas = f.readlines()
            break
        except Exception:
            continue
 
    idx_cabecalho = None
    for i, linha in enumerate(linhas):
        if ";" in linha and "Munic" in linha:
            idx_cabecalho = i
            break
    if idx_cabecalho is None:
        raise ValueError("Cabeçalho não encontrado no arquivo")

    # Ler o CSV a partir do índice do cabeçalho
    df = pd.read_csv(io.StringIO("".join(linhas[idx_cabecalho:])), sep=";", engine="python", on_bad_lines="skip")

    # remover coluna "Total" se existir 
    if "Total" in df.columns:
        df = df.drop(columns=["Total"])

    # identificar a coluna de localidade (1ª coluna) e extrair a UF do código IBGE
    col_local = df.columns[0]
    df["CodUF"]  = df[col_local].astype(str).str.extract(r"^\D*(\d{2})")
    df["Estado"] = df["CodUF"].map(mapa_uf)
    df = df.dropna(subset=["Estado"])

    # transformar formato largo -> longo
    df_long = df.melt(id_vars=["Estado"], var_name="Periodo", value_name="Internacoes")

    # filtrar apenas os períodos que correspondem a meses (ex: "Jan/2020", "Fev/2021", etc) 
    df_long = df_long[df_long["Periodo"].astype(str).str.contains(r"\d{4}/", na=False)]
    
    # limpar e converter a coluna de internações 
    df_long["Internacoes"] = df_long["Internacoes"].replace("-", 0)
    df_long["Internacoes"] = pd.to_numeric( df_long["Internacoes"], errors="coerce" ).fillna(0)

    # separar Ano e Mes
    df_long[["Ano", "MesTxt"]] = df_long["Periodo"].str.split("/", expand=True)
    df_long["Mes"] = df_long["MesTxt"].str.strip().map(mapa_meses)
    df_long["Ano"] = pd.to_numeric(df_long["Ano"], errors="coerce")
    df_long = df_long.dropna(subset=["Ano", "Mes"])

    # somar municípios por estado
    df_final = df_long.groupby( ["Estado", "Ano", "Mes"], as_index=False )["Internacoes"].sum()
    
    return df_final

def corr_lag(df, coluna_focos, coluna_inter, lag):
    """Correlação média (dentro de cada estado) entre focos de (t-lag) e
    internações de t. Calcular por estado controla a população."""
    rs = []
    for uf, g in df.groupby("Estado"):
        g = g.sort_values(["Ano", "Mes"])
        focos_def = g[coluna_focos].shift(lag)   # valor de 'lag' meses atrás
        inter = g[coluna_inter]
        m = focos_def.notna() & inter.notna()
        if m.sum() > 10 and focos_def[m].std() > 0:
            r, _ = stats.pearsonr(focos_def[m], inter[m])
            rs.append(r)
    return np.mean(rs)


#%%
# Configuração do caminho para os arquivos CSV
pasta = Path(r"C:\Users\saulo\Documents\5_Faculdade\3_Trabalhos\Global Solution\2_Semestre_1_Ano\1_dados_analise")

pasta_queimadas = os.path.join(pasta, "DataQueimadas")
pasta_queimadas = Path(pasta_queimadas)

pasta_sus = os.path.join(pasta, "DataSUS")
pasta_sus = Path(pasta_sus)

# Listar os arquivos CSV na pasta
arquivos_queimadas = list(pasta_queimadas.glob("*.csv"))
print(f"Total de arquivos de queimadas: {len(arquivos_queimadas)}")
# Listar os nomes dos arquivos encontrados
for arq_qmd in arquivos_queimadas:
    print(f" ->> {arq_qmd.name}")


#%%
# Carregar o primeiro arquivo CSV encontrado
amostra = pd.read_csv(arquivos_queimadas[0], sep=",", encoding="utf-8")
print("Colunas disponíveis na base bruta de queimadas:")
print(amostra.columns.tolist())
print("\nTipos de dado:")
print(amostra.dtypes)
print("\nPrimeiras linhas:")
amostra.head()


#%%
# Vericar se os arquivos são iguais
colunas_por_arquivo = {}
for arq in arquivos_queimadas:
    try:
        df_temp = pd.read_csv(arq, sep=",", encoding="utf-8", nrows=1)
        colunas_por_arquivo[arq.name] = df_temp.columns.tolist()
    except Exception as e:
        print(f"Erro ao ler {arq.name}: {e}")

referencia = list(colunas_por_arquivo.values())[0]
print(f"Colunas de referência ({list(colunas_por_arquivo.keys())[0]}):")
print(referencia)
print()
for nome, cols in colunas_por_arquivo.items():
    if cols != referencia:
        print(f"[!]  {nome} tem colunas DIFERENTES: {cols}")
    else:
        print(f"[✓]  {nome}")


#%%
# Descobrir o volume total de dados
resumo = []
for arq in arquivos_queimadas:
    try:
        df_resumo = pd.read_csv(arq, sep=",", encoding="utf-8")
        resumo.append({ "arquivo": arq.name,"linhas": len(df_resumo) })
    except Exception as e:
        print(f"Erro ao ler {arq.name}: {e}")

pd.DataFrame(resumo)

#%%
# Diagnóstico de qualidade da base de queimadas
total = len(amostra)
diagnostico = []
for col in amostra.columns:
    nulos  = amostra[col].isna().sum()
    unicos = amostra[col].nunique()
    if pd.api.types.is_numeric_dtype(amostra[col]):
        n999 = int((amostra[col] == -999).sum())
        minv = round(float(amostra[col].min()), 2)
        maxv = round(float(amostra[col].max()), 2)
    else:
        n999, minv, maxv = "-", "-", "-"
    diagnostico.append({
        "Coluna"    : col,
        "Tipo"      : str(amostra[col].dtype),
        "Cobertura%": round((1 - nulos / total) * 100, 1),
        "NaN"       : nulos,
        "Qtd -999"  : n999,
        "Mín"       : minv,
        "Máx"       : maxv,
        "Únicos"    : unicos,
    })
 
print(f"Raio-X da base ({total:,} linhas na amostra):\n")
print(pd.DataFrame(diagnostico).to_string(index=False))

# Conclusões do diagnóstico e Impacto da deduplicação (focos únicos)
total_bruto  = len(amostra)
total_unico  = len(amostra.drop_duplicates(subset=["Latitude", "Longitude", "DataHora"]))
removidos    = total_bruto - total_unico
 
print(f"Registros brutos           : {total_bruto:>10,}")
print(f"Focos únicos               : {total_unico:>10,}")
print(f"Removidos (multi-satélite) : {removidos:>10,}  "
      f"({removidos/total_bruto*100:.2f}% do total)")


#%%
# Criando uma base agregada com os dados tratados
resultado = []
biomas_lista = []

for arq in arquivos_queimadas:
    try:
        df = pd.read_csv(arq, sep=",", encoding="utf-8")

        # tratamentos
        df = df.drop_duplicates(subset=["Latitude", "Longitude", "DataHora"])
        df["RiscoFogo"]   = df["RiscoFogo"].replace(-999, np.nan)
        df["DiaSemChuva"] = df["DiaSemChuva"].replace(-999, np.nan)
        df["DataHora"]    = pd.to_datetime(df["DataHora"], errors="coerce")
        df = df.dropna(subset=["DataHora"])
        df["Ano"] = df["DataHora"].dt.year
        df["Mes"] = df["DataHora"].dt.month

        # agregação principal (Estado/Ano/Mês)
        agrupado = (df.groupby(["Estado", "Ano", "Mes"]).agg(
                        Qtd_Focos          = ("Estado",      "size"),
                        Media_Risco        = ("RiscoFogo",   "mean"),
                        Media_DiasSemChuva = ("DiaSemChuva", "mean"),
                        Media_FRP          = ("FRP",         "mean"),
                    ).reset_index())
        resultado.append(agrupado)

        # agrega bioma enquanto o df ainda está na memória
        temp_bio = (df.groupby("Bioma").agg(
                        Qtd_Focos          = ("Estado",      "size"),
                        Media_Risco        = ("RiscoFogo",   "mean"),
                        Media_DiasSemChuva = ("DiaSemChuva", "mean"),
                        Media_FRP          = ("FRP",         "mean"),
                    ).reset_index())
        biomas_lista.append(temp_bio)

        print(f"[✓] {arq.name}")

    except Exception as e:
        print(f"[!] Erro em {arq.name}: {e}")

df_queimadas = pd.concat(resultado, ignore_index=True)
df_queimadas.to_csv("queimadas_resumo_estado_mes.csv", index=False, encoding="utf-8")
 
print(f"\nBase salva → queimadas_resumo_estado_mes.csv")
print(f"Linhas: {len(df_queimadas):,}  |  Colunas: {df_queimadas.columns.tolist()}")
df_queimadas.head()

# Validação: os tratamentos funcionaram?
print("Descritiva da base tratada:\n")
print(df_queimadas.describe().round(3))
 
print("\nVerificação de negativos remanescentes:")
for col in ["Media_Risco", "Media_DiasSemChuva"]:
    neg = (df_queimadas[col] < 0).sum()
    status = "[✓] Nenhum" if neg == 0 else f"[!] {neg} encontrados — revisar"
    print(f"  {col}: {status}")


#%%
# Estatística descritiva dos focos por estado/mês (item 2.10 do planejamento)
# Média, mediana e desvio padrão — base para interpretar os gráficos seguintes
desc = df_queimadas["Qtd_Focos"].agg(["mean", "median", "std", "min", "max"])
print("Focos de queimada por estado/mês (2016-2025):")
print(f"  Média           : {desc['mean']:>12,.1f}")
print(f"  Mediana         : {desc['median']:>12,.1f}")
print(f"  Desvio padrão   : {desc['std']:>12,.1f}")
print(f"  Mínimo          : {desc['min']:>12,.0f}")
print(f"  Máximo          : {desc['max']:>12,.0f}")
print()
print("Interpretação: a mediana muito abaixo da média indica distribuição")
print("assimétrica — poucos estados/meses concentram a maioria dos focos.")

#%%
# Distribuição e outliers dos focos
# Histograma em escala log para enxergar toda a distribuição
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Histograma (mantém como estava)
sns.histplot(df_queimadas["Qtd_Focos"], bins=50, log_scale=True, ax=axes[0])
axes[0].set_title("Distribuição dos focos por estado/mês")
axes[0].set_xlabel("Qtd_Focos (escala log)")

# Boxplot com escala log — agora a caixa fica visível
sns.boxplot(y=df_queimadas["Qtd_Focos"], ax=axes[1])
axes[1].set_yscale("log")
axes[1].set_title("Boxplot — concentração e outliers (escala log)")
axes[1].set_ylabel("Qtd_Focos (escala log)")

# Calcular e anotar as estatísticas no próprio gráfico
q1  = df_queimadas["Qtd_Focos"].quantile(0.25)
med = df_queimadas["Qtd_Focos"].median()
q3  = df_queimadas["Qtd_Focos"].quantile(0.75)
iqr = q3 - q1
n_outliers = (df_queimadas["Qtd_Focos"] > q3 + 1.5 * iqr).sum()

stats_texto = (
    f"Q1  : {q1:>10,.0f}\n"
    f"Med : {med:>10,.0f}\n"
    f"Q3  : {q3:>10,.0f}\n"
    f"Out : {n_outliers:>10} pontos"
)
axes[1].text(0.62, 0.97, stats_texto,
             transform=axes[1].transAxes,
             verticalalignment="top",
             fontfamily="monospace", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()


#%%
# Verificar quais estados mais queimaram
ranking_estados = df_queimadas.groupby("Estado")["Qtd_Focos"].sum().reset_index().sort_values("Qtd_Focos", ascending=False)
ranking_estados["Perc_Total"] = ranking_estados["Qtd_Focos"]/ ranking_estados["Qtd_Focos"].sum()* 100
top15 = ranking_estados.head(15)

plt.figure(figsize=(15,7))

ax = sns.barplot( data=top15, x="Estado", y="Qtd_Focos" )

for i, row in enumerate(top15.itertuples()):
    ax.text( i, row.Qtd_Focos, f"{row.Perc_Total:.1f}%", ha="center", va="bottom", fontsize=9)

plt.title("Top 15 Estados com Mais Focos de Queimadas")
plt.xticks(rotation=30, ha="right")
plt.show()

print("Dois estados concentram quase 40% de todos os focos registrados na série histórica.")


#%%
# Criar uma base mensal para análise
base_mensal = df_queimadas.groupby(["Ano", "Mes"]).agg({ "Qtd_Focos": "sum",
                                                         "Media_Risco": "mean",
                                                         "Media_DiasSemChuva": "mean",
                                                         "Media_FRP": "mean"}).reset_index()
base_mensal.head()

#%%
# Verificar a correlação entre os focos e as variáveis de risco
corr_ambiental  = base_mensal[["Qtd_Focos", "Media_Risco", "Media_DiasSemChuva", "Media_FRP"]].corr()
print("Correlação entre focos e variáveis de risco:")
print(corr_ambiental.round(3))

#%%
# Verificar o risco médio por estado
risco_estado = df_queimadas.groupby("Estado")["Media_Risco"].mean().sort_values(ascending=False)
risco_estado = risco_estado.reset_index()

top15_risco = risco_estado.head(15)

plt.figure(figsize=(15,7))
sns.barplot(data=top15_risco,x="Estado",y="Media_Risco")
plt.title("Risco médio de fogo por estado")
plt.xticks(rotation=30, ha="right")
plt.show()


#%%
# Verificar o tempo médio sem chuva por estado
seca_estado = df_queimadas.groupby("Estado")["Media_DiasSemChuva"].mean().sort_values(ascending=False)
seca_estado = seca_estado.reset_index()

top15_seca = seca_estado.head(15)

plt.figure(figsize=(15,7))
sns.barplot(data=top15_seca,x="Estado",y="Media_DiasSemChuva")
plt.title("Tempo médio sem chuva por estado")
plt.xticks(rotation=30, ha="right")
plt.show()

#%%
# Verificar quais biomas mais queimaram
df_biomas = pd.concat(biomas_lista, ignore_index=True)
df_biomas.to_csv("queimadas_resumo_bioma.csv", index=False, encoding="utf-8")

df_biomas = df_biomas.groupby("Bioma").agg( Qtd_Focos = ("Qtd_Focos", "sum"),
                                            Media_Risco = ("Media_Risco", "mean"),
                                            Media_DiasSemChuva = ("Media_DiasSemChuva", "mean"),
                                            Media_FRP = ("Media_FRP", "mean") ).reset_index().sort_values("Qtd_Focos", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=df_biomas, x="Bioma", y="Qtd_Focos")
plt.title("Focos de Queimadas por Bioma (2016-2025)")
plt.xticks(rotation=30, ha="right")
plt.ylabel("Total de focos")
plt.tight_layout()
plt.show()

#%%
# Verificar o risco médio por bioma
plt.figure(figsize=(15, 7))
sns.barplot( data=df_biomas.sort_values("Media_Risco", ascending=False), x="Bioma", y="Media_Risco")
plt.title("Risco médio de fogo por bioma")
plt.xticks(rotation=30, ha="right")
plt.show()

#%%
# Verificar o tempo médio sem chuva por bioma
plt.figure(figsize=(15, 7))
sns.barplot(data=df_biomas.sort_values("Media_DiasSemChuva", ascending=False), x="Bioma", y="Media_DiasSemChuva")
plt.title("Tempo médio sem chuva por bioma")
plt.xticks(rotation=30, ha="right")
plt.show()

#%%
# Como os focos evoluíram ao longo do tempo?
evolucao_anual = df_queimadas.groupby("Ano")["Qtd_Focos"].sum().reset_index()

plt.figure(figsize=(15,7))
sns.lineplot( data=evolucao_anual,x="Ano",y="Qtd_Focos",marker="o" )
plt.title("Evolução dos focos de queimadas por ano")
plt.show()

#%%
# SAZONALIDADE 1 — Participação % de cada mês 
sazonalidade = df_queimadas.groupby("Mes")["Qtd_Focos"].sum().reset_index()
sazonalidade["Perc_Total"] = sazonalidade["Qtd_Focos"] / sazonalidade["Qtd_Focos"].sum() * 100
sazonalidade["Mes_Nome"] = sazonalidade["Mes"].map(nome_mes)
 
plt.figure(figsize=(15, 7))
ax = sns.barplot(data=sazonalidade, x="Mes_Nome", y="Qtd_Focos")
for i, row in enumerate(sazonalidade.itertuples()):
    ax.text(i, row.Qtd_Focos, f"{row.Perc_Total:.1f}%",
            ha="center", va="bottom", fontsize=9)
plt.title("Distribuição mensal dos focos — com participação %")
plt.xlabel("Mês")
plt.ylabel("Total de focos")
plt.tight_layout()
plt.show()
 
trimestre_seco = sazonalidade[sazonalidade["Mes"].isin([8, 9, 10])]["Perc_Total"].sum()
print(f"Ago+Set+Out concentram {trimestre_seco:.0f}% de todos os focos do ano.")
print("Setembro sozinho responde por ~30% — é o pico crítico da estação seca.")

#%%
# SAZONALIDADE 2 — Heatmap Ano x Mês
pivot_ano_mes = df_queimadas.pivot_table(
    index="Ano", columns="Mes", values="Qtd_Focos", aggfunc="sum")
pivot_ano_mes.columns = [nome_mes[m] for m in pivot_ano_mes.columns]
 
plt.figure(figsize=(15, 7))
sns.heatmap(pivot_ano_mes, cmap="YlOrRd", annot=False,
            cbar_kws={"label": "Focos de queimada"})
plt.title("Mapa de calor: focos por Ano x Mês (2016-2025)")
plt.xlabel("Mês")
plt.ylabel("Ano")
plt.tight_layout()
plt.show()
 
print("Leitura: a coluna Set/Ago é sempre a mais quente — padrão sazonal estável.")
print("2024 se destaca como o ano de seca mais severa da série.")

#%%
# SAZONALIDADE 3 — Cada região queima num mês diferente?
df_queimadas["Regiao"] = df_queimadas["Estado"].map(regiao_map)
 
saz_regiao = (df_queimadas.groupby(["Regiao", "Mes"])["Qtd_Focos"]
              .sum().reset_index())
# normaliza dentro de cada região (% do mês sobre o total da região)
saz_regiao["Perc_Regiao"] = (saz_regiao.groupby("Regiao")["Qtd_Focos"]
                             .transform(lambda x: x / x.sum() * 100))
 
plt.figure(figsize=(15, 7))
sns.lineplot(data=saz_regiao, x="Mes", y="Perc_Regiao",
             hue="Regiao", marker="o")
plt.title("Sazonalidade por região (% dos focos da região em cada mês)")
plt.xlabel("Mês")
plt.ylabel("% dos focos da região")
plt.xticks(range(1, 13), [nome_mes[m] for m in range(1, 13)])
plt.legend(title="Região")
plt.tight_layout()
plt.show()
 
print("Norte, Centro-Oeste e Sudeste picam em Set; Nordeste em Out; Sul em Ago.")

#%%
# SAZONALIDADE 4 — Focos seguem risco e seca?
saz_var = df_queimadas.groupby("Mes").agg(
    Focos = ("Qtd_Focos", "sum"),
    Risco = ("Media_Risco", "mean"),
    Seca = ("Media_DiasSemChuva", "mean"),
)
# normaliza cada coluna para 0-1 (só assim dá para comparar curvas de escalas diferentes)
saz_norm = (saz_var - saz_var.min()) / (saz_var.max() - saz_var.min())
 
plt.figure(figsize=(15, 7))
for col, cor in [("Focos", "orangered"), ("Risco", "darkgreen"), ("Seca", "steelblue")]:
    plt.plot(saz_norm.index, saz_norm[col], marker="o", label=col, color=cor)
plt.title("Focos x Risco x Dias sem chuva ao longo do ano (normalizado 0-1)")
plt.xlabel("Mês")
plt.ylabel("Intensidade relativa (0-1)")
plt.xticks(range(1, 13), [nome_mes[m] for m in range(1, 13)])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
 
print("Os dias sem chuva sobem ANTES dos focos (jul) e o risco acompanha o pico.")
print("Esse adiantamento da seca é o que sustenta a ideia de alerta antecipado.")

#%%
# Volume x Intensidade por bioma
fig, ax = plt.subplots(figsize=(13, 8))
 
scatter = ax.scatter(
    df_biomas["Qtd_Focos"],
    df_biomas["Media_Risco"],
    s=df_biomas["Media_FRP"] * 25,
    c=df_biomas["Media_DiasSemChuva"],
    cmap="YlOrRd",
    alpha=0.75,
    edgecolors="black",
    linewidths=1.2,
    zorder=3,
)
 
# deslocamento (dx, dy) em pontos, individual por bioma para não colidir
offsets = {
    "Amazônia":       (10,  10),
    "Cerrado":        (10,  10),
    "Caatinga":       (10,  10),
    "Mata Atlântica": (12,  18),   # sobe o rótulo
    "Pantanal":       (12, -22),   # desce o rótulo (evita choque com Mata Atlântica)
    "Pampa":          (10,  10),
}
 
for row in df_biomas.itertuples():
    dx, dy = offsets.get(row.Bioma, (8, 8))
    ax.annotate(
        row.Bioma,
        (row.Qtd_Focos, row.Media_Risco),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),  # liga rótulo ao ponto
    )
 
ax.set_xscale("log")
ax.set_xlabel("Total de focos (escala log) — VOLUME")
ax.set_ylabel("Risco médio de fogo (0 a 1) — CONDIÇÃO")
ax.set_title("Volume x Intensidade do fogo por bioma\n"
             "(tamanho do ponto = FRP médio · cor = dias sem chuva)")
ax.grid(alpha=0.3, zorder=0)
ax.margins(x=0.15, y=0.15)   # mais espaço nas bordas para os rótulos respirarem
 
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Média de dias sem chuva")
 
plt.tight_layout()
plt.show()







# ========================================================================================================================
# ========================================================================================================================

#%%
# Verificar os arquivos de SUS disponíveis
arquivos_sus = list(pasta_sus.glob("*.csv"))
print(f"Total de arquivos de SUS: {len(arquivos_sus)}")
# Listar os nomes dos arquivos encontrados
for arq_sus in arquivos_sus:
    print(arq_sus.name)


#%%


#%%
# Processar todos os arquivos do SUS e criar um DataFrame unificado
lista_sus = []
for arquivo in arquivos_sus:
    lista_sus.append(processar_datasus(arquivo))

df_sus = pd.concat(lista_sus, ignore_index=True)

df_sus = df_sus[(df_sus["Ano"] >= 2016) & (df_sus["Ano"] <= 2025)]
df_sus = (df_sus.groupby(["Estado", "Ano", "Mes"], as_index=False)["Internacoes"].sum())

df_sus["Ano"] = df_sus["Ano"].astype(int)
df_sus["Mes"] = df_sus["Mes"].astype(int)
df_sus["Internacoes"] = df_sus["Internacoes"].astype(int)

print(f"Base SUS completa: {len(df_sus):,} linhas | Total: {df_sus['Internacoes'].sum():,}")

#%%
# SUS-1 · ESTRUTURA DA BASE
print("Colunas:", df_sus.columns.tolist())
print("\nTipos de dado:")
print(df_sus.dtypes)
print("\nPrimeiras linhas:")
df_sus.head()
 
#%%
# SUS-2 · RAIO-X DE QUALIDADE (mesmo raio-X das queimadas, adaptado)
# A base do SUS já vem agregada e tratada; aqui confirmamos que está íntegra.
total = len(df_sus)
print(f"Raio-X da base do SUS ({total:,} linhas):\n")
 
diagnostico = []
for col in df_sus.columns:
    nulos  = df_sus[col].isna().sum()
    unicos = df_sus[col].nunique()
    if pd.api.types.is_numeric_dtype(df_sus[col]):
        zeros = int((df_sus[col] == 0).sum())
        minv  = round(float(df_sus[col].min()), 2)
        maxv  = round(float(df_sus[col].max()), 2)
    else:
        zeros, minv, maxv = "-", "-", "-"
    diagnostico.append({
        "Coluna"    : col,
        "Tipo"      : str(df_sus[col].dtype),
        "Cobertura%": round((1 - nulos / total) * 100, 1),
        "NaN"       : nulos,
        "Zeros"     : zeros,
        "Mín"       : minv,
        "Máx"       : maxv,
        "Únicos"    : unicos,
    })
print(pd.DataFrame(diagnostico).to_string(index=False))
 
# Completude: a base cobre todos os estados/anos/meses?
print(f"\nEstados: {df_sus['Estado'].nunique()} (esperado 27)")
print(f"Anos: {sorted(df_sus['Ano'].unique())}")
print(f"Linhas: {total} (esperado 27 x 10 x 12 = 3240)")
print(f"Duplicados (Estado/Ano/Mes): {df_sus.duplicated(['Estado','Ano','Mes']).sum()}")
 
 
#%%
# SUS-3 · ESTATÍSTICA DESCRITIVA (item 2.11 do planejamento)
desc = df_sus["Internacoes"].agg(["mean", "median", "std", "min", "max"])
print("Internações respiratórias por estado/mês (2016-2025):")
print(f"  Média         : {desc['mean']:>12,.1f}")
print(f"  Mediana       : {desc['median']:>12,.1f}")
print(f"  Desvio padrão : {desc['std']:>12,.1f}")
print(f"  Mínimo        : {desc['min']:>12,.0f}")
print(f"  Máximo        : {desc['max']:>12,.0f}")
print(f"\n  Total no período: {df_sus['Internacoes'].sum():,}")
 
#%%
# SUS-4 · DISTRIBUIÇÃO E OUTLIERS (item 2.12)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
sns.histplot(df_sus["Internacoes"], bins=50, ax=axes[0])
axes[0].set_title("Distribuição das internações por estado/mês")
axes[0].set_xlabel("Internações")
 
sns.boxplot(y=df_sus["Internacoes"], ax=axes[1])
axes[1].set_title("Boxplot — concentração e outliers")
axes[1].set_ylabel("Internações")
 
plt.tight_layout()
plt.show()
 
 
#%%
# SUS-5 · RANKING DE ESTADOS (espelha o ranking de focos)
ranking_sus = (df_sus.groupby("Estado")["Internacoes"].sum()
               .reset_index().sort_values("Internacoes", ascending=False))
ranking_sus["Perc_Total"] = ranking_sus["Internacoes"] / ranking_sus["Internacoes"].sum() * 100
 
top15_sus = ranking_sus.head(15)
 
plt.figure(figsize=(15, 7))
ax = sns.barplot(data=top15_sus, x="Estado", y="Internacoes")
for i, row in enumerate(top15_sus.itertuples()):
    ax.text(i, row.Internacoes, f"{row.Perc_Total:.1f}%",
            ha="center", va="bottom", fontsize=9)
plt.title("Top 15 Estados com Mais Internações Respiratórias (2016-2025)")
plt.xticks(rotation=30, ha="right")
plt.show()
 
 
#%%
# SUS-6 · EVOLUÇÃO ANUAL — procure o "vale da COVID" (2020-2021)
evolucao_sus = df_sus.groupby("Ano")["Internacoes"].sum().reset_index()
 
plt.figure(figsize=(12, 6))
sns.lineplot(data=evolucao_sus, x="Ano", y="Internacoes", marker="o")
plt.title("Evolução das internações respiratórias por ano")
plt.ylabel("Internações")
plt.grid(alpha=0.3)
plt.show()
print(evolucao_sus.to_string(index=False))
 
 
#%%
# SUS-7 · SAZONALIDADE — qual mês interna mais? (CHAVE PARA O CRUZAMENTO)
sazon_sus = df_sus.groupby("Mes")["Internacoes"].sum().reset_index()
sazon_sus["Perc_Total"] = sazon_sus["Internacoes"] / sazon_sus["Internacoes"].sum() * 100
sazon_sus["Mes_Nome"] = sazon_sus["Mes"].map(nome_mes)
 
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=sazon_sus, x="Mes_Nome", y="Internacoes")
for i, row in enumerate(sazon_sus.itertuples()):
    ax.text(i, row.Internacoes, f"{row.Perc_Total:.1f}%",
            ha="center", va="bottom", fontsize=8)
plt.title("Sazonalidade das internações respiratórias")
plt.xlabel("Mês")
plt.ylabel("Internações")
plt.tight_layout()
plt.show()
 
 
#%%
# SUS-8 · HEATMAP Ano x Mês (vê o vale da COVID e o padrão sazonal juntos)
pivot_sus = df_sus.pivot_table(index="Ano", columns="Mes",
                               values="Internacoes", aggfunc="sum")
pivot_sus.columns = [nome_mes[m] for m in pivot_sus.columns]
 
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_sus, cmap="Blues", annot=False,
            cbar_kws={"label": "Internações"})
plt.title("Mapa de calor: internações por Ano x Mês (2016-2025)")
plt.xlabel("Mês")
plt.ylabel("Ano")
plt.tight_layout()
plt.show()
 
 
#%%
# SUS-9 · SAZONALIDADE POR REGIÃO (a base do SUS usa SIGLA — atenção!)
df_sus["Regiao"] = df_sus["Estado"].map(regiao_sigla)
 
saz_reg_sus = df_sus.groupby(["Regiao", "Mes"])["Internacoes"].sum().reset_index()
saz_reg_sus["Perc_Regiao"] = (saz_reg_sus.groupby("Regiao")["Internacoes"]
                              .transform(lambda x: x / x.sum() * 100))
 
plt.figure(figsize=(13, 6))
sns.lineplot(data=saz_reg_sus, x="Mes", y="Perc_Regiao", hue="Regiao", marker="o")
plt.title("Sazonalidade das internações por região (% da região em cada mês)")
plt.xlabel("Mês")
plt.ylabel("% das internações da região")
plt.xticks(range(1, 13), [nome_mes[m] for m in range(1, 13)])
plt.legend(title="Região")
plt.tight_layout()
plt.show()

#%% 
# Harmonização das chaves para o cruzamento
df_queimadas_key = df_queimadas.copy()
df_queimadas_key["Estado"] = df_queimadas_key["Estado"].apply(estado_para_sigla)
 
# Conferência: nenhuma conversão pode falhar
falhas = df_queimadas_key["Estado"].isna().sum()
print(f"Conversões que falharam (devem ser 0): {falhas}")
print(f"Siglas resultantes: {sorted(df_queimadas_key['Estado'].dropna().unique())}")
 
#%%
# CRUZAMENTO — Merge entre queimadas e SUS para análise conjunta
print("Realizando o cruzamento entre as bases de queimadas e SUS...")
base = pd.merge(
    df_queimadas_key,
    df_sus,
    on=["Estado", "Ano", "Mes"],
    how="outer",
    indicator=True,
)
 
print("Resultado do cruzamento:")
print(base["_merge"].value_counts().to_string())
print(f"\nLinhas casadas: {(base['_merge']=='both').sum()} (esperado 3240)")
 
# Se tudo casou, removemos o indicador e seguimos
if (base["_merge"] != "both").sum() == 0:
    base = base.drop(columns=["_merge"])
    base = base.drop(columns=[c for c in ["Regiao_x", "Regiao_y"] if c in base.columns])
    print("\n[✓] Cruzamento 1:1 perfeito — nenhuma linha órfã.")
else:
    print("\n[!] Há linhas sem par — investigar antes de seguir.")
 
base.head()

#%% 
# Correlação global (mesmo mês) — o jeito ingênuo, mas muito comum de fazer
r_global, p_global = stats.pearsonr(base["Qtd_Focos"], base["Internacoes"])
rs_global, ps_global = stats.spearmanr(base["Qtd_Focos"], base["Internacoes"])
 
print("NÍVEL 1 — Correlação global (mesmo mês):")
print(f"  Pearson : r = {r_global:+.3f}  (p = {p_global:.1e})")
print(f"  Spearman: r = {rs_global:+.3f}  (p = {ps_global:.1e})")
 
#%%
# O resultado é um r positivo e significativo, mas isso não significa que mais focos causem mais internações no mesmo mês. Por quê?
ag = base.groupby("Estado").agg(Focos=("Qtd_Focos", "sum"),
                                Internacoes=("Internacoes", "sum"))
print("Estados que MAIS QUEIMAM:")
print(ag.sort_values("Focos", ascending=False).head(3).to_string())
print("\nEstados que MAIS INTERNAM:")
print(ag.sort_values("Internacoes", ascending=False).head(3).to_string())
print("\n-> São grupos diferentes. A internação é movida por POPULAÇÃO,")
print("   não por queimada. Por isso a correlação global mistura coisas")
print("   distintas (diferença entre estados, não relação temporal).")
 
#%%
base["Regiao"] = base["Estado"].map(regiao_sigla)
 
plt.figure(figsize=(11, 7))
sns.scatterplot(data=base, x="Qtd_Focos", y="Internacoes",
                hue="Regiao", alpha=0.5)
plt.xscale("log")
plt.xlabel("Focos de queimada (escala log)")
plt.ylabel("Internações respiratórias")
plt.title("Focos x Internações no mesmo mês (cada ponto = estado/mês)")
plt.legend(title="Região")
plt.tight_layout()
plt.show()
 
#%%
# Correlação DENTRO de cada estado (controla população) ---
correlacoes = []
for uf, g in base.groupby("Estado"):
    if g["Qtd_Focos"].std() > 0 and g["Internacoes"].std() > 0:
        r, p = stats.pearsonr(g["Qtd_Focos"], g["Internacoes"])
        correlacoes.append({"Estado": uf, "r": r, "p_valor": p})
 
corr_estados = pd.DataFrame(correlacoes).sort_values("r", ascending=False)
 
print("NÍVEL 2 — Correlação focos x internações DENTRO de cada estado:")
print(f"  Média das correlações: {corr_estados['r'].mean():+.3f}")
print(f"  Estados com r > 0: {(corr_estados['r'] > 0).sum()} de {len(corr_estados)}")
print(f"  Com r > 0 e significativo (p<0.05): "
      f"{((corr_estados['r'] > 0) & (corr_estados['p_valor'] < 0.05)).sum()}")
print("\nMaiores correlações positivas:")
print(corr_estados.head(5).to_string(index=False))
print("\nMaiores correlações negativas:")
print(corr_estados.tail(5).to_string(index=False))
 
#%%
# --- Visualizar as correlações por estado ---
plt.figure(figsize=(14, 6))
cores = ["#c0392b" if v < 0 else "#27ae60" for v in corr_estados["r"]]
plt.bar(corr_estados["Estado"], corr_estados["r"], color=cores)
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Correlação focos x internações no mesmo mês, por estado")
plt.ylabel("Correlação de Pearson (r)")
plt.xlabel("Estado")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.show()

#%%
# CRUZAMENTO - NÍVEL 3: defasagem temporal (lag) entre queimada e internação
base = base.sort_values(["Estado", "Ano", "Mes"]).reset_index(drop=True)

# --- ABORDAGEM 1 (ingênua): defasagem nos valores brutos ---
# Vamos ver que dá NEGATIVO — e que isso é artefato da sazonalidade.
print("ABORDAGEM 1 — defasagem nos valores brutos (r médio por estado):")
for lag in range(0, 5):
    r = corr_lag(base, "Qtd_Focos", "Internacoes", lag)
    print(f"  lag {lag} mês(es): r = {r:+.3f}")
print("  -> Negativo e mais forte com lag. NÃO é o efeito da fumaça:")
print("     é o reflexo dos dois ciclos anuais estarem fora de fase.")
 
#%%
# --- Remover a sazonalidade: trabalhar com ANOMALIAS ---
# Anomalia = valor - média daquele (estado, mês) ao longo dos anos.
# Isso responde: o mês ficou ACIMA ou ABAIXO do seu normal?
base["Focos_normal"] = base.groupby(["Estado", "Mes"])["Qtd_Focos"].transform("mean")
base["Inter_normal"] = base.groupby(["Estado", "Mes"])["Internacoes"].transform("mean")
base["Focos_anom"] = base["Qtd_Focos"] - base["Focos_normal"]
base["Inter_anom"] = base["Internacoes"] - base["Inter_normal"]
 
#%%
# --- ABORDAGEM 2 (correta): defasagem nas ANOMALIAS ---
print("ABORDAGEM 2 — defasagem nas anomalias (sazonalidade removida):")
print("Pergunta: queimada anormalmente alta -> internação anormalmente alta depois?")
resultados = []
for lag in range(0, 5):
    r_geral = corr_lag(base, "Focos_anom", "Inter_anom", lag)
    resultados.append({"lag": lag, "r": r_geral})
    print(f"  lag {lag} mês(es): r = {r_geral:+.3f}")
res_lag = pd.DataFrame(resultados)
 
#%%
# --- Comparar: Brasil todo x estados de queimada intensa ---
base["Regiao"] = base["Estado"].map(regiao_sigla)
fogo = base[base["Regiao"].isin(["Norte", "Centro-Oeste"])]
 
comparacao = []
for lag in range(0, 5):
    comparacao.append({
        "lag": lag,
        "Brasil": corr_lag(base, "Focos_anom", "Inter_anom", lag),
        "Norte+CO": corr_lag(fogo, "Focos_anom", "Inter_anom", lag),
    })
comp = pd.DataFrame(comparacao)
 
plt.figure(figsize=(11, 6))
plt.plot(comp["lag"], comp["Brasil"], marker="o", label="Brasil (27 estados)")
plt.plot(comp["lag"], comp["Norte+CO"], marker="s",
         label="Norte + Centro-Oeste (queimada intensa)")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Correlação queimada → internação (anomalias) por defasagem")
plt.xlabel("Defasagem (meses entre a queimada e a internação)")
plt.ylabel("Correlação média das anomalias (r)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
print(comp.to_string(index=False))

#%%
# CRUZAMENTO - NÍVEL 4: onde o sinal aparece?
# --- RECORTE 1: correlação das anomalias por REGIÃO e defasagem ---
def corr_lag_anom(df, lag):
    rs = []
    for uf, g in df.groupby("Estado"):
        g = g.sort_values(["Ano", "Mes"])
        fa = g["Focos_anom"].shift(lag); ia = g["Inter_anom"]
        m = fa.notna() & ia.notna()
        if m.sum() > 10 and fa[m].std() > 0:
            rs.append(stats.pearsonr(fa[m], ia[m])[0])
    return np.mean(rs) if rs else np.nan
 
tabela_reg = []
for reg in ["Norte", "Centro-Oeste", "Nordeste", "Sudeste", "Sul"]:
    sub = base[base["Regiao"] == reg]
    linha = {"Regiao": reg}
    for lag in range(0, 4):
        linha[f"lag{lag}"] = corr_lag_anom(sub, lag)
    tabela_reg.append(linha)
tab_reg = pd.DataFrame(tabela_reg)
print("Correlação das anomalias por região (quanto maior e positivo, mais sinal):")
print(tab_reg.to_string(index=False))
 
# Heatmap regional
plt.figure(figsize=(9, 5))
sns.heatmap(tab_reg.set_index("Regiao"), annot=True, fmt="+.3f",
            cmap="RdYlGn", center=0)
plt.title("Correlação queimada→internação (anomalias) por região e defasagem")
plt.xlabel("Defasagem (meses)")
plt.tight_layout()
plt.show()
 
#%%
# --- RECORTE 2: quais ESTADOS puxam o sinal (melhor lag entre 0-3) ---
linhas = []
for uf, g in base.groupby("Estado"):
    g = g.sort_values(["Ano", "Mes"])
    melhor = {"Estado": uf, "r": -1, "lag": None, "p": None}
    for lag in range(0, 4):
        fa = g["Focos_anom"].shift(lag); ia = g["Inter_anom"]
        m = fa.notna() & ia.notna()
        if m.sum() > 10 and fa[m].std() > 0:
            r, p = stats.pearsonr(fa[m], ia[m])
            if r > melhor["r"]:
                melhor = {"Estado": uf, "r": r, "lag": lag, "p": p}
    linhas.append(melhor)
por_estado = pd.DataFrame(linhas).sort_values("r", ascending=False)
print("Estados com associação positiva mais forte (melhor defasagem):")
print(por_estado.head(8).to_string(index=False))
 
#%%
# --- RECORTE 3: queimada EXTREMA -> internação sobe depois? (Brasil x Sudeste) ---
# Limiar: top 20% das anomalias positivas de foco = meses de queimada extrema.
# Comparamos a anomalia de internação N meses DEPOIS: extrema vs demais.
def teste_extremo(df, lag):
    df = df.copy()
    lim = df[df["Focos_anom"] > 0]["Focos_anom"].quantile(0.80)
    df["fut"] = df.groupby("Estado")["Inter_anom"].shift(-lag)
    alta = df[df["Focos_anom"] >= lim]["fut"].dropna()
    resto = df[df["Focos_anom"] < lim]["fut"].dropna()
    _, p = stats.mannwhitneyu(alta, resto, alternative="greater")
    return alta.mean(), resto.mean(), p
 
print("Queimada extrema -> internação (anomalia) N meses depois:\n")
print(f"{'lag':>4} | {'BRASIL extrema/resto (p)':>32} | {'SUDESTE extrema/resto (p)':>32}")
sudeste = base[base["Estado"].isin(["SP", "MG", "RJ", "ES"])]
for lag in range(1, 5):
    ab, rb, pb = teste_extremo(base, lag)
    asu, rsu, psu = teste_extremo(sudeste, lag)
    print(f"{lag:>4} | {ab:>+8.0f}/{rb:<+7.0f} p={pb:>5.3f}      | "
          f"{asu:>+8.0f}/{rsu:<+7.0f} p={psu:>5.3f}")



#%%
# API SIDRA: tabela 6579, nível UF (n3), variável 9324 (pop. estimada), todos os anos
URL_IBGE = "https://apisidra.ibge.gov.br/values/t/6579/n3/all/v/9324/p/all?formato=json"
 
req = urllib.request.Request(URL_IBGE, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    dados_ibge = json.loads(r.read().decode("utf-8"))
 
# Parsing: o 1º elemento é cabeçalho; D1C = código IBGE da UF, D3C = ano, V = população
registros = []
for item in dados_ibge[1:]:
    cod = item["D1C"]              # ex: "35"
    sigla = mapa_uf.get(cod)       # usa a fonte única do topo: "35" -> "SP"
    if sigla is None:
        continue
    registros.append({
        "Estado": sigla,
        "Ano": int(item["D3C"]),
        "Populacao": int(item["V"]),
    })
 
df_pop = pd.DataFrame(registros)
print(f"População baixada: {len(df_pop)} registros, "
      f"{df_pop['Estado'].nunique()} estados, anos {df_pop['Ano'].min()}-{df_pop['Ano'].max()}")
 
#%%
# --- Anos sem estimativa do IBGE (ex: 2007, 2010 = Censo; às vezes 2022/2023) ---
# Para garantir 2016-2025 completos, preenchemos anos faltantes por interpolação
# linear dentro de cada estado (e repetimos a ponta para anos no fim da série).
anos_alvo = list(range(2016, 2026))
completo = []
for uf, g in df_pop.groupby("Estado"):
    g = g.set_index("Ano")["Populacao"].sort_index()
    g = g.reindex(range(g.index.min(), 2026))      # garante todos os anos
    g = g.interpolate().ffill().bfill()            # preenche buracos
    for ano in anos_alvo:
        completo.append({"Estado": uf, "Ano": ano, "Populacao": int(g.loc[ano])})
df_pop = pd.DataFrame(completo)
print(f"População normalizada para {anos_alvo[0]}-{anos_alvo[-1]}: {len(df_pop)} linhas (esperado 270)")
 
#%%
# --- Juntar população à base e calcular a TAXA por 100 mil habitantes ---
base = base.merge(df_pop, on=["Estado", "Ano"], how="left")
 
# verificação: ninguém pode ficar sem população
sem_pop = base["Populacao"].isna().sum()
print(f"Linhas sem população (deve ser 0): {sem_pop}")
 
base["Internacoes_por_100k"] = base["Internacoes"] / base["Populacao"] * 100_000
 
#%%
# --- Recalcular o ranking de estados AGORA por taxa (não por total) ---
ranking_taxa = (base.groupby("Estado")
                .apply(lambda g: g["Internacoes"].sum() / g["Populacao"].mean() * 100_000)
                .sort_values(ascending=False).reset_index(name="Internacoes_por_100k"))
 
print("Ranking por TAXA (internações por 100 mil hab) — compare com o ranking bruto:")
print(ranking_taxa.head(10).to_string(index=False))
 
plt.figure(figsize=(15, 6))
top15 = ranking_taxa.head(15)
sns.barplot(data=top15, x="Estado", y="Internacoes_por_100k")
plt.title("Top 15 estados por TAXA de internação respiratória (por 100 mil hab.)")
plt.ylabel("Internações por 100 mil habitantes (total do período)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()

#%%
# CRUZAMENTO - NÍVEL 5: teste de hipótese — queimada alta (extrema) causa mais internação depois?
LAG = 3   # meses entre a queimada e a internação observada
 
# anomalias (se já criadas na Parte 3, estas linhas apenas recriam — sem efeito)
base["Focos_anom"] = base["Qtd_Focos"] - base.groupby(["Estado", "Mes"])["Qtd_Focos"].transform("mean")
base["Inter_anom"] = base["Internacoes"] - base.groupby(["Estado", "Mes"])["Internacoes"].transform("mean")
base["Inter_futura"] = base.groupby("Estado")["Inter_anom"].shift(-LAG)
 
def teste_hipotese(df, nome):
    """Compara internação futura após queimada ALTA (quartil superior das
    anomalias positivas) vs queimada BAIXA. Retorna o resultado do Mann-Whitney."""
    d = df.dropna(subset=["Inter_futura", "Focos_anom"])
    limiar = d[d["Focos_anom"] > 0]["Focos_anom"].quantile(0.75)
    alta  = d[d["Focos_anom"] >= limiar]["Inter_futura"]
    baixa = d[d["Focos_anom"] <  limiar]["Inter_futura"]
    u, p = stats.mannwhitneyu(alta, baixa, alternative="greater")
    return {
        "Grupo": nome,
        "n_alta": len(alta), "n_baixa": len(baixa),
        "mediana_alta": round(alta.median(), 1),
        "mediana_baixa": round(baixa.median(), 1),
        "U": round(u), "p_valor": round(p, 4),
        "Conclusao": "Rejeita H0 (efeito detectado)" if p < 0.05 else "Não rejeita H0",
    }
 
#%%
# Rodar o teste em três recortes
resultados = pd.DataFrame([
    teste_hipotese(base, "Brasil (27 estados)"),
    teste_hipotese(base[base["Regiao"] == "Sudeste"], "Sudeste"),
    teste_hipotese(base[base["Regiao"].isin(["Norte", "Centro-Oeste"])], "Norte+Centro-Oeste"),
])
 
print("TESTE DE HIPÓTESE — internação 3 meses após queimada alta vs baixa")
print("H0: internações iguais   |   H1: maiores após queimada alta\n")
print(resultados.to_string(index=False))
 
#%%
# Visualizar a distribuição que sustenta o teste (foco no Sudeste, onde deu significativo)
sudeste = base[base["Regiao"] == "Sudeste"].dropna(subset=["Inter_futura", "Focos_anom"])
limiar = sudeste[sudeste["Focos_anom"] > 0]["Focos_anom"].quantile(0.75)
sudeste = sudeste.copy()
sudeste["Grupo"] = np.where(sudeste["Focos_anom"] >= limiar,
                            "Após queimada ALTA", "Após queimada baixa")
 
plt.figure(figsize=(10, 6))
sns.boxplot(data=sudeste, x="Grupo", y="Inter_futura", showfliers=False)
plt.axhline(0, color="red", linestyle="--", linewidth=1, label="internação normal")
plt.title("Sudeste: internação (anomalia) 3 meses depois — queimada alta vs baixa")
plt.ylabel("Anomalia de internações (desvio do normal)")
plt.xlabel("")
plt.legend()
plt.tight_layout()
plt.show()


#%%
# MATRIZ DE CORRELAÇÃO
colunas_num = ["Qtd_Focos", "Media_Risco", "Media_DiasSemChuva",
               "Media_FRP", "Internacoes"]
colunas_num = [c for c in colunas_num if c in base.columns]
matriz_corr = base[colunas_num].corr()
 
plt.figure(figsize=(9, 7))
sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"label": "Correlação de Pearson"})
plt.title("Matriz de correlação entre as variáveis")
plt.tight_layout()
plt.show()
 
 
#%%
# SÉRIE TEMPORAL COMPARATIVA
serie = (base.groupby(["Ano", "Mes"])
         .agg(Focos=("Qtd_Focos", "sum"), Internacoes=("Internacoes", "sum"))
         .reset_index())
serie["Data"] = pd.to_datetime(
    serie["Ano"].astype(str) + "-" + serie["Mes"].astype(str) + "-01")
serie = serie.sort_values("Data")
 
fig, ax1 = plt.subplots(figsize=(15, 6))
cor_fogo, cor_saude = "#d35400", "#2980b9"
 
ax1.plot(serie["Data"], serie["Focos"], color=cor_fogo, label="Focos de queimada")
ax1.set_xlabel("Tempo")
ax1.set_ylabel("Focos de queimada", color=cor_fogo)
ax1.tick_params(axis="y", labelcolor=cor_fogo)
 
ax2 = ax1.twinx()   # segundo eixo Y compartilhando o mesmo X
ax2.plot(serie["Data"], serie["Internacoes"], color=cor_saude, label="Internações")
ax2.set_ylabel("Internações respiratórias", color=cor_saude)
ax2.tick_params(axis="y", labelcolor=cor_saude)
 
plt.title("Focos de queimada × Internações respiratórias no tempo (2016-2025)")
fig.tight_layout()
plt.show()


#%%
# TAXA DE VARIAÇÃO ANUAL ---
anual = base.groupby("Ano").agg(Focos=("Qtd_Focos", "sum"),
                                Internacoes=("Internacoes", "sum"))
anual["Var_Focos_%"] = anual["Focos"].pct_change() * 100
anual["Var_Inter_%"] = anual["Internacoes"].pct_change() * 100
 
print("Taxa de variação ano a ano (%):")
print(anual[["Var_Focos_%", "Var_Inter_%"]].round(1).to_string())
 
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(anual.index))
largura = 0.4
ax.bar(x - largura/2, anual["Var_Focos_%"], largura,
       label="Variação focos %", color=cor_fogo)
ax.bar(x + largura/2, anual["Var_Inter_%"], largura,
       label="Variação internações %", color=cor_saude)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(anual.index)
ax.set_ylabel("Variação em relação ao ano anterior (%)")
ax.set_title("Taxa de variação anual — focos e internações")
ax.legend()
plt.tight_layout()
plt.show()

#%%
# ============================================================================
# CRUZAMENTO - NÍVEL 6: uso do solo — área plantada (4ª base)
# ============================================================================
# Integra a ÁREA PLANTADA por estado (IBGE/PAM, SIDRA tabela 5457, variável 8331).
# Mesma lógica de download da população (mesma família de API do IBGE).
# A PAM cobre 2016-2024 (a safra de 2025 ainda não foi divulgada).
URL_AREA = "https://apisidra.ibge.gov.br/values/t/5457/n3/all/v/8331/p/2016-2025?formato=json"
req = urllib.request.Request(URL_AREA, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    dados_area = json.loads(r.read().decode("utf-8"))

registros = []
for item in dados_area[1:]:                  # 1º elemento é o cabeçalho
    sigla = mapa_uf.get(item["D1C"])          # D1C = código IBGE da UF
    if sigla is None:
        continue
    try:
        valor = float(item["V"])              # V = área plantada (hectares)
    except (TypeError, ValueError):
        continue                              # ignora "..."/"-" (indisponível)
    registros.append({"Estado": sigla, "Ano": int(item["D3C"]),
                       "Area_Agricola_Ha": valor})

df_area = pd.DataFrame(registros)
print(f"Área plantada baixada: {len(df_area)} registros, "
      f"{df_area['Estado'].nunique()} estados, "
      f"anos {df_area['Ano'].min()}-{df_area['Ano'].max()}")

#%%
# Integra a área à base e cria os indicadores NORMALIZADOS
base = base.drop(columns=[c for c in ["Area_Agricola_Ha"] if c in base.columns])  # re-run seguro
base = base.merge(df_area, on=["Estado", "Ano"], how="left")

base["Focos_por_1000ha"] = base["Qtd_Focos"] / base["Area_Agricola_Ha"] * 1000
base["Focos_por_100k"]   = base["Qtd_Focos"] / base["Populacao"]        * 100_000
# (Internacoes_por_100k já foi criada na seção de população)

sem_area = int(base["Area_Agricola_Ha"].isna().sum())
print(f"Base com área integrada. Linhas sem área (meses de 2025): {sem_area}")

#%%
# NÍVEL 6.1 — o fogo acompanha a agricultura? (correlação por estado)
# total de focos (2016-2025) x área plantada média anual (2016-2024)
focos_uf = base.groupby("Estado")["Qtd_Focos"].sum()
area_uf  = df_area.groupby("Estado")["Area_Agricola_Ha"].mean()
agro = pd.concat([focos_uf, area_uf], axis=1).reset_index()
agro.columns = ["Estado", "Focos", "Area_ha"]
agro["Regiao"] = agro["Estado"].map(regiao_sigla)

r,  p  = stats.pearsonr(agro["Area_ha"], agro["Focos"])
rs, ps = stats.spearmanr(agro["Area_ha"], agro["Focos"])
rl, pl = stats.pearsonr(np.log10(agro["Area_ha"]), np.log10(agro["Focos"]))
print("Focos x área plantada por estado:")
print(f"  Pearson  (linear) : r = {r:+.3f}  (p = {p:.3f})")
print(f"  Spearman (rank)   : r = {rs:+.3f}  (p = {ps:.3f})")
print(f"  Pearson  (log-log): r = {rl:+.3f}  (p = {pl:.3f})")

plt.figure(figsize=(11, 7))
sns.scatterplot(data=agro, x="Area_ha", y="Focos", hue="Regiao", s=130,
                edgecolor="white", linewidth=1)
for _, row in agro.iterrows():
    if row["Estado"] in ["MT", "PA", "PR", "RS", "SP", "AM", "GO"]:
        plt.annotate(row["Estado"], (row["Area_ha"], row["Focos"]),
                     xytext=(6, 4), textcoords="offset points", fontweight="bold")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Área plantada média anual (ha, escala log)")
plt.ylabel("Focos de queimada — total 2016-2025 (escala log)")
plt.title(f"Focos × área plantada por estado (Spearman r = {rs:.2f})")
plt.legend(title="Região")
plt.tight_layout()
plt.show()

print("Associação positiva, porém MODERADA: PR e RS plantam muito e queimam")
print("pouco; o fogo segue a fronteira agrícola sobre biomas secos (Cerrado/Amazônia).")

#%%
# NÍVEL 6.2 — focos por hectare de lavoura: o fogo do Norte não é agrícola
rank_ha = (base.dropna(subset=["Focos_por_1000ha"])
               .groupby("Estado")["Focos_por_1000ha"].mean()
               .sort_values(ascending=False).reset_index())
rank_ha["Regiao"] = rank_ha["Estado"].map(regiao_sigla)
print("Focos por 1.000 ha de lavoura — média estadual (2016-2024):")
print(rank_ha.round(1).to_string(index=False))

plt.figure(figsize=(15, 6))
sns.barplot(data=rank_ha, x="Estado", y="Focos_por_1000ha", hue="Regiao", dodge=False)
plt.yscale("log")
plt.title("Focos de queimada por 1.000 ha de lavoura — média estadual (2016-2024)")
plt.ylabel("Focos / 1.000 ha (escala log)")
plt.xlabel("")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Região", ncol=5, fontsize=8)
plt.tight_layout()
plt.show()

print("Normalizado pela lavoura, o ranking se INVERTE: o Norte dispara")
print("(AM ~201, AC ~77 focos/1.000 ha) e o Sul some (RS ~0,2). No Norte o fogo")
print("é de ABERTURA de área (desmatamento), não de manejo agrícola consolidado.")

#%%
