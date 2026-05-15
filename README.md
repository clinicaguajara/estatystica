# Estatystica

**Estatystica** é uma aplicação web interativa desenvolvida com [Streamlit](https://streamlit.io/) para realizar testes estatísticos clássicos de forma intuitiva e em **português-brasileiro**, com código e estrutura interna escritos em **inglês**.

## Funcionalidades

- Upload de arquivos `.csv`, `.xls`, `.xlsx`, `.sav`, `.zsav` e `.mat`
- Estatísticas descritivas (média, mediana, moda, desvio padrão)
- Testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov)
- Testes paramétricos (t de Student, ANOVA)
- Testes não-paramétricos (Mann-Whitney, Kruskal-Wallis)
- Correlações (Pearson, Spearman)
- Regressão linear simples e múltipla
- Visualização de sinais temporais multicanais, potência espectral e espectrograma
- Interpretação textual dos resultados estatísticos (em português)
- Exportação dos resultados (.csv ou .pdf)

## 📁 Arquitetura do Projeto

Estatystica/
│ 
├── .streamlit/
│   └── config.toml
│ 
├── assets/
│ 
├── components/
│ 
├── modules/
│ 
├── pages/
│ 
├── utils/
│ 
├── requirements.txt
│ 
├── Estatystica.py
│ 
└── README.md
