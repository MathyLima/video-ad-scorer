# video-ad-scorer
AI-powered analysis and recommendations engine for video ad creatives — Klike Data Science Challenge
---
## EDA
---
### Valores faltantes
 Após analisar o dataset fornecido, foi possível identificar atributos com valores faltantes, sendo eles
 * **`has_subtitle`** (9.2%)
 * **`music_voice_ratio`** (7.6%)
 * **`cpc`** (5.6%)
 * **`revenue`** (5.0%)
 * **`avg_watch_time_s`** (5.4%)
 * **`engagement_rate`** (6.4%)
 ![Gráfico de valores faltantes](./imagens/quantidade_faltantes_coluna.png)
 ![Gráfico de valores faltantes Porcentagem](./imagens/porcentagem_faltantes_coluna.png)

 Os números sugerem que os valores faltantes não estão ligados à problemas generalizados na coleta dos dados. Entretanto algumas colunas podem ter padrões específicos de ausências, sendo necessário mais análises, como checar se os valores faltantes estão relacionados a uma plataforma ou grupo específico de vídeos ou campanhas.
---
### Escala nos atributos
 Dado o contexto da aplicação, é natural que alguns atributos apresentem uma variação maior em sua escala de valores. Ao analisar os dados, é possível destacar essas características.
 * **`impressions`**
 * **`revenue`**
 * **`spend`**
 * **`clicks`**
 * **`conversions`**
 * **`video_duration_s`**
 ![Gráfico de distribuição](./imagens/escala_atributos.png)
 ![Gráfico de distribuição boxplot](./imagens/boxplot_escala.png)

 Onde seus valores extremos positivos nem sempre indicam ruídos nos dados, já que vídeos virais podem gerar milhares de impressões, enquanto outros podem variar em poucas unidades.
 Contudo, para modelos de machine learning, essas diferenças de escala podem gerar problemas, pois muitas métricas e algoritmos (como regressão linear, redes neurais e até cálculo de correlação) são sensíveis à magnitude dos valores.
 Portanto, é recomendado aplicar normalização ou padronização nos atributos, de forma a reduzir o impacto dos valores extremos e permitir que o modelo aprenda de forma mais equilibrada sem que atributos de maior magnitude dominem a influência.
---
### Tratamento de dados Faltantes
- **`has_subtitle`** :apresenta 46 nulos (~9% do conjunto de dados) distribuídos proporcionalmente entre plataformas. Com isto, pode-se inferir que não é problema com coleta de dados em alguma plataforma específica, onde a ausência dos valores pode significar que a informação não foi registrada e a ausência do registro sugere que o atributo não estava presente. Preenchidos com **`False`**, assumindo ausência de legenda.
- Para os atributos **`video_duration_s`**, **`cpc`**, **`revenue`**, **`avg_watch_time_s`** e **`engagement_rate`** primeiramente foi avaliado a porcentagem de dados faltantes, como ela é baixa, não justifica a remoção das categorias. Com isto, o preenchimento dos dados por média ou mediana se apresentam como melhor estratégia, para isso, foi utilizado a medida de assimetria ou **skewness** de cada coluna, onde |**skewness**| > 1 representa uma distribuição assimétrica, o que caracteriza o preenchimento por mediana, pois nesse caso a média seria uma medida enganosa. Já |**skewness**| < 1 representa uma distribuição simétrica, apontando o cenário de preenchimento por média.
![Distribuição de cada atributo e o valor de assimetria](/imagens/valor_assimetria_atributos.png)
---
### Tratamento de Outliers
 Outliers têm como definição valores extremos ligados aos dados, sejam eles muito grandes ou muito pequenos. No entanto, nem sempre a presença de outlier significa que o registro deve ser removido. Existem atributos que naturalmente podem conter valores extremos, principalmente quando se fala de caraterísticas inseridas no contexto de redes sociais, como impressões, quantidade de cliques, valor gasto e receita gerada.
- **`impressions`** e **`clicks`**: são metricas que indicam a "quantidade de algo" que crescem muito, onde seus valores podem assumir várias ordens de magnitude. Sendo assim, como o objetivo final é a utilização para regressão, pode-se aplicar log, que comprimir a escala da característica, mantendo sua ordem relativa, o que o impacto dos valores extremos em análises e modelos, já que as estatísticas como média e correlação ficam menos influenciadas por esses valores.
- **`spend`**, **`revenue`** e **`conversions`**: são métricas financeiras e de resultado que podem variar bastante entre campanhas, assumindo diferentes ordens de magnitude. Para padronizar a escala sem distorcer a relação entre os valores, aplicou-se o MinMaxScaler, que transforma cada atributo para o intervalo [0, 1] com base no mínimo e máximo observados. Isso reduz o impacto de valores extremos em modelos de machine learning, evitando que atributos de maior magnitude dominem o aprendizado em detrimento dos demais.
Após a normalização entre mínimo e máximo e a aplicação de log, o resultado obtido:
 ![BoxPlot pós tratamento de escala](./imagens/distribuicao_apos_norm_lg.png)
---


---