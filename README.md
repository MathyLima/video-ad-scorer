# video-ad-
---
AI-powered analysis and recommendations engine for video ad creatives — Klike Data Science Challenge
---
## EDA

### Visão geral dos dados
![Visão dos atributos numéricos](./imagens/inicial_atrib_num.png)
 Como observado na figura acima, **` klike_score `**, **` musice_voice_ratio `**, **` ctr `**, **` engagement_rate `** são atributos com distribuição apoximadamente normais:
 * Possuem curva em sino
 * KDE bem centralizado
 * Simétrico
  As features **` impressions `**, **` clicks `**, **` spend `**, **` conversions `**, **` revenue `**, **` roas `**, **` avg_watch_time_s `**,**` cpc `**, **`video_duration_s`** possuem uma distribuição fortemente assimétrica, o que indica presença de outliers por ter muitos valores pequenos e poucos valores grandes ( Diferença em escala causando assimetria ). Mas é importante avaliar se esses Outliers são naturais ou não e eles devem ser completamente eliminados ou se eles expõem a distribuição natural do contexto.

  I
  Isso significa que poucos vídeos performam muito ( o que é comum em plataformas digitais ).
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
 * **`music_voice_ratio`**
 ![Gráfico de distribuição](./imagens/escala_atributos.png)
 ![Gráfico de distribuição boxplot](./imagens/boxplot_escala.png)

  Valores extremos positivos nem sempre indicam a presença de ruídos nos dados, uma vez que vídeos virais podem gerar milhares de impressões, enquanto outros podem apresentar valores limitados a poucas unidades. à magnitude dos valores.
  Diversos modelos de machine learning são sensíveis a essas diferenças de escala, podendo apresentar problemas durante o treinamento, pois muitas métricas e algoritmos, como regressão linear, redes neurais e até o cálculo de correlação, dependem diretamente da magnitude dos valores.
  Assim, é recomendado aplicar normalização ou padronização nos atributos, de forma a reduzir o impacto dos valores extremos e permitir que o modelo aprenda de forma mais equilibrada sem que atributos de maior magnitude dominem a influência.
  Embora métodos baseados em árvores, como o XGBoost, sejam naturalmente robustos à escala dos atributos, tais transformações foram consideradas com o objetivo de melhorar a interpretabilidade estatística, a estabilidade numérica e a comparabilidade entre diferentes métricas analisadas.
---
### Tratamento de dados Faltantes
- **`has_subtitle`** :apresenta 46 nulos (~9% do conjunto de dados) distribuídos proporcionalmente entre plataformas. Com isto, pode-se inferir que não é problema com coleta de dados em alguma plataforma específica, onde a ausência dos valores pode significar que a informação não foi registrada e a ausência do registro sugere que o atributo não estava presente. Preenchidos com **`False`**, assumindo ausência de legenda.
- Para os atributos **`video_duration_s`**, **`cpc`**, **`revenue`**, **`avg_watch_time_s`** e **`engagement_rate`** primeiramente foi avaliado a porcentagem de dados faltantes, como ela é baixa, não justifica a remoção das categorias. Com isto, o preenchimento dos dados por média ou mediana se apresentam como melhor estratégia, para isso, foi utilizado a medida de assimetria ou **skewness** de cada coluna, onde |**skewness**| > 1 representa uma distribuição assimétrica, o que caracteriza o preenchimento por mediana, pois nesse caso a média seria uma medida enganosa. Já |**skewness**| < 1 representa uma distribuição simétrica, apontando o cenário de preenchimento por média. Por outro lado, **`music_voice_ratio`** apresenta um **skewness** < 1, sendo assim é recomendado aplicar a média
![Distribuição de cada atributo e o valor de assimetria](/imagens/valor_assimetria_atributos.png)
---
### Tratamento de Outliers
 Outliers têm como definição valores extremos ligados aos dados, sejam eles muito grandes ou muito pequenos. No entanto, nem sempre a presença de outlier significa que o registro deve ser removido. Existem atributos que naturalmente podem conter valores extremos, principalmente quando se fala de caraterísticas inseridas no contexto de redes sociais, como impressões, quantidade de cliques, valor gasto e receita gerada.
- **`impressions`**, **`clicks`**, **`spend`**, **`revenue`** e **`conversions`**: são metricas que indicam a "quantidade de algo" que crescem muito, onde seus valores podem assumir várias ordens de magnitude. Para definir a estratégia a ser utilizada, foram avaliadas 3 estratégias: 
  * Escalonador MinMax sem log: normaliza os dados sem alterar a escala original.
  * Log seguido de MinMax: aplica log1p antes da normalização, comprimindo a escala e reduzindo a influência de valores muito altos.
  * RobustScaler: normaliza os dados usando estatísticas robustas (mediana e IQR), menos sensíveis a outliers.Escalonador MinMax sem utilização da converção do log para normalizar os dados, o uso do log antes de escalonar e o RobustScaler.
 ![Comparação de estratégias de normalização](./imagens/estrategias_normalizacao.png)
  Observa-se que a estratégia que torna os dados mais simétricos e distribuidos é o log seguido de MinMax.
  Portanto, como o objetivo final é aplicar modelos de regressão, recomenda-se:
  * Aplicar log1p para comprimir a escala da variável, mantendo sua ordem relativa e reduzindo o impacto de valores extremos nas análises, estatísticas como média e correlação, e nos modelos.
  * Em seguida, aplicar o MinMaxScaler, que transforma todos os valores para a faixa entre 0 e 1. O escalonamento só deve ser feito após o log, pois valores muito grandes poderiam comprimir demais os valores pequenos se o MinMax fosse aplicado diretamente.
 ![BoxPlot pós tratamento de escala](./imagens/distribuicao_apos_norm_lg.png)

---
### Relação atributo criativo x atributo performance
Com o objetivo de entender quais características do criativo estão associadas ao desempenho das campanhas, foi realizada uma análise de correlação entre atributos do vídeo e métricas de performance.
Foram considerados como atributos criativos elementos diretamente relacionados à construção do anúncio, como presença de *hook*, legenda, CTA, rosto humano, densidade de texto, duração do vídeo e proporção entre música e voz, esses atributos foram comparados com métricas de performance relevantes, incluindo CTR, engajamento, conversões, tempo médio assistido e o `klike_score`, métrica alvo do desafio.

![Gráfico de correlação criativo x performance](./imagens/criativo_x_performance.png)

* Observa-se que a presença de *hook* apresenta a maior associação positiva com o `klike_score`, sugerindo que capturar a atenção do usuário nos primeiros segundos do vídeo pode ser um fator relevante para o desempenho geral do anúncio.
* A presença de rostos humanos também demonstra correlação positiva moderada com métricas de performance, indicando possível aumento de conexão emocional e retenção do público.
* A duração do vídeo apresenta forte correlação com o tempo médio assistido, comportamento esperado, já que conteúdos mais longos tendem a permitir maior tempo de visualização absoluta.
* Anúncios em formato horizontal tendem a ter mais tempo de visualização absoluta, já o vertical impacta negativamente na métrica, mas tem um impacto maior nos índices de engajamento.
* Por outro lado, criativos com alta densidade de texto apresentam correlação negativa com o `klike_score`, sugerindo que excesso de informação visual pode reduzir a efetividade do anúncio em ambientes de consumo rápido, como plataformas sociais.

A análise de correlação não estabelece causalidade, mas permite identificar padrões iniciais que auxiliam na compreensão do comportamento dos usuários e na definição de hipóteses para modelagem preditiva e recomendações criativas.
---
### Comportamento x plataforma
Foram considerados como atributos criativos elementos diretamente relacionados à construção do anúncio, como presença de *hook*, legenda, CTA, rosto humano, densidade de texto, duração do vídeo e proporção entre música e voz,essas features foram comparadas com métricas de performance relevantes, incluindo CTR, engajamento, conversões, tempo médio assistido e o `klike_score`, métrica alvo do desafio.
![alt text](./imagens/criativo_x_plataforma.png)
As correlações entre os entre os atributos criativos apresentam algumas semelhanças e diferenças para a criação de conteúdo para cada plataforma. Como semelhança, podemos destacar a preferência pela presença de rosto nos anúncios, um hook aos primeiros segundos, a presença de cta, legendas e uma proporção entre fala e música equilibrada. Entretanto, as plataformas também demonstram suas particularidades:
 **`TikTok`** apresenta preferência por vídeos no formato vertical, com baixa duração e densidade de texto equilibrada.
 **`Meta`** apresenta um equilíbrio entre baixa e média presença de textos, preferencialmente em formato vertical.
 **`LinkedIn`** apresenta preferência por vídeos em formato vertical e com duração maior que os outros.
![alt text](./imagens/criativo_x_perfomance_x_plataforma_Meta.png)

![alt text](./imagens/criativo_x_perfomance_x_plataforma_LinkedIn.png)
![alt text](./imagens/criativo_x_perfomance_x_plataforma_TikTok.png)

  De forma geral, os resultados indicam que o que funciona em uma plataforma não necessariamente funciona em outra — e que cada canal possui sensibilidades distintas a determinados elementos criativos.
  O atributo has_hook é o que apresenta correlação positiva mais consistente com o klike_score nas três plataformas (Meta: 0.59, TikTok: 0.54, LinkedIn: 0.50), sugerindo que iniciar o vídeo com um elemento de atenção é uma prática universalmente eficaz. Da mesma forma, a presença de rosto humano (has_face) e de CTA (has_cta) contribuem positivamente para o score em todos os canais, embora com magnitudes diferentes.
  Por outro lado, a densidade de texto alta (text_density_high) é consistentemente prejudicial ao klike_score nas três plataformas (LinkedIn: -0.34, Meta: -0.32, TikTok: -0.27), reforçando que anúncios sobrecarregados visualmente tendem a underperformar independentemente do canal.
  As diferenças entre plataformas ficam mais evidentes ao observar métricas específicas:

  No LinkedIn, a duração do vídeo (video_duration_s_lg_mm) tem correlação forte com o tempo médio assistido (0.61), indicando que o público dessa plataforma tolera — e consome — conteúdos mais longos.
  No Meta, essa mesma variável apresenta a maior correlação com avg_watch_time_s_lg_mm entre todas as plataformas (0.73), mas correlação negativa com CTR (-0.14), sugerindo que vídeos mais longos prendem a atenção mas reduzem o clique.
  No TikTok, vídeos mais longos também aumentam o tempo assistido (0.65), porém o formato e o ritmo do conteúdo parecem ser mais determinantes para engajamento do que a duração em si.
---
### Visualizações Marketing
 **Ranking de impacto por plataforma: mostra lado a lado quais atributos ajudam ou prejudicam o klike_score em cada canal.**
![alt text](./imagens/ranking_impacto_plataforma.png)
 **Radar do perfil criativo ideal: compara visualmente o "receituário" de cada plataforma**
![alt text](./imagens/radar_perfil_criativo_plataforma.png)
**Universal vs. específico: deixa claro o que funciona em todos os canais (hook, rosto, CTA) vs. o que muda dependendo da plataforma**
![alt text](./imagens/universal_x_especifico.png)
---

---
