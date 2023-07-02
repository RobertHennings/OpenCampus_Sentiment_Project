# OpenCampus_Sentiment_Project

This repository deals with the application of Natural Language Processing methods (NLP) as part of the Opencampus course "Machine Learning with TensorFlow", offered in the summer semester 2023. The materials uploaded here represent the final project of the course. 
As my own project, I chose a scientific problem: Are journalists' opinions influenced by short sale data and can trading strategies be derived from it?
The original plan was to do the investigation on a german (home market) level, but the data quality one could obtain from the Bundesanzeiger website was found to be unsufficient to carry out exhaustive analysis. Therefore I switched to the US-capital market and found better data quality and availability.
I investigated the reserach questions by first scraping the short sale data of the US capital market through a web scraper, file by file, from the FINRA site. 
Subsequently, various NLP models were set up, to evaluate the sentiment in the news articles written by the journalists. The ultimate plan is to create a numeric sentiment score from the categorical sentiment classe (positive, neutral, negative), for which the model outputs have to be transformed into numerical values. 
Examples can be found in this markdown file, following similar approaches as the Data and Analytics Vendor RavenPack.
The desired output to create such a score and observe it over time would look like follows:

Positive sentiment: Scores in the range 60 - 100
Neutral sentiment: Scores in the range 40 - 60
Negative sentiment: Scores in the range 0 - 40

From this company specific sentiment time series inferences can be drawn comparing it to the short sale time series of the according company.

In terms of the models, I followed four approaches:

1) Setting up my own model, trained on the Financial Phrasebank dataset.
2) Using the baseline model of the Hugging face Transformer pipeline.
3) Using the Bert Baseline Model.
4) Optimization and extension of the FinBert model.

These trained and/or optimized models then should classifiy the news article text, output a numeric score, that will be compared with the behaviour of the respective short sale quotes. Finally from these results, portfolios can be formed (Long and Short Portfolios) and inferences from the achieved returns drawn, to rate a potential effect, finally answering the research questions.
## Useful Links
### FINRA Daily Short Sale Files
* FINRA Website: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files

### Sentiment Score
* RavenPack Composite Score: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECAsQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fcomposite-sentiment-score&usg=AOvVaw2XEsquUqi9b66mnngCF5TK&opi=89978449

* RavenPack Sentiment Factor: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECBAQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fconstructing-sentiment-factor&usg=AOvVaw28ZUGK6EGXGFPP-RuYWStU&opi=89978449

* RavenPack Sentiment Index: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECAoQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fintroducing-ravenpack-sentiment-index&usg=AOvVaw15aaTb82Q743g5ZBvRjRcd&opi=89978449
