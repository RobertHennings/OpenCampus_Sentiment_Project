# OpenCampus_Sentiment_Project

This repository deals with the application of Natural Language Processing methods (NLP) as part of the Opencampus course "Machine Learning with TensorFlow", offered in the summer semester 2023. The materials uploaded here represent the final project of the course. 
As my own project, I chose a scientific problem: Are journalists' opinions influenced by short sale data and can trading strategies be derived from it?
I investigated this by first scraping the short sale data of the US capital market through a web scraper, file by file, from the FINRA site. 
Subsequently, various NLP models were set up.
I followed four approaches:

1) Setting up my own model, trained on the Fin Phrase dataset.
2) Using the baseline model of the Hugging face Transformer pipeline.
3) Using the Bert Baseline Model.
4) Optimization and extension of the FinBert model.

## Useful Links
### Sentiment Score
* RavenPack Composite Score: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECAsQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fcomposite-sentiment-score&usg=AOvVaw2XEsquUqi9b66mnngCF5TK&opi=89978449

* RavenPack Sentiment Factor: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECBAQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fconstructing-sentiment-factor&usg=AOvVaw28ZUGK6EGXGFPP-RuYWStU&opi=89978449

* RavenPack Sentiment Index: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi8jYiNu-X_AhWvSPEDHTyCAo4QFnoECAoQAQ&url=https%3A%2F%2Fwww.ravenpack.com%2Fresearch%2Fintroducing-ravenpack-sentiment-index&usg=AOvVaw15aaTb82Q743g5ZBvRjRcd&opi=89978449
