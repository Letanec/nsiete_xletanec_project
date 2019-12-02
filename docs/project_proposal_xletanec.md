# Návrh projektu: Rozpoznávanie sentimentu z textu
## Meno: Richard Letanec

## Motivácia
V projekte by som sa chcel venovať rozpoznávaniu sentimentu z anglického textu. Rozpoznávanie sentimentu v texte sa využíva pri analýze používateľských recenzií, hodnotení alebo v príspevkoch na socialnych sieťach. Cieľ projektu je klasifikácia skúmaného textu do troch kategórií sentimentu: pozitívny, negatívny a neutrálny.


## Súvisiace práce
Sentiment Analysis on News and Politics Text: http://cs230.stanford.edu/projects_winter_2019/reports/15811341.pdf

V projekte študenti analyzovali názory verejnosti na aktuálne politické udalosti. Ako dataset využili komentáre z twitteru. V projekte vytvorili a otestovali tri metódy riešenia problému: Multinomial Naive Bayes Classifier, GRU s GloVe a CNN s využitím max-poolingu.

Sentiment140, Twitter Sentiment Classification using Distant Supervision: https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf

V práci bola riešená klasifikácia sentimentu v twitter správach s využitím vzdialeného dozoru. Pri trénovaní využívali twitter správy s emotikonmi. Pri riešení využili metódy Naive Bayes, Maximum Entropy a SVM. 



## Dataset
http://help.sentiment140.com/for-students/ 

Tento dataset vo formáte CSV obsahuje 1600000 záznamov twitter správ s odstránenými emotikonmi. 
Každý záznam obsahuje 6 polí:
    
    1. Sentiment správy (0 - negatívny, 2 - neutrálny, 4 - pozitívny)
    2. ID správy
    3. Dátum pridania správy
    4. Query
    5. Meno používateľa
    6. Text správy

## Zdroje
    http://help.sentiment140.com/for-students/ 
    http://cs230.stanford.edu/projects_winter_2019/reports/15811341.pdf
    https://cs230.stanford.edu/past-projects/
    https://www.aclweb.org/anthology/W18-6231.pdf
