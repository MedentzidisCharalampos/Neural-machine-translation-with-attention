# Neural-machine-translation-with-attention
We train a sequence to sequence (seq2seq) model for Spanish to English translation.   
After training the model we will be able to input a Spanish sentence, such as `¿todavia estan en casa?`, and return the English translation: `are you still at home?`

# Download and prepare the dataset
We use a language dataset provided by http://www.manythings.org/anki/ This dataset contains language translation pairs in the format:

`May I borrow this book?` `¿Puedo tomar prestado este libro?`

There are a variety of languages available, but we'll use the English-Spanish dataset. For convenience, we've hosted a copy of this dataset on Google Cloud, but you can also download your own copy. After downloading the dataset, here are the steps we take to prepare the data:

1. Add a start and end token to each sentence.
2. Clean the sentences by removing special characters.
3. Create a word index and reverse word index (dictionaries mapping from word → id and id → word).
4. Pad each sentence to a maximum length.

The output after the preprocessing of the dataset:   
English:  
`<start> if you want to sound like a native speaker , you must be willing to practice saying the same sentence over and over in the same way that banjo players practice the same phrase over and over until they can play it correctly and at the desired tempo . <end>`  
Spanish:    
`<start> si quieres sonar como un hablante nativo , debes estar dispuesto a practicar diciendo la misma frase una y otra vez de la misma manera en que un musico de banjo practica el mismo fraseo una y otra vez hasta que lo puedan tocar correctamente y en el tiempo esperado . <end>`    

# Background on the Attention Mechanism

The key idea of the attention mechanism is to establish direct short-cut connections between the target and the source by paying "attention" to relevant source content as we translate. A nice byproduct of the attention mechanism is an easy-to-visualize alignment matrix between the source and target sentences.

![alt text](https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/attention%20visualization.png)



