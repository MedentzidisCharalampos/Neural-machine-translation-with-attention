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

# Background on Neural Machine Translation (NMT)
An NMT system first reads the source sentence using an encoder to build a "thought" vector, a sequence of numbers that represents the sentence meaning; a decoder, then, processes the sentence vector to emit a translation, as illustrated in Figure 1. This is often referred to as the encoder-decoder architecture. In this manner, NMT addresses the local translation problem in the traditional phrase-based approach: it can capture long-range dependencies in languages, e.g., gender agreements; syntax structures; etc., and produce much more fluent translations as demonstrated by Google Neural Machine Translation systems.

<p align="center">
  <img  src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/encoder_decoder_architecture.jpg">
  <p>Figure 1. Encoder-decoder architecture – example of a general approach for NMT. An encoder converts a source sentence into a "meaning" vector which is passed through a decoder to produce a translation.</p>
</p>


NMT models vary in terms of their exact architectures. A natural choice for sequential data is the recurrent neural network (RNN), used by most NMT models. Usually an RNN is used for both the encoder and decoder. The RNN models, however, differ in terms of: (a) directionality – unidirectional or bidirectional; (b) depth – single- or multi-layer; and (c) type – often either a vanilla RNN, a Long Short-term Memory (LSTM), or a gated recurrent unit (GRU). Interested readers can find more information about RNNs and LSTM on this blog post.

We consider as examples a deep multi-layer RNN which is unidirectional and uses LSTM as a recurrent unit. We show an example of such a model in Figure 2. In this example, we build a model to translate a source sentence "I am a student" into a target sentence "Je suis étudiant". At a high level, the NMT model consists of two recurrent neural networks: the encoder RNN simply consumes the input source words without making any prediction; the decoder, on the other hand, processes the target sentence while predicting the next words.

<p align="center">
  <img  src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/neural_machine_translation.jpg">
  <p>Figure 2. Neural machine translation – example of a deep recurrent architecture proposed by for translating a source sentence "I am a student" into a target sentence "Je suis étudiant".</p>
</p>


At the bottom layer, the encoder and decoder RNNs receive as input the following: first, the source sentence, then a boundary marker "<s>" which indicates the transition from the encoding to the decoding mode, and the target sentence. For training, we will feed the system with the following tensors, which are in time-major format and contain word indices:

encoder_inputs [max_encoder_time, batch_size]: source input words.
decoder_inputs [max_decoder_time, batch_size]: target input words.
decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to the left by one time step with an end-of-sentence tag appended on the right.

Embedding:  

Given the categorical nature of words, the model must first look up the source and target embeddings to retrieve the corresponding word representations. For this embedding layer to work, a vocabulary is first chosen for each language. The embedding weights, one set per language, are learned during training.

Encoder:  

Once retrieved, the word embeddings are then fed as input into the main network, which consists of two multi-layer RNNs – an encoder for the source language and a decoder for the target language. These two RNNs, in principle, can share the same weights.

Decoder: 


