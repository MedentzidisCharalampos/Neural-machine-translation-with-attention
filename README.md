# Neural-machine-translation-with-attention
We train a sequence to sequence (seq2seq) model for Spanish to English translation.   
After training the model we will be able to input a Spanish sentence, such as `¿todavia estan en casa?`, and return the English translation: `are you still at home?`

# Download and prepare the dataset
We use a language dataset provided by http://www.manythings.org/anki/ This dataset contains language translation pairs in the format:

`May I borrow this book?` `¿Puedo tomar prestado este libro?`

There are a variety of languages available, but we'll use the English-Spanish dataset. The steps we take to prepare the data:

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

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/attention%20visualization.png" />
</p>  
<p align="center">  Figure 1. Attention visualization. <p>

We now describe an instance of the attention mechanism proposed in (Luong et al., 2015), which has been used in several state-of-the-art systems including open-source toolkits such as OpenNMT.


<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/attention_mechanism.jpg" />
</p>    
<p align="center">  Figure 2. Attention Mechanism. <p>  
As illustrated in Figure 2, the attention computation happens at every decoder time step. It consists of the following stages:

1. The current target hidden state is compared with all source states to derive attention weights (can be visualized as in Figure 1).
2. Based on the attention weights we compute a context vector as the weighted average of the source states.
3. Combine the context vector with the current target hidden state to yield the final attention vector
4. The attention vector is fed as an input to the next time step (input feeding). The first three steps can be summarized by the equations below:

![alt text](https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/attention_equation.jpg)

Here, the function score is used to compared the target hidden state with each of the source hidden states, and the result is normalized to produced attention weights. There are various choices of the scoring function; popular scoring functions include the multiplicative and additive forms given in Eq. (4). 

![alt text](https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/score_function.jpg)

# Training

1. Pass the input through the encoder which return encoder output and the encoder hidden state.
2. The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
3. The decoder returns the predictions and the decoder hidden state.
4. The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
5. Use teacher forcing to decide the next input to the decoder.
6. Teacher forcing is the technique where the target word is passed as the next input to the decoder.
7. The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

# Translate

1. The evaluate function is similar to the training loop, except we don't use teacher forcing here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
2. Stop predicting when the model predicts the end token.
3. Store the attention weights for every time step.

# Examples

Input: <start> hace mucho frio aqui . <end>  
Predicted translation: it s very cold here . <end>   

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/output1.png" />
</p>    
 
Input: <start> esta es mi vida . <end>  
Predicted translation: this is my life . <end>   

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/output2.png" />
</p>    
 
Input: <start> ¿ todavia estan en casa ? <end>  
Predicted translation: are you still at home ? <end>   

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/output3.png" />
</p>    
 
Input: <start> trata de averiguarlo . <end>  
Predicted translation: try to figure it out . <end>   

<p align="center">
  <img src="https://github.com/MedentzidisCharalampos/Neural-machine-translation-with-attention/blob/main/output4.png" />
</p>    
