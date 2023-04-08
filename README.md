# Chatbot
## **Building a simple chat-bot using Microsofts MetaLWOz dataset.**

#### This repo contains all the required resources in order to create a simple chat-bot with the use of MetaLWOz dataset. 

#### Briefly, my implementation took into account:

  • **(1) Data pre-processing.** Prepare the training data (pairs of sentences from the provided data set).
  
  • **(2) Neural Network Structure.** Choosing an appropriate neural network structure that can model the problem.
  
  • **(3) Loss Function.** Selecting an appropriate loss function.
  
  • **(4) Training.** Training of the model on sentence pairs ([input, output]).
  
  • **(5) Testing - Inference.** A txt as well as a gif are provided with some test conversations with the chatbot.

In general, the chat-bot behaved as expected without having exciting results. This was expected because we had limitations in RAM and the total training time of the model. Still from the results it was seen that the model didn’t overcome the problem of overfitting, especially for discussions that existed arbitrarily in the training data. Some examples are the chatbot's constant references to setting an alarm, in questions like what’s the number and what do you need help with.

One way to overcome this is to train separate models for each of the 47 domains or at least for the similar ones. This could be done by creating different vocabularies for each domain and prompting the user to select a type of conversation he wishes to have. This way, the encoder – decoder model would be able to obtain and process only data from the selected vocabulary.

Finally, the results might be quite different if we didn’t delete all the questions that were over 13 words. This would lead to a new bigger vocabulary and the training step would become more time-consuming and source-demanding

### Here's an example of conversation with the chatbot:
![](https://github.com/christakakis/chatbot/blob/main/test_conv.gif)

**I have to mention that for implementing this project I produced, changed and obtained code from several sources that I link on the provided PDF file here in this repo.
Also I should mention that the file **metalwoz-v1** provided in this repo was downloaded from https://www.microsoft.com/en-us/research/project/metalwoz/.**

This repository was initially created to store my personal python codes but also be available to others trying to build or understand something similar.
The codes contained in this repo are made specifically for a Machine Learning and Natural Language Processing course of my MSc program.
