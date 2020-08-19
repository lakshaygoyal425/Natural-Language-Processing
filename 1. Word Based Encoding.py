from tensorflow.keras.preprocessing.text import Tokenizer


# Put the sentences into the array
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
    ] 

tokenizer = Tokenizer(num_words=100)
# Tokenizer strips punctuation outs

tokenizer.fit_on_texts(sentences)

# It returns the dictionary containing key value pairs, 
# where key is the word , and the value is the token for that word 
word_index = tokenizer.word_index
print(word_index)