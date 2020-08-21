from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
    ]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post')

print(word_index)
print(sequences)
print(padded)



# padded = pad_sequences(sequences, padding='post', maxlen=5)
# If you want the sentence of max 5 words then use maxlen but it
# will take sequence from last
# padded = pad_sequences(sequences,truncating='post',padding='post')
# So by using truncating it will lose the data from end