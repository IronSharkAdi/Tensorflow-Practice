import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


sentences =[
            
            'I love cats ',
            'I love dogs',
            'I fucking love math'

]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
print(tokenizer)


