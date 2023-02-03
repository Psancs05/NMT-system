import tensorflow as tf
import tensorflow_text as tf_text
import os

SPLIT_DATA_PATH = 'split-data/'
VOCAB_PATH = 'vocab/'
MAX_VOCAB_SIZE = 5000
BUFFER_SIZE = 100000
BATCH_SIZE = 64


def open_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.readlines()


def create_datasets():
    src_raw = open_file(SPLIT_DATA_PATH + 'source_eng-spa.txt')
    trg_raw = open_file(SPLIT_DATA_PATH + 'target_eng-spa.txt')

    # Create a train dataset with 80% of the data
    train_raw = (
        tf.data.Dataset
        .from_tensor_slices((src_raw[:int(len(src_raw) * 0.8)], trg_raw[:int(len(trg_raw) * 0.8)]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    # Create a validation dataset with 10% of the data
    val_raw = (
        tf.data.Dataset
        .from_tensor_slices((src_raw[int(len(src_raw) * 0.8):int(len(src_raw) * 0.9)], trg_raw[int(len(trg_raw) * 0.8):int(len(trg_raw) * 0.9)]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    # Create a test dataset with 10% of the data
    test_raw = (
        tf.data.Dataset
        .from_tensor_slices((src_raw[int(len(src_raw) * 0.9):], trg_raw[int(len(trg_raw) * 0.9):]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

    return train_raw, val_raw, test_raw


def preproces_text(sentence):
    # Split accented characters
    sentence = tf_text.normalize_utf8(sentence, 'NFKD')

    # Lowercase
    sentence = tf.strings.lower(sentence)

    # Keep only letters, spaces, and punctuation
    sentence = tf.strings.regex_replace(sentence, '[^ a-z.?!,¿]', '')

    # Add space between words and punctuation
    sentence = tf.strings.regex_replace(sentence, '[.?!,¿]', r' \0 ')

    # Srtip whitespace
    sentence = tf.strings.strip(sentence)

    # Add start and end tokens
    sentence = tf.strings.join(['[START]', sentence, '[END]'], separator=' ')

    return sentence

def text_vectorization(dataset):
    context_text_processor = tf.keras.layers.TextVectorization(
        standardize = preproces_text,
        max_tokens = MAX_VOCAB_SIZE,
        ragged = True,
    )
    context_text_processor.adapt(dataset.map(lambda context, target: context))

    target_text_processor = tf.keras.layers.TextVectorization(
        standardize = preproces_text,
        max_tokens = MAX_VOCAB_SIZE,
        ragged = True,
    )
    target_text_processor.adapt(dataset.map(lambda context, target: target))

    return context_text_processor, target_text_processor

def process_text(context, target):

    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)

    trg_in = target[:,:-1].to_tensor()
    trg_out = target[:,1:].to_tensor()
  
    return (context, trg_in), trg_out

def save_processor(processor, file_name):
    # Create the folder if it doesn't exist
    if not os.path.exists(VOCAB_PATH):
        os.makedirs(VOCAB_PATH)

    # TODO: Checkear si esta bien guardado asi 
    # Save the vocabulary
    with open(file_name, 'w') as f:
        f.write('\n'.join(processor.get_vocabulary()))
    

if __name__ == '__main__':
    train, val, test = create_datasets()
    context_text_processor, target_text_processor = text_vectorization(train)

    print(context_text_processor.get_vocabulary()[:10])
    print(target_text_processor.get_vocabulary()[:10])

    train_ds = train.map(process_text, tf.data.AUTOTUNE)
    val_ds = val.map(process_text, tf.data.AUTOTUNE)

    # Save train and validation datasets
    tf.data.experimental.save(train_ds, 'train_ds')
    tf.data.experimental.save(val_ds, 'val_ds')

    print('------------------')

    sentence = 'This is a sample sentence.'
    vectorized_sentence = context_text_processor([sentence])
    print(vectorized_sentence)

    print('------------------')
    save_processor(context_text_processor, VOCAB_PATH + 'context_vocab.txt')
    save_processor(target_text_processor, VOCAB_PATH + 'target_vocab.txt')   