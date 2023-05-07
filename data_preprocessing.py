import tensorflow as tf
import tensorflow_text as tf_text
import os

SPLIT_DATA_PATH = 'split-data/'
VOCAB_PATH = 'vocab/'
MAX_VOCAB_SIZE = 32768 #TODO: Numeros cercanos a potencias de 2 mejoran el rendimiento (menos tiempo de entrenamiento)
BUFFER_SIZE = 32768
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


def standardize_text(sentence):
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
        standardize = standardize_text,
        max_tokens = MAX_VOCAB_SIZE,
        ragged = True,
    )
    context_text_processor.adapt(dataset.map(lambda context, target: context))

    target_text_processor = tf.keras.layers.TextVectorization(
        standardize = standardize_text,
        max_tokens = MAX_VOCAB_SIZE,
        ragged = True,
    )
    target_text_processor.adapt(dataset.map(lambda context, target: target))

    return context_text_processor, target_text_processor

def process_text(context, target, context_text_processor, target_text_processor):

    context = context_text_processor(context).to_tensor()
    target = target_text_processor(target)

    trg_in = target[:,:-1].to_tensor()
    trg_out = target[:,1:].to_tensor()
  
    return (context, trg_in), trg_out