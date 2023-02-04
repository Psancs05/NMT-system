from data_preprocessing import create_datasets, text_vectorization, process_text
import tensorflow as tf

if __name__ == '__main__':
    train, val, test = create_datasets()
    context_text_processor, target_text_processor = text_vectorization(train)

    # print(context_text_processor.get_vocabulary()[:10])
    # print(target_text_processor.get_vocabulary()[:10])

    train_ds = train.map(lambda x, y: process_text(x, y, context_text_processor, target_text_processor), tf.data.AUTOTUNE)
    val_ds = val.map(lambda x, y: process_text(x, y, context_text_processor, target_text_processor), tf.data.AUTOTUNE)

    # sentence = 'This is a sample sentence.'
    # vectorized_sentence = context_text_processor([sentence])
    # print(vectorized_sentence)