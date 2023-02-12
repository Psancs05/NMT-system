from data_preprocessing import create_datasets, text_vectorization, process_text
from model import Transformer
import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True,
        reduction = 'none',
    )
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def masked_acc(label, pred):
    pred = tf.argmax(pred, axis = 2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return  tf.reduce_sum(match) / tf.reduce_sum(mask)

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train, val, test = create_datasets()
    context_text_processor, target_text_processor = text_vectorization(train)

    # print(context_text_processor.get_vocabulary()[:10])
    # print(target_text_processor.get_vocabulary()[:10])

    train_ds = train.map(lambda x, y: process_text(x, y, context_text_processor, target_text_processor), tf.data.AUTOTUNE)
    val_ds = val.map(lambda x, y: process_text(x, y, context_text_processor, target_text_processor), tf.data.AUTOTUNE)

    # sentence = 'This is a sample sentence.'
    # vectorized_sentence = context_text_processor([sentence])
    # print(vectorized_sentence)

    #! Model
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1

    transformer = Transformer(
        num_layers = num_layers,
        d_model = d_model,
        num_heads = num_heads,
        dff = dff,
        input_vocab_size = context_text_processor.vocabulary_size(),
        target_vocab_size = target_text_processor.vocabulary_size(),
        dropout_rate = dropout_rate,
    )

    # # Just for getting one example
    # for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1):
    #     break
    # output = transformer((ex_context_tok, ex_tar_out))
    # print(ex_tar_out.shape)
    # print(ex_context_tok.shape)
    # print(output.shape)

    # attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    # print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)

    # print(transformer.summary())

    #! Loss and Optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    #! Compile
    transformer.compile(
        loss = masked_loss,
        optimizer = optimizer,
        metrics = [masked_acc],
    )

    transformer.fit(
        train_ds,
        epochs = 10,
        validation_data = val_ds,
    )
