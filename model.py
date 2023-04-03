import tensorflow as tf
import numpy as np

# from dataclasses import dataclass

# @dataclass
# class ModelDimensions:
#     """Dimensions of the model."""
#     dff: int
#     d_model: int
#     num_heads: int
#     num_layers: int
#     dropout_rate: int

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]        # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth    # (1, depth)

    angle_rates = 1 / (10000**depths)                   # (1, depth)
    angle_rads = positions * angle_rates                # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis = -1,
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

# ----------------------------

#! Positional encoding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]

        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positional_encoding
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        
        return x


#! Add and norm layer
class AddNorm(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x, attn_output):
        x = self.add([x, attn_output])
        x = self.norm(x)

        return x

    
#! Global self attention layer
class GlobalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs) -> None:
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = key_dim,
            dropout = dropout_rate,
        )
        self.addnorm = AddNorm()

    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
        )

        x = self.addnorm(x, attn_output)
        return x
  



#! Cross attention layer
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs) -> None:
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = key_dim,
            dropout = dropout_rate,
        )
        self.addnorm = AddNorm()

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query = x,
            key = context,
            value = context,
            return_attention_scores = True,
        )

        # Cache the attention scores for plotting later
        self.last_attn_scores = attn_scores

        x = self.addnorm(x, attn_output)
        return x
     

#! Causal self attention layer
class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs) -> None:
        super().__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = key_dim,
            dropout = dropout_rate,
        )
        self.addnorm = AddNorm()


    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x,
            use_causal_mask = True,
        )

        x = self.addnorm(x, attn_output)
        return x


#! Feed forward layer
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1) -> None:
        super().__init__()

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.addnorm = AddNorm()

    def call(self, x):
        x = self.addnorm(x, self.seq(x))
        return x


#! Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1) -> None:
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate,
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    
#! Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(
            vocab_size = vocab_size,
            d_model = d_model,
        )
        self.enc_layers = [
            EncoderLayer(
                d_model = d_model,
                dff = dff,
                num_heads = num_heads,
                dropout_rate = dropout_rate,
            ) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # 'x' is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x) # Shape '(batch_size, seq_len, d_model)'

        # Add dropout
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x # Shape '(batch_size, seq_len, d_model)'


#! Decoder layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1) -> None:
        super(DecoderLayer, self).__init__() #TODO: Why is DecoderLayer needed here?

        self.causal_self_attention = CausalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout=dropout_rate,
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x)
        x = self.cross_attention(x, context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x) # Shape '(batch_size, seq_len, d_model)'
        return x


#! Decoder
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1) -> None:
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
    )
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(
            d_model = d_model,
            num_heads = num_heads,
            dff = dff,
            dropout_rate = dropout_rate,
            ) for _ in range(num_layers)]
    self.last_attn_scores = None

  def call(self, x, context):
     # 'x' is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)
    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores
    # The shape of x is (batch_size, target_seq_len, d_model)
    return x


#! Transformer
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1) -> None:
    super().__init__()

    self.encoder = Encoder(
        num_layers = num_layers,
        d_model = d_model,
        num_heads = num_heads,
        dff = dff,
        vocab_size = input_vocab_size,
        dropout_rate = dropout_rate,
    )
    self.decoder = Decoder(
        num_layers = num_layers,
        d_model = d_model,
        num_heads = num_heads,
        dff = dff,
        vocab_size = target_vocab_size,
        dropout_rate = dropout_rate,
    )
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with '.fit' you must pass all your inputs in the first argument.
    context, x = inputs

    context = self.encoder(context) # (batch_size, context_len, d_model)
    x = self.decoder(x, context) # (batch_size, target_len, d_model)

    # Final linear layer output
    logits = self.final_layer(x)
    return logits
  

#! --------------------- Training --------------------- #
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        op = tf.cast(self.d_model, tf.float32) # Convertimos aqui a tensor para que no de error de serializacion al guardar la clase
        return tf.math.rsqrt(op) * tf.math.minimum(arg1, arg2)
    
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