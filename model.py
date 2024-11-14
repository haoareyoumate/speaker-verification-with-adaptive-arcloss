import tensorflow as tf
from tensorflow.keras import layers


class split_layer(tf.keras.Layer):
    def __init__(self, width=40):
        super().__init__()
        self.width = width

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs_shape = tf.shape(inputs)
        if inputs_shape.shape[0] == 3:
            output = tf.reshape(inputs, (inputs_shape[0], inputs_shape[1], inputs_shape[2]//self.width, self.width, 1))
            output = tf.transpose(output,perm=[0,2,1,3,4])
        else:
            output = tf.reshape(inputs, (inputs_shape[0], inputs_shape[1]//self.width, self.width, 1))
            output = tf.transpose(output,perm=[1,0,2,3])
        return output

class attention_pool(tf.keras.Layer):
    def __init__(self, embed_vector=256, intermediate=128):
        super().__init__()
        self.embed_vector = embed_vector
        self.intermediate = intermediate
    
    def build(self, input_shape):
        self.embed_kernel = self.add_weight(shape=(input_shape[-1], self.embed_vector),
                                            initializer="glorot_uniform",
                                            trainable=True)
        self.embed_bias = self.add_weight(shape=(self.embed_vector,),
                                          initializer="zeros",
                                          trainable=True)
        self.intermediate_kernel = self.add_weight(shape=(input_shape[-1], self.intermediate),
                                                   initializer="glorot_uniform",
                                                   trainable=True)
        self.intermediate_bias = self.add_weight(shape=(self.intermediate,),
                                                 initializer="zeros",
                                                 trainable=True)
        self.attention_weight = self.add_weight(shape=(self.intermediate, 1),
                                                initializer="glorot_uniform",
                                                trainable=True)
        
    def call(self, inputs):
        embed_vector = tf.linalg.matmul(inputs, self.embed_kernel) + self.embed_bias
        embed_vector = tf.math.tanh(embed_vector)
        attention = tf.linalg.matmul(inputs, self.intermediate_kernel) + self.intermediate_bias
        attention = tf.math.tanh(attention)
        attention = tf.linalg.matmul(attention, self.attention_weight)
        attention = tf.nn.softmax(attention, axis=-2)
        embed_vector = embed_vector*attention
        embed_vector = tf.math.reduce_sum(embed_vector, axis=-2)
        return embed_vector

class ResidualBlock(tf.keras.Layer):
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding="same", 
                                   kernel_initializer="he_normal")
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding="same", 
                                   kernel_initializer="he_normal")
        self.bn = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x += inputs
        return self.relu2(x)
    
class AllConvolution(tf.keras.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="valid", kernel_initializer="he_normal", activation='relu')
        self.res1 = ResidualBlock(16)
        self.maxpool = layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="valid", kernel_initializer="he_normal", activation='relu')
        self.res2 = ResidualBlock(32)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal", activation='relu')
        self.res3 = ResidualBlock(64)
        self.conv4 = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="valid", kernel_initializer="he_normal", activation='relu')
        self.conv5 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="valid", kernel_initializer="he_normal", activation='relu')
        
        self.flatten = layers.Flatten(data_format="channels_last")
    
    def build(self, inputshape):
        pass

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.res1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv5(x)
        return self.flatten(x)
    
    def compute_output_shape(self, input_shape):
        return (512,)

class MyTimeDistributed(layers.TimeDistributed):
    def call(self, inputs, training=True, mask=None):
        input_shape = tf.shape(inputs)
        mask_shape = None if mask is None else tuple(mask.shape)
        batch_size = input_shape[0]
        timesteps = input_shape[1]

        if mask_shape is not None and mask_shape[:2] != (batch_size, timesteps):
            raise ValueError(
                "`TimeDistributed` Layer should be passed a `mask` of shape "
                f"({batch_size}, {timesteps}, ...), "
                f"received: mask.shape={mask_shape}"
            )

        def time_distributed_transpose(data):
            """Swaps the timestep and batch dimensions of a tensor."""
            axes = [1, 0, *range(2, len(data.shape))]
            return tf.transpose(data, perm=axes)

        inputs = time_distributed_transpose(inputs)
        if mask is not None:
            mask = time_distributed_transpose(mask)

        def step_function(i):
            kwargs = {}
            if self.layer._call_has_mask_arg and mask is not None:
                kwargs["mask"] = mask[i]
            if self.layer._call_has_training_arg:
                kwargs["training"] = training
            return self.layer.call(inputs[i], **kwargs)

        # Implementation #1: is the time axis is static, use a Python for loop.

        if inputs.shape[0] is not None:
            outputs = tf.stack(
                [step_function(i) for i in range(inputs.shape[0])]
            )
            return time_distributed_transpose(outputs)

        # Implementation #2: use tf.vectorized_map.
        outputs = tf.vectorized_map(step_function, tf.range(timesteps))
        return time_distributed_transpose(outputs)
    
class EmbedModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mel = layers.MelSpectrogram(fft_length=400, sequence_stride=160, sampling_rate=16000, num_mel_bins=40,
                                         min_freq=125.0, max_freq=3800.0, power_to_db=True,mag_exp=2.0)
        self.split_input_layer = split_layer()
        
        self.all_convolution = MyTimeDistributed(AllConvolution())
        
        self.dense1 = layers.Dense(256, activation='relu')
        self.gru = layers.GRU(256, return_sequences=True)
        
        self.attention_pool_layer = attention_pool()
        self.dense2 = layers.Dense(128, activation="linear")
        

    def call(self, inputs):
        x = self.mel(inputs)
        x = self.split_input_layer(x)
        x = self.all_convolution(x)
        x = self.dense1(x)
        x = self.gru(x)
        x = self.attention_pool_layer(x)
        return self.dense2(x)
    
    def build(self, input_shape):
        pass

    def compute_output_shape(self, input_shape):
        # This method explicitly defines the output shape of the model
        return (128,)

class CosineSimilarityLayer(layers.Layer):
    def __init__(self, output_dim):
        super(CosineSimilarityLayer, self).__init__()
        self.output_dim = output_dim

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        
    def call(self, inputs):
        x_normalized = tf.reshape(tf.math.l2_normalize(inputs, axis=-1), shape= tf.concat([tf.shape(inputs),[1]], axis=0) )
        w_normalized = tf.math.l2_normalize(self.W, axis=-1)
        
        cosine_similarity = tf.math.reduce_sum(x_normalized * w_normalized, axis = -2)
        return cosine_similarity

class CosineSimilarityModel(tf.keras.Model):
    def __init__(self,output_dim=462):
        super().__init__()
        self.output_dim = output_dim
        self.only_layer = CosineSimilarityLayer(output_dim)
        
    def build(self, input_shape):
        pass

    def call(self, inputs):
        return self.only_layer(inputs)
    
    def compute_output_shape(self, input_shape):
        return (self.output_dim,)
    
    
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate=1e-5, warmup_target=2e-4, warmup_steps=4_500, decay_rate=0.9, decay_steps=1000):
        self.initial_learning_rate = initial_learning_rate
        self.total_step_delta = warmup_target - initial_learning_rate
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def _warmup_function(self, step):
        completed_fraction = step / self.warmup_steps
        return self.total_step_delta * completed_fraction + self.initial_learning_rate

    def _decay_function(self, step):
        return self.warmup_target*( self.decay_rate**((step-self.warmup_steps)/self.decay_steps) )

    def __call__(self, step):
        learning_rate = tf.where(step <= self.warmup_steps, x=self._warmup_function(step), y=self._decay_function(step))
        return learning_rate
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
            "decay_rate": self.decay_rate,
            "decay_steps": self.decay_steps
        }