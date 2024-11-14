import tensorflow as tf
from math import pi

class ArcLoss(tf.keras.losses.Loss):

    def __init__(self, margin=0.5, scale=16.):
        
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)

    def call(self, y_true, y_pred):
        

        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

        cos_t_margin = tf.where(y_true==1., x=(cos_t * self.cos_m - sin_t * self.sin_m), y=cos_t)

        logits = (cos_t_margin) * self.scale
        
        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits, axis=-1)

        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config



class AdaptiveArcLoss(tf.keras.losses.Loss):

    def __init__(self, margin_initial=0.0, update_margin_rate=0.1, update_mean_angle_rate=0.05, margin_ratio=0.5, scale=16, mean_angle=pi/2):
        
        super().__init__()
        self.margin = tf.Variable(margin_initial, trainable=False)
        self.mean_angle = tf.Variable(mean_angle, trainable=False)
        self.update_margin_rate = update_margin_rate
        self.update_mean_angle_rate = update_mean_angle_rate
        self.margin_ratio = margin_ratio
        self.scale = scale

    def update_margin(self, y_true, y_pred):
        y_pred_no_grad = tf.stop_gradient(y_pred)
        theta = tf.math.acos(tf.boolean_mask(y_pred_no_grad, y_true==1.))
        self.mean_angle.assign( (1-self.update_mean_angle_rate) * self.mean_angle + self.update_mean_angle_rate * tf.math.reduce_mean(theta) )
        self.margin.assign( (1-self.update_margin_rate) * self.margin + self.margin_ratio * self.update_margin_rate * (pi/2 - self.mean_angle) )

    def call(self, y_true, y_pred):
        cos_m = tf.math.cos(self.margin)
        sin_m = tf.math.sin(self.margin)
        
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))
        
        cos_t_margin = tf.where(y_true==1., x=(cos_t * cos_m - sin_t * sin_m), y=cos_t)
        logits = (cos_t_margin) * self.scale
        
        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits, axis=-1)
        
        self.update_margin(y_true, y_pred)

        return losses

    def get_config(self):
        config = super(AdaptiveArcLoss, self).get_config()
        config.update({"margin": self.margin.numpy(), "mean_angle": self.mean_angle.numpy(), "update_margin_rate": self.update_margin_rate,
                      "update_mean_angle_rate": self.update_mean_angle_rate, "margin_ratio":self.margin_ratio, "scale": self.scale})
        return config
    
class AdaptiveArcLossVer2(tf.keras.losses.Loss):

    def __init__(self, initial_margin=0.2, update_mean_angle_rate=0.01, margin_ratio=0.5, scale=16, mean_angle=pi/2):
        
        super().__init__()
        self.mean_angle = tf.Variable(mean_angle, trainable=False)
        self.margin_ratio = margin_ratio
        self.margin = tf.Variable(initial_margin + margin_ratio * (pi/2 - mean_angle), trainable=False)
        self.update_mean_angle_rate = update_mean_angle_rate
        self.scale = scale
        self.initial_margin = initial_margin

    def update_margin(self, y_true, y_pred):
        y_pred_no_grad = tf.stop_gradient(y_pred)
        theta = tf.math.acos(tf.boolean_mask(y_pred_no_grad, y_true==1.))
        self.mean_angle.assign( (1-self.update_mean_angle_rate) * self.mean_angle + self.update_mean_angle_rate * tf.math.reduce_mean(theta) )
        self.margin.assign( self.initial_margin + self.margin_ratio * (pi/2 - self.mean_angle) )

    def call(self, y_true, y_pred):
        cos_m = tf.math.cos(self.margin)
        sin_m = tf.math.sin(self.margin)
        
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))
        
        cos_t_margin = tf.where(y_true==1., x=(cos_t * cos_m - sin_t * sin_m), y=cos_t)
        logits = (cos_t_margin) * self.scale
        
        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits, axis=-1)
        
        self.update_margin(y_true, y_pred)

        return losses

    def get_config(self):
        config = super(AdaptiveArcLossVer2, self).get_config()
        config.update({"margin": self.margin.numpy(), "mean_angle": self.mean_angle.numpy(), "initial_margin": self.initial_margin,
                      "update_mean_angle_rate": self.update_mean_angle_rate, "margin_ratio":self.margin_ratio, "scale": self.scale})
        return config