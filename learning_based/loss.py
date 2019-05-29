import tensorflow as tf
from utils import batch_matvec_mul
def loss_fun(xhat, x):
    loss = 0.
    for xhatk in xhat:
        lk = tf.losses.mean_squared_error(labels=x, predictions=xhatk)
        loss += lk
        tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
    #print "Only using the last layer loss"
    #loss = lk
    #print "regularizations added"
    #loss +=  0.01 * tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return loss

def loss_yhx(y, xhat, H):
    loss = 0.
    for xhatk in xhat:
        lk = tf.losses.mean_squared_error(labels=y, predictions=batch_matvec_mul(H,xhatk))
        loss += lk
        tf.add_to_collection(tf.GraphKeys.LOSSES, lk)
    return loss
