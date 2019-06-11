from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, load_model
import keras.backend as K

import model as tcav_model
import tcav as tcav
import utils as utils
import activation_generator as act_gen
import tensorflow as tf
import utils_plot as utils_plot

sess = K.get_session()

# Your code for training and creating a model here. In this example, I saved the model previously
# using model.save and am loading it again in keras here using load_model.
model = load_model('./experiment_models/model.h5')

# Modified version of PublicImageModelWrapper in TCAV's models.py
# This class takes a session which contains the already loaded graph.
# This model also assumes softmax is used with categorical crossentropy.
class CustomPublicImageModelWrapper(tcav_model.ImageModelWrapper):
    def __init__(self, sess, label_path, image_shape, endpoints_dict, name, image_value_range):
        super(self.__class__, self).__init__(image_shape)
        
        self.sess = sess
        self.labels = tf.gfile.Open(label_path).read().splitlines()
        self.model_name = name
        self.image_value_range = image_value_range

        # get endpoint tensors
        self.ends = {'input': endpoints_dict['input_tensor'], 'prediction': endpoints_dict['prediction_tensor']}
        
        self.bottlenecks_tensors = self.get_bottleneck_tensors()
        
        # load the graph from the backend
        graph = tf.get_default_graph()

        # Construct gradient ops.
        with graph.as_default():
            self.y_input = tf.placeholder(tf.int64, shape=[None])

            self.pred = tf.expand_dims(self.ends['prediction'][0], 0)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.one_hot(
                        self.y_input,
                        self.ends['prediction'].get_shape().as_list()[1]),
                    logits=self.pred))
        self._make_gradient_tensors()

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)

    @staticmethod
    def create_input(t_input, image_value_range):
        """Create input tensor."""
        def forget_xy(t):
            """Forget sizes of dimensions [1, 2] of a 4d tensor."""
            zero = tf.identity(0)
            return t[:, zero:, zero:, :]

        t_prep_input = t_input
        if len(t_prep_input.shape) == 3:
            t_prep_input = tf.expand_dims(t_prep_input, 0)
        t_prep_input = forget_xy(t_prep_input)
        lo, hi = image_value_range
        t_prep_input = lo + t_prep_input * (hi-lo)
        return t_input, t_prep_input

    @staticmethod
    def get_bottleneck_tensors():
        """Add Inception bottlenecks and their pre-Relu versions to endpoints dict."""
        graph = tf.get_default_graph()
        bn_endpoints = {}
        for op in graph.get_operations():
            # change this below string to change which layers are considered bottlenecks
            # use 'ConcatV2' for InceptionV3
            # use 'MaxPool' for VGG16 (for example)
            if 'ConcatV2' in op.type:
                name = op.name.split('/')[0]
                bn_endpoints[name] = op.outputs[0]
            
        return bn_endpoints
      
      
      
# input is the first tensor, logit and prediction is the final tensor.
# note that in keras, these arguments should be exactly the same for other models (e.g VGG16), except for the model name
endpoints_v3 = dict(
    input=model.inputs[0].name,
    input_tensor=model.inputs[0],
    logit=model.outputs[0].name,
    prediction=model.outputs[0].name,
    prediction_tensor=model.outputs[0],
)

# endpoints_v3 should look like this
#endpoints_v3 = {
#    'input': 'input_1:0',
#    'input_tensor': <tf.Tensor 'input_1:0' shape=(?, 224, 224, 3) dtype=float32>,
#    'logit': 'dense_2/Softmax:0',
#    'prediction': 'dense_2/Softmax:0',
#    'prediction_tensor': <tf.Tensor 'dense_2/Softmax:0' shape=(?, 2) dtype=float32>
#}


# instance of model wrapper, change the labels and other arguments to whatever you need
mymodel = CustomPublicImageModelWrapper(sess, 
        ['nv', 'mel'], [224, 224, 3], endpoints_v3, 
        'InceptionV3_public', (-1, 1))

# create your image activation generator, see Run TCAV notebook for info on these arguments
act_generator = act_gen.ImageActivationGenerator(mymodel, source_dir, activation_dir, max_examples=200)

# again, see Run TCAV for more details
mytcav = tcav.TCAV(sess,
        target, concepts, bottlenecks,
        act_generator, alphas,
        cav_dir=cav_dir,
        num_random_exp=11)

results = mytcav.run(run_parallel=False)

utils_plot.plot_results(results, num_random_exp=11)