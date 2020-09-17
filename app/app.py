import tensorflow_hub as hub
import tensorflow as tf
import os
import re
import numpy as np
import tensorflow.keras as keras
from tqdm import tqdm_notebook
from bert import bert_tokenization
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import set_session
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import multi_gpu_model
from waitress import serve

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 512
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module(tf_hub):
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(tf_hub)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )
    return bert_tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
def convert_single_example(tokenizer, example, max_seq_length=512):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    #print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=512):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

class BertLayer(Layer):
    
    '''BertLayer which support next output_representation param:
    
    pooled_output: the first CLS token after adding projection layer () with shape [batch_size, 768]. 
    sequence_output: all tokens output with shape [batch_size, max_length, 768].
    mean_pooling: mean pooling of all tokens output [batch_size, max_length, 768].
        You can simple fine-tune last n layers in BERT with n_fine_tune_layers parameter. For view trainable parameters call model.trainable_weights after creating model.
    
    '''
    
    def __init__(self, n_fine_tune_layers=3,output_size = 768, tf_hub = None, output_representation = 'pooled_output',supports_masking = True, is_trainable = False, **kwargs):
        
        self.n_fine_tune_layers = n_fine_tune_layers
        self.is_trainable = is_trainable
        self.output_size = 768
        self.tf_hub = tf_hub
        self.output_representation = output_representation
        self.supports_masking = True
        
        super(BertLayer, self).__init__(**kwargs)
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_fine_tune_layers': self.n_fine_tune_layers,
            'is_trainable': self.is_trainable,
            'output_size': self.output_size,
            'tf_hub': self.tf_hub,
            'output_representation': self.output_representation,
            'supports_masking': self.supports_masking,
        })
        return config
    def build(self, input_shape):
        self.bert = hub.Module(
            self.tf_hub,
            trainable=self.is_trainable,
            name="{}_module".format(self.name)
        )
        
        
        variables = list(self.bert.variable_map.values())
        if self.is_trainable:
            # 1 first remove unused layers
            trainable_vars = [var for var in variables if not "/cls/" in var.name]
            
            
            if self.output_representation == "sequence_output" or self.output_representation == "mean_pooling":
                # 1 first remove unused pooled layers
                trainable_vars = [var for var in trainable_vars if not "/pooler/" in var.name]
            # Select how many layers to fine tune
            trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
            
            # Add to trainable weights
            for var in trainable_vars:
                self._trainable_weights.append(var)

            # Add non-trainable weights
            for var in self.bert.variables:
                if var not in self._trainable_weights:
                    self._non_trainable_weights.append(var)
        else:
             for var in variables:
                self._non_trainable_weights.append(var)
                

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        
        if self.output_representation == "pooled_output":
            pooled = result["pooled_output"]
            
        elif self.output_representation == "mean_pooling":
            result_tmp = result["sequence_output"]
        
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result_tmp, input_mask)
            
        elif self.output_representation == "sequence_output":
            
            pooled = result["sequence_output"]
       
        return pooled
    def compute_mask(self, inputs, mask=None):
        
        if self.output_representation == 'sequence_output':
            inputs = [K.cast(x, dtype="bool") for x in inputs]
            mask = inputs[1]
            
            return mask
        else:
            return None
        
        
    def compute_output_shape(self, input_shape):
        if self.output_representation == "sequence_output":
            return (input_shape[0][0], input_shape[0][1], self.output_size)
        else:
            return (input_shape[0][0], self.output_size)

#Confidence estimation: please mind that it's only a proxy and does not represent a confidence interval

def get_confidence(array):
    elist=[]
    for i,ele in enumerate(array):
        elist.append((ele,i,))
    elist.sort(key=lambda x: x[0], reverse=True)
    return (elist[0][0]-elist[1][0])/elist[0][0]

sess = tf.Session()
global graph
graph = tf.get_default_graph()
set_session(sess)
model=load_model('/model/who_multibert_model.h5',custom_objects={'BertLayer':BertLayer})



from flask import Flask, Response, request
import pandas as pd
from io import StringIO
import json
import traceback
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
#cors = CORS(app, resources={r"/predict": {"origins": "*"}})

tokenizer = create_tokenizer_from_hub_module(bert_path)


@app.route("/")
def hello():
    return "This is a TF-serving app"

@app.route("/ping", methods=['GET'])
@cross_origin()
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    """
    try:
        return Response(response='{"status": "ok"}', status=200, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """
    Do an inference on a single request.
    """
    try:
        if request.content_type == 'application/json':
            data = request.get_json()
            names=['bias','clickbait','conspiracy','fake','hate','junksci','political','reliable','rumor','satire','unreliable']
            art = data["emm_text_text"] # make sure the column is correct
            test_examples = convert_text_to_examples([art], np.zeros(len([art])))
            (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)
            with graph.as_default():
                set_session(sess)
                prediction = model.predict([test_input_ids, test_input_masks, test_segment_ids], verbose = 1)
            inf = prediction[0].tolist()
            conf=get_confidence(prediction.flatten()) 
            data["probability"]=json.dumps(inf)
            data["confidence"]=str(conf)

            ##data.to_json('Predictions.json') #######  does it have to be a certain path?
        else:
            return Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')
        results_str = json.dumps(data)

        # return
        return Response(response=results_str, status=200, mimetype='application/json')
    except Exception:
        return traceback.format_exc()

@app.route('/invocations', methods=['POST'])
@cross_origin()
def batch_predict():
    """
    Do an inference on a batch of data.
    """
    try:
        if request.content_type == 'application/json':
            names=['bias','clickbait','conspiracy','fake','hate','junksci','political','reliable','rumor','satire','unreliable']
            data = request.get_json()
            response_list = []
            for i in data:
                art = i["emm_text_text"] # make sure the column is correct
                test_examples = convert_text_to_examples([art], np.zeros(len([art])))
                (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)
                with graph.as_default():
                    set_session(sess)
                    prediction = model.predict([test_input_ids, test_input_masks, test_segment_ids], verbose = 1)
                inf = prediction[0].tolist()
                conf=get_confidence(prediction.flatten()) 
                i["probability"]=json.dumps(inf)
                i["confidence"]=str(conf)
                response_list.append(i)
            ##data.to_json('Predictions.json') #######  does it have to be a certain path?
        else:
            return Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')
        results_str = json.dumps(response_list)
        # return
        return Response(response=results_str, status=200, mimetype='application/json')
    except Exception:
        return traceback.format_exc()


if __name__ == "__main__":

    # Only for debugging while developing
    # app.run(host='0.0.0.0', debug=False, port=8000)
    # To be used for production (Waitress)
    serve(app, host='0.0.0.0', port=8000)