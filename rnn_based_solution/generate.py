import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import helper
import random
#import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    #print("probabilities : ", len(probabilities))
    word_id = np.random.choice(len(probabilities), p=probabilities)
    #print("prob : ", probabilities[word_id])
    #print("mx prob : ", max(probabilities))
    #probabilities = np.array(probabilities).tolist()
    #local_prob = []
    #for idx in range(len(probabilities)):
    #    if probabilities[idx] > 0.01:
    #        local_prob.append(probabilities[idx])
    #word_id = np.random.choice(len(local_prob), p=local_prob)
    #word_id = probabilities.index(max(probabilities)) 
    #probabilities = np.array(probabilities).tolist()
    #word_id = probabilities.index(max(probabilities))
    return int_to_vocab[word_id]

def generate():
    gen_length = random.randint(6,12)
    # homer_simpson, moe_szyslak, or Barney_Gumble
    #prime_word = 'Sheldon'
    prime_word = 'sheldon'

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
        # Sentences generation setup
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})
            pred_word = pick_word(probabilities[0][0], int_to_vocab)
            gen_sentences.append(pred_word)
        # Remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')
            
        print(tv_script)

for i in range(100):
    generate()
