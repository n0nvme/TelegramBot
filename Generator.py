import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle

corpora = ""


with open("Untitled.txt", encoding='utf-8') as fin:
    text = fin.read()
    corpora += text

with open("chuck_pr.txt", encoding='utf-8') as fin:
    text = fin.read()
    corpora += text

corpora = corpora.replace('\x00', '')
corpora = corpora.replace('-', '')
corpora = corpora.replace('\n', ' ')
corpora = corpora.replace('ё', 'е')

#тут будут все уникальные токены (буквы, цифры)
tokens = np.unique(list(corpora))

tokens = list(tokens)

assert len(tokens) == 71

token_to_id = {token: i for i, token in enumerate(tokens)}

id_to_token = {i:token for token, i in token_to_id.items()}

#Преобразуем всё в токены
corpora_ids = [token_to_id[symbol] for symbol in corpora]


def sample_random_batches(source, n_batches=10, seq_len=20):
    X_batch = []
    y_batch = []

    for i in range(n_batches):
        start = np.random.randint(0, len(source) - seq_len - 2)

        x = source[start: start + seq_len]
        y = source[start + seq_len]

        X_batch.append(x)
        y_batch.append(y)

    return X_batch, y_batch

#длина последоватеьности при обучении (как далеко распространяются градиенты)
seq_length = 20

# Максимальный модуль градиента
grad_clip = 5

input_sequence = T.matrix('input sequence','int32')
target_values = T.ivector('target y')

n_tokens = len(tokens)

input0 = lasagne.layers.InputLayer(shape=(None, None),input_var=input_sequence)

emb0 = lasagne.layers.EmbeddingLayer(input0, n_tokens, n_tokens)

lstm0 = lasagne.layers.LSTMLayer(emb0, 256, grad_clipping=grad_clip)
lstm1 = lasagne.layers.LSTMLayer(lstm0, 256, only_return_final=True, grad_clipping=grad_clip)

dense0 = lasagne.layers.DenseLayer(lstm1, 256)

output0 = lasagne.layers.DenseLayer(dense0, n_tokens, nonlinearity=lasagne.nonlinearities.softmax)

lasagne.layers.set_all_param_values(output0, pickle.load(open("mat_dump.pkl", "rb")))

# Веса модели
weights = lasagne.layers.get_all_params(output0,trainable=True)

network_output = lasagne.layers.get_output(output0)
#если вы используете дропаут - не забудьте продублировать всё в режиме deterministic=True

loss = lasagne.objectives.categorical_crossentropy(network_output, target_values).mean()

updates = lasagne.updates.adam(loss, weights, learning_rate=0.01)

#функция потерь без обучения
compute_cost = theano.function([input_sequence, target_values], loss, allow_input_downcast=True)

# Вероятности с выхода сети
probs = theano.function([input_sequence],network_output,allow_input_downcast=True)


def max_sample_fun(probs):
    return np.argmax(probs)


def proportional_sample_fun(probs, T=2.):
    new_probs = (probs ** T)
    new_probs /= new_probs.sum()

    return np.random.choice(np.arange(0, len(tokens)), p=new_probs)


# The next function generates text given a phrase of length at least SEQ_LENGTH.
# The phrase is set using the variable generation_phrase.
# The optional input "N" is used to set the number of characters of text to predict.

def generate_sample(sample_fun, seed_phrase=None, N=200):
    '''
    Сгенерировать случайный текст при помощи сети

    sample_fun - функция, которая выбирает следующий сгенерированный токен

    seed_phrase - фраза, которую сеть должна продолжить. Если None - фраза выбирается случайно из corpora

    N - размер сгенерированного текста.

    '''

    if seed_phrase is None:
        start = np.random.randint(0, len(corpora) - seq_length)
        seed_phrase = corpora[start:start + seq_length]
        print("Using random seed:", seed_phrase)
    while len(seed_phrase) < seq_length:
        seed_phrase = " " + seed_phrase
    if len(seed_phrase) > seq_length:
        seed_phrase = seed_phrase[len(seed_phrase) - seq_length:]
    assert type(seed_phrase) is str

    sample_ix = []
    x = list(map(lambda c: token_to_id.get(c, 0), seed_phrase))
    x = np.array([x])

    for i in range(N):
        # Pick the character that got assigned the highest probability
        ix = sample_fun(probs(x).ravel())
        # Alternatively, to sample from the distribution instead:
        # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
        sample_ix.append(ix)
        x[:, 0:seq_length - 1] = x[:, 1:]
        x[:, seq_length - 1] = 0
        x[0, seq_length - 1] = ix

    random_snippet = seed_phrase + ''.join(id_to_token[ix] for ix in sample_ix)
    print("----\n %s \n----" % random_snippet)

seed = u"Каждый человек должен"
sampling_fun = proportional_sample_fun
result_length = 300

generate_sample(sampling_fun,seed,result_length)



