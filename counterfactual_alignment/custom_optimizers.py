import optax
from config import hyperparams

sgd_opt = optax.sgd(hyperparams['learning_rate'] ,hyperparams['momentum'] )
adam_opt = optax.adam(hyperparams['learning_rate'])