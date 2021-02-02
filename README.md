# Tensorflow Implementation of Performer
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/performer/performer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


An implementation of <a href="https://arxiv.org/abs/2009.14794">Performer</a>, a linear attention-based transformer variant with a **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features approach (FAVOR+).


<img src="https://imgur.com/anaqXSD.png" width="500px"></img>


### Initial Setup
Create a Python 3 virtual environment and activate:
```
virtualenv -p python3 env
source ./env/bin/activate
```
Install requirements by running:
```
pip install -r requirements.txt
```
Then export project to python path:
```
export PYTHONPATH=$PATH_TO_REPO/performer
```
To test the scripts, run `pytest` in the root directory, you may wish to
install `pytest` separately

### Usage
`Performer` inherites from a lightly modified version of tf-nightly's `MultiHeadAttention` and is made to be fully
compatible with the parents' use cases, with added flexibility for performing attention in linear time and space complexity. Currently masked attention is not supported.
```python
from performer.networks.linear_attention import Performer

layer = Performer(num_heads=2, # Number of attention heads
                  key_dim=2, # Size of each attention head for query and key
                  attention_method='linear', # attention method, 'linear' or 'quadratic'
		  supports=2, # only used in 'linear' attention, number of random features
		  attention_axes=None # axes over which the attention is applied.
		  )
query = tf.keras.Input(shape=[8, 16])
key = tf.keras.Input(shape=[4, 16])
output_tensor = layer([query, key])
print(output_tensor.shape)
# (None, 8, 16)
```

`Performer` supports attention in any arbituary axis, below is an example of 2D
self-attention over a 5D input tensor on axes 2 and 3.

```python
layer = Performer(num_heads=2, key_dim=2, attention_method='linear',
                  supports=10, attention_axes=(2, 3))
input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
output_tensor = layer([input_tensor, input_tensor])
print(output_tensor.shape)
# (None, 5, 3, 4, 16)
```



### Citations

```bibtex
@misc{choromanski2020rethinking,
    title   = {Rethinking Attention with Performers},
    author  = {Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
    year    = {2020},
    eprint  = {2009.14794},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
