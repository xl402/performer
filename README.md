# Tensorflow Implementation of Performer
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/performer/performer)


An implementation of <a href="https://arxiv.org/abs/2009.14794">Performer</a>, a linear attention-based transformer variant with a **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features approach (FAVOR+).


<img src="https://imgur.com/anaqXSD.png" width="500px"></img>


### Initial Setup
This repo requires Python 3.7 and above. Install requirements by running:
```
pip install -r requirements.txt
```
Then export project to python path:
```
export PYTHONPATH=$PATH_TO_REPO/performer
```
To test the scripts, run `pytest` in the root directory

### Usage
`Performer` inherites tensorflow's `MultiHeadAttention` and is made to be fully
compatible with the parents' use cases, with added flexibility for performing attention in linear time and space complexity.
```python
from performer.networks.model import Performer

layer = Performer(num_heads=2, # Number of attention heads
                  key_dim=2, # Size of each attention head for query and key
                  attention_method='linear', # method for computing attention, 'linear' or 'quadratic'
		  supports=2, # only used in 'linear' attention, number of random features
		  attention_axes=None # axes over which the attention is applied.
		  )
target = tf.keras.Input(shape=[8, 16])
source = tf.keras.Input(shape=[4, 16])
output_tensor = layer(target, source)
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
