# Tensorflow Implementation of Performer
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/xl402/performer/performer)


An implementation of <a href="https://arxiv.org/abs/2009.14794">Performer</a>, a linear attention-based transformer variant with a **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features approach (FAVOR+).


<img src="https://imgur.com/anaqXSD.png" width="500px"></img>


### Initial Setup
#### Install dependencies
This repo requires Python 3.7 and above. Install requirements by running:
```
pip install -r requirements.txt
```
Then expoert `src` to path:
```
export PYTHONPATH=PATH_TO_REPO/src
```
To test the scripts, run `pytest` in the root directory
