

## Installation 

The environment has been tested with Python 3.8 and the versions specified in the requirements.txt file

### Extra stuff
Do it trust me, it will run
```
pip install ipython
```

```bash
cd atcenv
pip install -r requirements.txt
python setup.py install
```

## Usage

```python
from atcenv import Environment

# create environment
env = Environment()

# reset the environment
obs = env.reset()

# set done status to false
done = False

# execute one episode
while not done:
    # compute the best action with your reinforcement learning policy
    action = ...

    # perform step
    obs, rew, done, info = env.step(action)
    
    # render (only recommended in debug mode)
    env.render()

env.close()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
