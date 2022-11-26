

## Installation 

The environment has been tested with Python 3.8 and the versions specified in the requirements.txt file

### Extra stuff
No it is not a jupyter thing, it is wild but it will work. It is maintianed by jupyter team but you may use it as a package.
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
