# subgoal-based-reward-shaping
This is for implementations of Subgoal-based Reward Shaping.

# Setup
This implementation requires external packages, and you first install the packages by pip or other package management tools.

```
pip install -r requirements
```

# Run
You can run a learning agent in each domain. 
## Navigation in a four-room domain

```
python -m fourroom --config=in/configs/fourooms.ini
```

## Navigation in a pinball domain

```
python -m pinball --config=in/configs/pinball.ini
```

## Fetch robot picks and places

```
python -m picknplace --config=in/config/picknplace.ini
```

# Analysis

The program for analysis is [here](https://github.com/takato86/analyze_result).