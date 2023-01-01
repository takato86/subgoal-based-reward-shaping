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
python -m fourroom --config=fourroom/in/configs/fourooms.ini
```

## Navigation in a pinball domain

```
python -m pinball --config=pinball/in/configs/pinball.ini
```

## Fetch robot picks and places

```
mpiexec -n 8 python -m picknplace --config=picknplace/in/configs/picknplace.ini
```

# Config file

We place a config file per algorithm and label the file the name of algorithm like `sarsa`.
If you want to try LEARNED with HUMAN, you select`learned_human`.

# Analysis

The program for analysis is [here](https://github.com/takato86/analyze_result).