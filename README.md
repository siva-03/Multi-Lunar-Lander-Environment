# Multi-Lunar-Lander-Environment

This is my course project for CSCI 7000 - Deep Reinforcement Learning for Robotics in Fall 2023 at University of Colorado Boulder.
Our work creates a multi-agent lunar lander environment akin to the OpenAI lunar lander following the PettingZoo API. We have evaluated the environment we built
successfully on a number of compliance tests that PettingZoo provides. We implemented agents using Deep QNetworks and evaluated their performance in our environment.

[![Watch the video](https://img.youtube.com/vi/zEJbj0CxVS0/maxresdefault.jpg)](https://youtu.be/zEJbj0CxVS0)

###You can run the code using following python command:

TO TRAIN THE ALGORITHM
```python ./pettingzoo/sisl/train_lunar.py```

TO VISUALIZE THE ENVIRONMENT
```python ./pettingzoo/sisl/run_lunar.py``` 

TO FIND THE MULTILUNARLANDER FILE SEE 
```multilunarlander_base.py``` in ```sisl``` folder

TO SWITCH OFF THE RENDERING FOR FASTER TRAINING
make ```render = None``` in MultiLunarLanderEnv class in ```multilunarlander_base.py```


