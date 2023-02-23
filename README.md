# Multi-Lunar-Lander-Environment - Video Demo

This is my course project for CSCI 7000 - Deep Reinforcement Learning for Robotics in Fall 2023 at University of Colorado Boulder.
Our work creates a multi-agent lunar lander environment akin to the OpenAI lunar lander following the PettingZoo API. We have evaluated the environment we built
successfully on a number of compliance tests that PettingZoo provides. We implemented agents using Deep QNetworks and evaluated their performance in our environment.

> If you like my work, I am interested in summer 2023 internship opportunities - gapa2065@colorado.edu or https://www.linkedin.com/in/siva-gangadhar-pabbineedi-a118a319a/

[![Watch the video](https://img.youtube.com/vi/zEJbj0CxVS0/maxresdefault.jpg)](https://youtu.be/zEJbj0CxVS0)

### You can run the code using following python command:

TO TRAIN THE ALGORITHM
```python ./pettingzoo/sisl/train_lunar.py```

TO VISUALIZE THE ENVIRONMENT
```python ./pettingzoo/sisl/run_lunar.py``` 

TO FIND THE MULTILUNARLANDER FILE THAT WE WROTE SEE
```multilunarlander_base.py``` in ```sisl``` folder

TO SWITCH OFF THE RENDERING FOR FASTER TRAINING
make ```render = None``` in MultiLunarLanderEnv class in ```multilunarlander_base.py```

## End Note:

DQN implementation was an adaptation from https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html.
Most of the code in this repo belongs to PettingZoo (https://pettingzoo.farama.org/). We were only trying to add one more interesting environment to it in the form
of the file ```multilunarlander_base.py``` in ```sisl``` folder.

> NO COPYRIGHT INFRINGEMENT INTENDED WHATSOEVER. PLEASE MESSAGE ME IF YOU HAVE ANY OBJECTION TO ANYTHING IN THIS REPO.
