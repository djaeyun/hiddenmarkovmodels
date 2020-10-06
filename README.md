# Assignment 6: Hidden Markov Models


## Overview
Hidden Markov Models are used extensively in Artificial Intelligence, Pattern Recognition, Computer Vision, and many other fields.  If a system has unobservable (hidden) states and each state is independent of the prior, then we can create a model of that system using probability distributions over a sequence of observations.  The idea is that we can provide this system with a series of observations to use to query what is the most likely sequence of states that generated these observations.

## The Project
The goal of this project is to demonstrate the power of probabalistic models. This will build a word recognizer for American Sign Language (ASL) video sequences. In particular, this project employs [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research. This will be done by recognizing the x and y coordinates of the hand signals over time, while utilizing the HMM to generate the most likely sequences of letters to ulimately detect the final word.
