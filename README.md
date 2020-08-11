# Assignment 6: Hidden Markov Models


## Overview
Hidden Markov Models are used extensively in Artificial Intelligence, Pattern Recognition, Computer Vision, and many other fields.  If a system has unobservable (hidden) states and each state is independent of the prior, then we can create a model of that system using probability distributions over a sequence of observations.  The idea is that we can provide this system with a series of observations to use to query what is the most likely sequence of states that generated these observations.

## The Project
The goal of this project is to demonstrate the power of probabalistic models. You will build a word recognizer for American Sign Language (ASL) video sequences. In particular, this project employs [hidden Markov models (HMM's)](https://en.wikipedia.org/wiki/Hidden_Markov_model) to analyze a series of measurements taken from videos of American Sign Language (ASL) collected for research (see the [RWTH-BOSTON-104 Database](http://www-i6.informatik.rwth-aachen.de/~dreuw/database-rwth-boston-104.php)).

In each video, an ASL signer is signing a meaningful sentence. In a typical ASL recognition system, you observe the XY coordinates of the speaker's left hand, right hand, and nose for every frame. The following diagram shows how the positions of the left hand (Red), right hand (Blue), and nose (Green) change over time in video number #66. Saturation of colors represents time elapsed.

<img src="./demo/hands_nose_position.png" alt="hands nose position">

In this assignment, for the sake of simplicity, you will only use the Y-coordinates of each hand to construct your HMM. In Part 1 you will build a one dimensional model, recognizing words based only on a series of right-hand Y coordinates; in Part 2 you will go multidimensional and utilize both hands. At this point, you will have two observed coordinates at each time step (frame) representing right hand & left hand Y positions.

The words you will be recognizing are "BUY", "HOUSE", and "CAR". These individual signs can be seen in the sign phrases from our dataset:

<img src="./demo/buy_house_slow.gif"> 

<p style="text-align:center; font-weight:bold"> JOHN CAN BUY HOUSE </p> 

<img src="./demo/buy_car_slow.gif"> 

<p style="text-align:center;  font-weight:bold"> JOHN BUY CAR [FUTURE] </p> 

