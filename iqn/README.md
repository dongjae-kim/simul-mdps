## Train
To run the script version:
`python run.py -info iqn_run1`

To run the script version on the Atari game Pong:
`python run.py -env PongNoFrameskip-v4 -info iqn_pong1`


### Observe training results
  `tensorboard --logdir=runs`
  

#### Dependencies
Trained and tested on:
<pre>
Python 3.6 
PyTorch 1.4.0  
Numpy 1.15.2 
gym 0.10.11 
</pre>

## From
[IQN-and-Extensions](https://github.com/BY571/IQN-and-Extensions)
