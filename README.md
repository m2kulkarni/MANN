## Pytorch implementation of Memory Augmented Neural Network(Santoro et al.)

Download the omniglot Dataset from [here](https://github.com/brendenlake/omniglot) and put all the images(evaluation + background) in one folder. Then run resize_images.py there. 

Basic implementaion is in mann_pytorch.ipynb 

### Tasks Completed
- [x] Basic Implementation of MANN using LSTM
- [ ] Fix Training loss error
- [ ] Training for:-
    - 5 shot 1 way
    - 1 shot 5 way
- [ ] Use Bi-LSTM/multi layered LSTM
- [ ] Use NTM module

#### Loss stalls at 1.609 and Accuracy is very low(~20%)
![Tensorboard](tensorboard.png)
