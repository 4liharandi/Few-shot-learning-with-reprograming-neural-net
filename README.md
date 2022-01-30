# Few-shot-learning-with-adversarial-reprograming-neural-net
This repository contains few shot learning with ensemble adversarial reprogramming of pretrained neural net.
Addversarial Reporogramming come from paper "Adversarial Reprogramming of Neural Networks", Gamaleldin F. Elsayed 2018.
Pretrained network which used in this project is resnet18. for better results, you can use a better-pretrained network like Efficient net.
For train, you should follow all codes in one script and then run.


# training

 python main.py --num_epoch 100 --lr 0. --num_ensembles 10
 
 In training several accuray shown which based on how to ensemble results:
 1.average of all distribuations 
 2.weighted average of all distribuations 
 3.choise most repeted class 
 4.choise best ensemble
