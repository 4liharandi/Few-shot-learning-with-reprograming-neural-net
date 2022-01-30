# Few-shot-learning-with-adversarial-reprograming-neural-net
This repository contains few shot learning with ensemble adversarial reprogramming of pretrained neural net.
Addversarial Reporogramming come from paper "Adversarial Reprogramming of Neural Networks", Gamaleldin F. Elsayed 2018.
Pretrained network which used in this project is resnet18. for better results, you can use a better-pretrained network like Efficient net.
For train, you should follow all codes in one script and then run.


# training

 python main.py --num_epoch 100 --lr 0.01 --num_ensembles 10 --num_episode 100 --num_way 5 --num_shot 5 --num_query 10
 
 For 10 ensemble for each episode need near 60 seconds time.
 
 In training several accuray shown which based on how to ensemble results:
 1.average of all distribuations 
 2.weighted average of all distribuations 
 3.choise most repeted class 
 4.choise best ensemble
