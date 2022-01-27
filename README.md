# Few-shot-learning-with-adversarial-reprograming-neural-net
This repository contains few shot learning with ensemble adversarial reprogramming of pretrained neural net.
Pretrained network which used in this project is resnet18. for better results, you can use a better-pretrained network like Efficient net.
For train, you should follow all codes in one script and then run. (add other codes to main file.)
Train contains Omniglot dataset with 100 episodes. you can change default numbers which are num epoch, num ensemble, and...


# training
python main.py --num_epoch 100 --lr 0. --num_ensembles 10
