# Dynamic Defense Against Byzantine Poisoning Attacks in Federated Learning

Federated learning, as a distributed learning that conducts the training on the local devices without accessing to the training data, is vulnerable to Byzatine poisoning adversarial attacks.  We argue that the federated learning model has to avoid those kind of adversarial attacks through filtering out the adversarial clients by means of the federated aggregation operator. We propose a dynamic federated aggregation operator that dynamically discards those adversarial clients and allows to prevent the corruption of the global learning model. We assess it as a defense against adversarial attacks deploying a deep learning classification model in a federated learning setting on the Fed-EMNIST Digits, Fashion MNIST and CIFAR-10 image datasets. The results show that the dynamic selection of the clients to aggregate enhances the performance of the global learning model and discards the adversarial and poor (with low quality models) clients. The paper is available at [this link](https://www.sciencedirect.com/science/article/abs/pii/S0167739X22000784).

In this repository, we provide the implementation of the DDaBA federated aggregation operator in two Federated Learning frameworks, namely: Flower and Sherpa.ai Federated Learning.  Likewise, we show its behavior in each implementation on an image classification problem with the [ EMNIST Digits datset](https://www.nist.gov/itl/products-and-services/emnist-dataset).


## Implementation in Flower

We provide the implementation in  [Flower](https://flower.dev/).

**Requirements**. 

* The Flower framework. Follow the [official instructions](https://flower.dev/docs/installation.html) to install it.
* Python version >= 3.6.

**Usage**. You have to follow the following steps to run the image classification experiment with the Flower implementation and to use the code in [this directory](./flower/).

1- Open a terminal and start a server with the DDaBA strategy.

```
python ./flower/server.py
```

2- Run the first client in other terminal.

```
python ./flower/client.py
```

3- In different terminals, add as many clients as you want in the federated configuration (min. 2 according to the framework details).

```
python ./flower/client.py
```

The clients will show the results of the training in each learning of round and the server after each aggregation.

We also provide a [Jupyter notebook](./flower/ddaba.ipynb) to show its behavior with 2 clients.

## Implementation in Sherpa.ai Federaeted Learninng

We also provide the implementation in [Sherpa.ai Federated Learning](https://github.com/rbnuria/Sherpa.ai-Federated-Learning-Framework.git).

We provide a [Jupyter notebook](./shfl/ddaba.ipynb) in which we set up the entire federated setup of a simple image classification experiment and detail the code of the DDaBA aggregation mechanism. The [file rfout.py](./shfl/ddaba.py) contains the implementation of DDaBA.

**Requirements**. 

* The Sherpa.ai FL framework. Clone [this GitHub repository](https://github.com/rbnuria/Sherpa.ai-Federated-Learning-Framework.git).
* Python version >= 3.6.

**Usage**. Once you have clone the Github repository, move the [Jupyter notebook (ddaba.ipynb)](./shfl/ddaba.ipynb) and the implementation of the aggregation [python file (ddaba.py)](./shfl/ddaba.py) to the root directory and run all cells of the notebook.

## Citation
If you use this dataset, please cite:

```
Rodríguez-Barroso, N., Martínez-Cámara, E., Luzón, M. V., & Herrera, F. (2022). Dynamic defense against byzantine poisoning attacks in federated learning. Future Generation Computer Systems, 133, 1-9. https://doi.org/10.1016/j.future.2022.03.003

```
or 

```
@article{RODRIGUEZBARROSO20221,
title = {Dynamic defense against byzantine poisoning attacks in federated learning},
journal = {Future Generation Computer Systems},
volume = {133},
pages = {1-9},
year = {2022},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2022.03.003},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X22000784},
author = {Nuria Rodríguez-Barroso and Eugenio Martínez-Cámara and M. Victoria Luzón and Francisco Herrera},
keywords = {Federated learning, Deep learning, Adversarial attacks, Byzantine attacks, Dynamic aggregation operator},
abstract = {Federated learning, as a distributed learning that conducts the training on the local devices without accessing to the training data, is vulnerable to Byzantine poisoning adversarial attacks. We argue that the federated learning model has to avoid those kind of adversarial attacks through filtering out the adversarial clients by means of the federated aggregation operator. We propose a dynamic federated aggregation operator that dynamically discards those adversarial clients and allows to prevent the corruption of the global learning model. We assess it as a defense against adversarial attacks deploying a deep learning classification model in a federated learning setting on the Fed-EMNIST Digits, Fashion MNIST and CIFAR-10 image datasets. The results show that the dynamic selection of the clients to aggregate enhances the performance of the global learning model and discards the adversarial and poor (with low quality models) clients.}
}
```


## Contact
Nuria Rodríguez Barroso - rbnuria@ugr.es
