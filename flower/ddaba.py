"""Dynamic Defense against Byzantie Poisoning Attacks (DDaBA)[Rodríguez-Barroso et al., 2021] strategy.
"""

from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy import Strategy

import numpy as np
from numpy import linalg as LA

import tensorflow as tf

data_shape_list = []

def _serialize(data):
    
    '''
    Serialize function. 

    '''
    global data_shape_list
    data = [np.array(j) for j in data]
    data_shape_list = [j.shape for j in data]
    serialized_data = [j.ravel() for j in data]
    serialized_data = np.hstack(serialized_data)
    return serialized_data

def _deserialize(data):
    '''
    Deserialize function. 

    '''
    global data_shape_list
    firstInd = 0
    deserialized_data = []
    for shp in data_shape_list:
        if len(shp) > 1:
            shift = np.prod(shp)
        elif len(shp) == 0:
            shift = 1
        else:
            shift = shp[0]
        tmp_array = data[firstInd:firstInd+shift]
        tmp_array = tmp_array.reshape(shp)
        deserialized_data.append(tmp_array)
        firstInd += shift
    return deserialized_data

def _evaluate_model(weights):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    
    x_val, y_val = x_train[0:int(0.15*len(x_train))], y_train[0:int(0.15*len(x_train))]

    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    use_nchw_format = False
    data_format = 'channels_first' if use_nchw_format else 'channels_last'
    data_shape = (1, 28, 28) if use_nchw_format else (28, 28, 1)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=data_shape, data_format=data_format))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', data_format=data_format))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format=data_format))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    losses = []
    
    for w in weights:
        model.set_weights(w)
        loss, _ = model.evaluate(x_val, y_val)
        losses.append(loss)
        
    return losses

def q_function(x, b, y_b, c):
        if x <= b:
            return x / b * y_b
        elif x <= c:
            return (x - b) / (c - b) * (1 - y_b) + y_b
        else:
            return 1

def _get_ponderation_weights(losses):
    #We sort the vector of clients' performance
    print(losses)
    ordered_idx = np.argsort(-losses)
    undo_argsort = np.argsort(ordered_idx)
    losses = losses[ordered_idx]
    num_clients = len(losses)

    #We set the exponential distribution
    exp_distribution = np.asarray([losses[0] - i for i in losses])

    print(exp_distribution)

    dist_lambda = 1/np.mean(exp_distribution)
    first_quartil = np.log(10/9)/dist_lambda
    outliers = (np.log(4) + 1.5*np.log(3))/dist_lambda

    b = len(exp_distribution[exp_distribution < first_quartil])/num_clients
    c = 1 - len(exp_distribution[exp_distribution > outliers])/num_clients


    y_b = 2*b*num_clients/(2*b*num_clients + (c-b)*num_clients)


    ponderation = np.array([q_function((i+1)/num_clients, b, y_b, c) - q_function(i/num_clients, b, y_b, c) for i in range(num_clients)])  

    #Undo argsort
    ponderation = ponderation[undo_argsort]
    
    return ponderation


def aggregate_rfout(results: List[Tuple[Weights, int]]) -> Weights:
    '''
    Agregation function using RFOut. 

    '''
    
    #Get serialized weights of clients
    weights = [np.array([layer for layer in w]) for w, num_examples in results]
    
    losses = _evaluate_model(weights)
    
    ponderation = _get_ponderation_weights(np.asarray(losses))
    
    ponderated_weights = [p*w for p, w in zip(ponderation, weights)]
    
    return sum(ponderated_weights)
    


class DDaBA(Strategy):
    """Configurable DDaBA strategy implementation, based on the FedAvg strategy implementation (https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedavg.py)."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
       
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

    def __repr__(self) -> str:
        rep = f"DDaBA(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            print(DEPRECATION_WARNING)
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if a centralized evaluation
        # function is provided
        if self.eval_fn is not None:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        # Performance
        return weights_to_parameters(aggregate_rfout(weights_results)), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                    evaluate_res.accuracy,
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {}