from multipledispatch import dispatch
from multipledispatch.variadic import Variadic

from numpy import linalg as LA
import numpy as np

import shfl

from shfl.federated_aggregator.federated_aggregator import FederatedAggregator


class WeightedFedAvgAggregator(FederatedAggregator):
    """
    Implementation of Weighted Federated Averaging Aggregator.
    The aggregation of the parameters is weighted by the number of data \
    in every node.

    It implements [Federated Aggregator](../federated_aggregator/#federatedaggregator-class)
    """

    def aggregate_weights(self, clients_params):
        """
        Implementation of abstract method of class
        [AggregateWeightsFunction](../federated_aggregator/#federatedaggregator-class)

        # Arguments:
            clients_params: list of multi-dimensional (numeric) arrays.
            Each entry in the list contains the model's parameters of one client.

        # Returns:
            aggregated_weights: aggregator weights representing the global learning model
        """
        ponderated_weights = [self._ponderate_weights(i_client, i_weight)
                              for i_client, i_weight
                              in zip(clients_params, self._percentage)]

        return self._aggregate(*ponderated_weights)

    @dispatch((np.ndarray, np.ScalarType), np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Weighting of arrays"""
        return params * weight

    @dispatch(list, np.ScalarType)
    def _ponderate_weights(self, params, weight):
        """Weighting of (nested) lists of arrays"""
        ponderated_weights = [self._ponderate_weights(i_params, weight)
                              for i_params in params]
        return ponderated_weights

    @dispatch(Variadic[np.ndarray, np.ScalarType])
    def _aggregate(self, *ponderated_weights):
        """Aggregation of ponderated arrays"""
        return np.sum(np.array(ponderated_weights), axis=0)

    @dispatch(Variadic[list])
    def _aggregate(self, *ponderated_weights):
        """Aggregation of ponderated (nested) lists of arrays"""
        aggregated_weights = [self._aggregate(*params)
                              for params in zip(*ponderated_weights)]
        return aggregated_weights


class DDaBA(WeightedFedAvgAggregator):
    """
    Class of the IOWA version of [WeightedFedAvgAggregator](../federated_aggregator/#weightedfedavgaggregator-class)
    """

    def __init__(self):
        super().__init__()
        

    def q_function(self, x):
        if x <= self._b:
            return x / self._b * self._y_b
        elif x <= self._c:
            return (x - self._b) / (self._c - self._b) * (1 - self._y_b) + self._y_b
        else:
            return 1
        
    def set_performance(self, performance):
        self._performance = performance
        self._get_ponderation_weights()

    def _get_ponderation_weights(self):
        #We sort the vector of clients' performance
        ordered_idx = np.argsort(-self._performance)
        undo_argsort = np.argsort(ordered_idx)
        self._performance = self._performance[ordered_idx]
        num_clients = len(self._performance)
        
        #We set the exponential distribution
        exp_distribution = np.asarray([self._performance[0] - i for i in self._performance])
        
        print(exp_distribution)

        dist_lambda = 1/np.mean(exp_distribution)
        first_quartil = np.log(10/9)/dist_lambda
        outliers = (np.log(4) + 1.5*np.log(3))/dist_lambda
        
        self._b = len(exp_distribution[exp_distribution < first_quartil])/num_clients
        self._c = 1 - len(exp_distribution[exp_distribution > outliers])/num_clients
        
        
        self._y_b = 2*self._b*num_clients/(2*self._b*num_clients + (self._c-self._b)*num_clients)
        
        print(self._b)
        print(self._c)

        self._percentage = np.array([self.q_function((i+1)/num_clients) - self.q_function(i/num_clients) for i in range(num_clients)])  
        print(self._percentage)
        
        #Undo argsort
        self._percentage = self._percentage[undo_argsort]
