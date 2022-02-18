import shfl
import numpy as np
from ddaba import DDaBA

class DynamicFederatedGovernment(shfl.federated_government.FederatedGovernment):
    """
    Class used to represent the IOWA Federated Government which implements [FederatedGovernment](../federated_government/#federatedgovernment-class)

    # Arguments:
        model_builder: Function that return a trainable model (see: [Model](../model))
        federated_data: Federated data to use. (see: [FederatedData](../private/federated_operation/#federateddata-class))
        aggregator: Federated aggregator function (see: [Federated Aggregator](../federated_aggregator))
        model_param_access: Policy to access model's parameters, by default non-protected (see: [DataAccessDefinition](../private/data/#dataaccessdefinition-class))
        dynamic: boolean indicating if we use the dynamic or static version (default True)
        a: first argument of linguistic quantifier (default 0)
        b: second argument of linguistic quantifier (default 0.2)
        c: third argument of linguistic quantifier (default 0.8)
        y_b: fourth argument of linguistic quantifier (default 0.4)
        k: distance param of the dynamic version (default 3/4)
    """

    def __init__(self, model_builder, federated_data, model_params_access=None, dynamic=True, a=0,
                 b=0.2, c=0.8, y_b=0.4, k=3/4):
        super().__init__(model_builder, federated_data, DDaBA(), model_params_access)

        self._a = a
        self._b = b
        self._c = c
        self._y_b = y_b
        self._k = k
        self._dynamic = dynamic

    def performance_clients(self, data_val, label_val):
        """
        Evaluation of local learning models over global test dataset.

        # Arguments:
            val_data: validation dataset
            val_label: corresponding labels to validation dataset

        # Returns:
            client_performance: Performance for each client.
        """
        client_performance = []
        for data_node in self._federated_data:
            # Predict local model in test
            local_performance = data_node.performance(data_val, label_val)
            client_performance.append(local_performance)

        return np.array(client_performance)

    def run_rounds(self, n, test_data, test_label):
        """
        Implementation of the abstract method of class [FederatedGovernment](../federated_government/#federatedgoverment-class)

        Run one more round beginning in the actual state testing in test data and federated_local_test.

        # Arguments:
            n: Number of rounds
            test_data: Test data for evaluation between rounds
            test_label: Test label for evaluation between rounds

        """
        randomize = np.arange(len(test_label))
        np.random.shuffle(randomize)
        test_data = test_data[randomize, ]
        test_label = test_label[randomize]

        # Split between validation and test
        validation_data = test_data[:int(0.15*len(test_label)), ]
        validation_label = test_label[:int(0.15*len(test_label))]

        test_data = test_data[int(0.15 * len(test_label)):, ]
        test_label = test_label[int(0.15 * len(test_label)):]

        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.train_all_clients()
            self.evaluate_clients(test_data, test_label)
            client_performance = self.performance_clients(validation_data, validation_label)
            self._aggregator.set_ponderation(client_performance, self._dynamic, self._a, self._b, self._c, self._y_b,
                                             self._k)
            self.aggregate_weights()
            self.evaluate_global_model(test_data, test_label)
            print("\n\n")

            
class DDaBAFederatedGovernment(DynamicFederatedGovernment):
    
    def __init__(self, model_builder, federated_data, aggregator):
        super().__init__(model_builder, federated_data)
        self._aggregator = aggregator

    
    def evaluate_global_model(self, data_test, label_test):
        evaluation = self._model.evaluate(data_test, label_test)
        print("Global model test performance : " + str(evaluation))
        return evaluation
    
    def aggregate_selected_weights(self, num_round = None, data_val = None, label_val = None):
        total_nodes = self._federated_data.num_nodes()
        
        global_weights = self._model.get_model_params()
         
        weights = []
        client_performance = []
        for i, data_node in enumerate(self._federated_data):
            data_node.train_model()
            client_weights = data_node.query_model_params()

            combined_weights = None

            combined_weights = [i - j for i,j in zip(client_weights, global_weights)]
            weights.append(combined_weights)

            #DDaBA
            self._model.set_model_params([i + j for i,j in zip(global_weights, combined_weights)])
            client_performance.append(self._model.evaluate(data_val, label_val)[1])
            #
        
        #DDaBA
        self._aggregator.set_performance(np.array(client_performance))
        #
        
        aggregated_weights = self._aggregator.aggregate_weights(weights)

        combined_aggregated_weights = [i + j for i,j in zip(global_weights, aggregated_weights)]
        self._model.set_model_params(combined_aggregated_weights)
        
    def run_rounds(self, n, test_data, test_label):
        evaluations = []
        
        test_size = 0.8

        validation_data = test_data[int(test_size*test_data.shape[0]):]
        test_data = test_data[0:int(test_size*test_data.shape[0])]
    
        validation_labels = test_label[int(test_size*test_label.shape[0]):]
        test_label = test_label[0:int(test_size*test_label.shape[0])]
        
        for i in range(0, n):
            print("Accuracy round " + str(i))
            self.deploy_central_model()
            self.aggregate_selected_weights(i, validation_data, validation_labels)
            print("\nEvaluating global task:")
            evaluations.append(self.evaluate_global_model(test_data, test_label))
            print("\n\n")
            
        return evaluations