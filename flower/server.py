import flwr as fl
from typing import List, Optional, Tuple
from flwr.common import Weights
from ddaba import DDaBA
import tensorflow as tf
    
strategy = DDaBA()
fl.server.start_server(config={"num_rounds": 10}, strategy = strategy)



