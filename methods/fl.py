"""
Base class for Federated Learning implementation.
This class serves as an abstract base class that defines the core functions for FL.
It is recommended not to modify this base class directly.
Instead, inherit from this class and implement the specific functionality in child classes
to maintain flexibility and extensibility of the FL framework.
"""

class FL:

    def __init__(self):
        """
        Initialize the FL class.
        Child classes should implement specific initialization logic.
        """
        pass

    def method_init(self):
        """
        Initialize the FL method specific parameters and configurations.
        Child classes should implement method-specific requirements and settings.
        """
        pass

    def server_init(self):
        """
        Initialize the FL server.
        Child classes should implement server-side initialization logic including model initialization.
        """
        pass

    def clients_init(self):
        """
        Initialize the FL clients.
        Child classes should implement client-side initialization logic including data distribution.
        """
        pass

    def candidates_sampling(self):
        """
        Sample candidates from the client pool for current round.
        Child classes should implement specific sampling strategies.
        """
        pass

    def clients_update(self):
        """
        Update clients' states and parameters.
        Child classes should implement how clients update their local models or states.
        """
        pass

    def server_update(self):
        """
        Update server's states and parameters.
        Child classes should implement how server aggregates and updates the global model.
        """
        pass

    def local_update(self):
        """
        Perform local update on client side.
        Child classes should implement local training or update logic.
        """
        pass
    
    def evaluate(self):
        """
        Evaluate the model performance.
        Child classes should implement specific evaluation metrics and procedures.
        """
        pass

    def training(self):
        """
        Execute the main training loop of federated learning.
        Child classes should implement the complete training workflow.
        """
        pass
