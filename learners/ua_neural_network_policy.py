from .neural_network_policy import NeuralNetworkPolicy


class UANeuralNetworkPolicy(NeuralNetworkPolicy):

     def predict(self, observation, metadata):
        action, uncertainty = self.parametrization.predict([observation])
        return action, uncertainty