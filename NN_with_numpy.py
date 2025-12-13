import math
import time
import numpy as np

np.random.seed(42)

class Node():

    def __init__(self, node_type):
        self.value = None
        self.node_type = node_type

class Neuron(Node):

    def __init__(self, activation_function_type = "LINEAR", bias = np.random.uniform(low = -1, high = 1), weight = None):
        super().__init__("NEURON")
        self.bias = bias
        self.weight = weight
        self.activation_function_type = activation_function_type

    def set_weight(self, weight):
        self.weight = weight

    def set_bias(self,bias):
        self.bias = bias

    def calculate_output(self, connected_nodes):
        input_matrix = np.array([n.value for n in connected_nodes]).T
        pre_activated_value = np.dot(input_matrix,self.weight) + self.bias
        self.value = self.get_activated_value(pre_activated_value)

    def get_activated_value(self,value):
        if self.activation_function_type == "LINEAR":
            return value

        if self.activation_function_type == "RELU":
            return np.maximum(0, value)

    def get_derivative(self):
        if self.activation_function_type == "LINEAR":
            return np.ones_like(self.value)

        elif self.activation_function_type == "RELU":
            return np.where(self.value>0, 1, 0)

    def update(self,new_weight, bias):
        self.weight += new_weight
        self.bias += bias

class Input_Node(Node):

    def __init__(self):
        super().__init__("INPUT_NODE")

class Model():
    def __init__(self, layer, load = False):
        self.layer = layer
        self.loss_function_type = None
        self.learning_rate = None
        if not load:
            for layer_index,assign_layer in enumerate(layer[1:]):
                for neuron in assign_layer:
                    neuron.set_weight(np.array([np.random.uniform(low = -1, high = 1) for inputs in range(len(layer[layer_index]))]))

    def optimiser(self,loss_function,learning_rate):
        self.loss_function = loss_function.upper()
        self.learning_rate = learning_rate

    def fit(self, train_dataset, validation_dataset, epoch, shuffle = True, change_learning_rate = lambda learning_rate: learning_rate * 1, print_detail =1):
        if shuffle:
            np.random.shuffle(train_dataset)

        time_took_per_epoch = []
        lr_per_epoch = []
        mae_per_epoch =[]
        mse_per_epoch = []

        for epoch_i in range(epoch):
            print(f"\n\nEpoch: {epoch_i+1}\n----------------")
            start_time = time.perf_counter()
            lr_per_epoch.append(self.learning_rate)

            for batch_data in train_dataset:
                batch_label = batch_data[1]
                batch_input = batch_data[0]
                batch_size = len(batch_input)
                self.feed_foward(batch_input)
                self.backpropagation(batch_label,batch_size)

            end_time = time.perf_counter()
            time_took = end_time - start_time
            time_took_per_epoch.append(time_took)

            if print_detail >= 1:
                val_dataset_size = len(validation_dataset)
                if len(self.layer[-1]) == 1:
                    mae = 0
                    mse = 0
                    for ae, se in [self.get_accuracy(val_data) for val_data in validation_dataset]:
                        mae += ae
                        mse += se

                    mae /= val_dataset_size
                    mse /= val_dataset_size
                    mae_per_epoch.append(mae)
                    mse_per_epoch.append(mse)

                    print(f"VAL - MAE: {mae} MSE: {mse}")

                else:
                    pass

                print(f"Time Took: {time_took:.4f} secs\nLearning Rate: {self.learning_rate}\n")
                if print_detail >= 2:
                    for layer_i, layer in enumerate(self.layer):
                        print(f"Layer: {layer_i + 1}")
                        for node_i, node in enumerate(layer):
                            print(f"Node: {node_i + 1}  Type: {node.node_type}  ", end = "")
                            if node.node_type == "NEURON":
                                print(f"Weight: {node.weight}  Bias: {node.bias}", end = "")
                            print("")
                        print("")

            self.learning_rate = change_learning_rate(self.learning_rate)

        print("Finished Training!")
        return {"learning_rate": lr_per_epoch, "mae":mae_per_epoch, "mse":mse_per_epoch, "time_took_per_epoch":time_took_per_epoch, "total_time_took":sum(time_took_per_epoch)}

    def get_accuracy(self,dataset):
        predicts = self.feed_foward(dataset[0])
        if len(self.layer[-1]) == 1:
            return [abs(predicts[0] - dataset[1][0]),math.pow(predicts[0] - dataset[1][0],2)]
        else:
            pass

    def backpropagation(self,batch_label, batch_size):
        # output layer
        previous_layer = self.layer[-2]
        output_layer = self.layer[-1]

        predictions = np.array([n.value for n in output_layer]).T
        loss_gradients = self.get_loss_gradient(predictions,batch_label)
        output_derivatives = np.array([n.get_derivative() for n in output_layer]).T
        layer_deltas = loss_gradients * output_derivatives

        # hidden layer
        for layer_index in range(len(self.layer) - 1, 0, -1):
            current_layer = self.layer[layer_index]
            prev_layer = self.layer[layer_index - 1]
            prev_layer_values = np.array([n.value for n in prev_layer]).T

            # Calculate Gradients
            weight_gradients = np.dot(prev_layer_values.T, layer_deltas) / batch_size
            bias_gradients = np.sum(layer_deltas, axis=0) / batch_size

            # Update
            for i, neuron in enumerate(current_layer):
                neuron.update(-self.learning_rate * weight_gradients[:, i],
                              -self.learning_rate * bias_gradients[i])

            # Prepare Next Delta
            if layer_index > 1:
                current_weights = np.array([n.weight for n in current_layer])
                propagated_error = np.dot(layer_deltas, current_weights)
                prev_derivatives = np.array([n.get_derivative() for n in prev_layer]).T
                layer_deltas = propagated_error * prev_derivatives

    def feed_foward(self, batch_features):
        # Input Layer
        for node_index, node in enumerate(self.layer[0]):
            node.value = batch_features[:, node_index]

        # Hidden and Output Layers
        for layer_index, layer in enumerate(self.layer[1:]):
            previous_layer = self.layer[layer_index]
            for neuron in layer:
                neuron.calculate_output(previous_layer)

        return np.array([output_node.value for output_node in self.layer[-1]]).T

    def get_loss_gradient(self, values, target_values):
        if self.loss_function == "MEAN_SQUARED_ERROR":
            return values - target_values

        if self.loss_function == "MEAN_ABSOLUTE_ERROR":
            return np.where(values>target_values,1,-1)

    def predict(self,feature):
        return self.feed_foward(feature)

    def save_model(self, file_path):
        with open(file_path,"w") as file:
            file.write("position\ttype\tactivation_function\tbias\tweights")

            for layer_position, layer in enumerate(self.layer):
                for node_position, node in enumerate(layer):
                    node_type = node.node_type
                    weights_str = node.weight.tolist() if node_type == 'NEURON' else ''

                    file.write(f"\n{[layer_position,node_position]}\t"
                               f"{node_type}\t"
                               f"{node.activation_function_type if node_type == 'NEURON' else ''}\t"
                               f"{node.bias if node_type == 'NEURON' else ''}\t"
                               f"{weights_str}")

def load_model(file_path):
    import json
    with open(file_path,"r") as file:
        tsv_lines = file.readlines()[1:]

    layer_str = []
    for layer in tsv_lines:
        layer_index = int(json.loads(layer[:layer.index('\t')])[0])
        if len(layer_str) != layer_index+1:
            layer_str.append([])

    for line in tsv_lines:
        line_info = line.split('\t')
        node_position = list(map(int,json.loads(line_info.pop(0))))
        node_type = line_info.pop(0)
        activation_function = line_info.pop(0)
        bias = line_info.pop(0)
        weights = line_info.pop(0)

        if node_type == "INPUT_NODE":
            layer_str[node_position[0]].insert(node_position[1],Input_Node())
        else:
            neuron = Neuron(activation_function, float(bias), list(map(float,json.loads(weights))))
            layer_str[node_position[0]].insert(node_position[1],neuron)

    return Model(layer_str,True)


def make_batch(dataset, batch_size):
    num_batches = len(dataset) // batch_size
    batched_data = []

    for i in range(num_batches):
        batch = dataset[i * batch_size: (i + 1) * batch_size]

        inputs = np.array([item[0] for item in batch])
        labels = np.array([item[1] for item in batch])

        batched_data.append([inputs, labels])

    return batched_data