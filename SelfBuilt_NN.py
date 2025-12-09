import math
import random
from random import uniform as r_float
import time

random.seed(42)

class Node():

    def __init__(self, node_type):
        self.value = None
        self.node_type = node_type

class Neuron(Node):

    def __init__(self, activation_function_type = "LINEAR", bias = r_float(-1,1), weight = None):
        super().__init__("NEURON")
        self.bias = bias
        self.weight = weight
        self.activation_function_type = activation_function_type

    def set_weight(self, weight):
        self.weight = weight

    def set_bias(self,bias):
        self.bias = bias

    def calculate_output(self, connected_nodes):
        pre_activated_value = self.bias

        for n,w in zip(connected_nodes,self.weight):
            pre_activated_value += n.value*w

        self.value = self.get_activated_value(pre_activated_value)

    def get_activated_value(self,value):
        if self.activation_function_type == "LINEAR":
            return value

        if self.activation_function_type == "RELU":
            return max(0, value)

    def get_derivative(self,value):
        if self.activation_function_type == "LINEAR":
            return 1

        elif self.activation_function_type == "RELU":
            return 1 if value > 0 else 0

    def update(self,new_weight, bias):
        for weight_i in range(len(new_weight)):
            self.weight[weight_i] += new_weight[weight_i]
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
            for layer_index in range(1,len(layer)):
                for neuron in layer[layer_index]:
                    neuron.set_weight([r_float(-1,1) for inputs in range(len(layer[layer_index-1]))])

    def optimiser(self,loss_function,learning_rate):
        self.loss_function = loss_function.upper()
        self.learning_rate = learning_rate


    def fit(self,train_dataset, validation_dataset, epoch, batch = 1, shuffle = True, change_learning_rate = lambda learning_rate: learning_rate*1, print_detail =1):
        if shuffle:
            random.shuffle(train_dataset)

        time_took_per_epoch = []
        lr_per_epoch = []
        mae_per_epoch =[]
        mse_per_epoch = []


        for epoch_i in range(epoch):
            print(f"\n\nEpoch: {epoch_i+1}\n----------------")
            start_time = time.perf_counter()
            lr_per_epoch.append(self.learning_rate)

            for batch_i in range(0,len(train_dataset), batch):
                batch_label = []
                batch_input = []
                batch_predict = []

                for contestant_i in range(batch_i,(batch_i+batch) if batch_i + batch < len(train_dataset) else len(train_dataset)):
                    batch_predict.append(self.feed_foward(train_dataset[contestant_i][0]))
                    batch_label.append(train_dataset[contestant_i][1])
                    batch_input.append([[node.value for node in layer] for layer in self.layer[:-1]])

                self.backpropagation(batch_label,batch_predict,batch_input)

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

        return {"learning_rate": lr_per_epoch, "mae":mae_per_epoch, "mse":mse_per_epoch, "time_took":time_took_per_epoch}

    def get_accuracy(self,dataset):
        predicts = self.feed_foward(dataset[0])
        if len(self.layer[-1]) == 1:
            return [abs(predicts[0] - dataset[1][0]),math.pow(predicts[0] - dataset[1][0],2)]
        else:
            pass



    def backpropagation(self,batch_label,batch_predict,batch_input):
        loss_gradients_per_sample = [self.get_loss_gradient(v,t_v) for v, t_v in zip(batch_predict,batch_label)]
        batch_size = len(batch_label)

        if batch_size == 0:
            return

        # output layer
        previous_layer_index = len(self.layer) - 2

        for output_neuron_index, output_node in enumerate(self.layer[-1]):

            total_weight_adjustments = [0] * len(output_node.weight)
            total_bias_adjustment = 0

            for sample_i in range(batch_size):
                raw_error = loss_gradients_per_sample[sample_i][output_neuron_index]

                neuron_derivative = output_node.get_derivative(batch_predict[sample_i][output_neuron_index])
                current_delta = raw_error * neuron_derivative

                total_bias_adjustment += self.get_bias_adjust(current_delta)
                previous_layer_activations = batch_input[sample_i][previous_layer_index]

                for input_node_index, activation_k in enumerate(previous_layer_activations):
                    weight_adjust = self.get_weight_adjust(current_delta, activation_k)
                    total_weight_adjustments[input_node_index] += weight_adjust


            avg_weight_adjustments = [w / batch_size for w in total_weight_adjustments]
            avg_bias_adjustment = total_bias_adjustment / batch_size

            output_node.update(avg_weight_adjustments, avg_bias_adjustment)


        # hidden layer
        next_layer_loss_gradient = loss_gradients_per_sample.copy()

        for neuron_layer_index in range(len(self.layer[1:-1]),0, -1):
            current_layer = self.layer[neuron_layer_index]
            next_layer = self.layer[neuron_layer_index + 1]
            previous_layer_index = neuron_layer_index - 1

            loss_gradient_replacement = [[] for batch in range(batch_size)]

            for neuron_index, neuron in enumerate(current_layer):
                total_weight_adjustments = [0] * len(neuron.weight)
                total_bias_adjustment = 0

                for sample_i in range(batch_size):
                    current_deltas = next_layer_loss_gradient[sample_i]
                    sum_delta = 0

                    for next_neuron_index, next_neuron in enumerate(next_layer):
                        next_nueron_delta = current_deltas[next_neuron_index]
                        sum_delta += next_neuron.weight[neuron_index] * next_nueron_delta

                    current_neuron_delta = sum_delta * neuron.get_derivative(batch_input[sample_i][neuron_layer_index][neuron_index])
                    total_bias_adjustment += self.get_bias_adjust(current_neuron_delta)
                    loss_gradient_replacement[sample_i].append(current_neuron_delta)

                    for weight_i,weight in enumerate(neuron.weight):
                        weight_adjust = self.get_weight_adjust(current_neuron_delta,batch_input[sample_i][previous_layer_index][weight_i])
                        total_weight_adjustments[weight_i] += weight_adjust


                avg_weight_adjustments = [w / batch_size for w in total_weight_adjustments]
                avg_bias_adjustment = total_bias_adjustment / batch_size

                neuron.update(avg_weight_adjustments, avg_bias_adjustment)


            next_layer_loss_gradient = loss_gradient_replacement



    def feed_foward(self,features):
        # input layer
        for node_index in range(len(self.layer[0])):
            node = self.layer[0][node_index]
            node.value = features[node_index]

        # hidden layer and output layer
        for layer_index in range(1,len(self.layer)):
            for neuron in self.layer[layer_index]:
                neuron.calculate_output(self.layer[layer_index-1])

        # get predict value
        return [output_node.value for output_node in self.layer[-1]]

    def get_loss_gradient(self, values, target_values):
        if self.loss_function == "MEAN_SQUARED_ERROR":
            return [v-t_v for v,t_v in zip(values,target_values)]

        if self.loss_function == "MEAN_ABSOLUTE_ERROR":
            return

    def get_weight_adjust(self, loss_gradient, _input):
        return -self.learning_rate * loss_gradient * _input

    def get_bias_adjust(self, loss_gradient):
        return -self.learning_rate * loss_gradient

    def predict(self,feature):
        return self.feed_foward(feature)

    def save_model(self, file_path):
        with open(file_path,"a") as file:
            file.write("position\ttype\tactivation_function\tbias\tweights")

            for layer_position, layer in enumerate(self.layer):
                for node_position, node in enumerate(layer):
                    node_type = node.node_type
                    file.write(f"\n{[layer_position,node_position]}\t"
                               f"{node_type}\t"
                               f"{node.activation_function_type if node_type == 'NEURON' else ''}\t"
                               f"{node.bias if node_type == 'NEURON' else ''}\t"
                               f"{node.weight if node_type == 'NEURON' else ''}")

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