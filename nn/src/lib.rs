use std::collections::VecDeque;
use std::iter::{zip};
// use std::ops::Deref;
use rand::Rng;
use matrix::{ColumnVector, Matrix};
use std::{fmt};
use std::fmt::Debug;
use std::io::{BufReader, Read, Write};
use crate::SerializerIteratorNNState::{Biases, LayerAmount, LayerSizes, Weights};
use itertools::{Itertools};


fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + std::f32::consts::E.powf(-z))
}

fn relu(z: f32) -> f32 {
    if z < 0.0 {
        0.0
    } else {
        z
    }
}

fn softmax(z: &ColumnVector, index: usize) -> f32 {
    z.data[index] / z.average()
}

fn squared_error(output_vector: &ColumnVector, desired_output: &ColumnVector) -> f32 {
    (output_vector - desired_output).magnitude_squared()
}

#[derive(PartialEq)]
struct NeuralNetwork {
    pub weights: Vec<Matrix>,
    pub activation_values: VecDeque<ColumnVector>,
    pub biases: Vec<ColumnVector>,
}

impl NeuralNetwork {
    fn _forward_pass_one_step<'a>(&mut self, layer_index: usize) {
        //The moves may still trigger memory allocation.
        let input = self.activation_values.pop_front().unwrap();
        let mut result = self.activation_values.pop_front().unwrap();
        let weights = &self.weights[layer_index];
        let bias = &mut self.biases[layer_index];
        input._mul_matrix(weights, &mut result);
        result += bias;
        self.activation_values.push_back(input);
        self.activation_values.push_front(result);
    }

    pub fn calculate_all_activation_values(&mut self, input: &ColumnVector) {
        for (index, elem) in input.data.iter().enumerate() {
            self.activation_values[0].data[index] = *elem;
        }

        for index in 0..self.weights.len() {
            NeuralNetwork::_forward_pass_one_step(self, index);
        }
        let output = self.activation_values.pop_front().unwrap();
        self.activation_values.push_back(output);
    }

    pub fn calculate_mean_square_error(mut self, inputs: &Vec<ColumnVector>, expected_outputs: &Vec<ColumnVector>) -> f32 {
        zip(inputs, expected_outputs).map(|(input, expected)| {
            self.calculate_all_activation_values(input);
            (self.activation_values.back().unwrap() - expected).magnitude_squared()
        }).reduce(|acc, x| acc + x).unwrap() / (2.0 * (expected_outputs.len() as f32))
    }

    pub fn new(&self, layer_sizes: &[usize], default_value: Option<f32>) -> NeuralNetwork {
        if layer_sizes.len() < 2 {
            panic!("Cannot generate neural network with less than 2 layers.");
        } else {
            let mut weights = Vec::with_capacity(layer_sizes.len() - 1);
            let mut biases = Vec::with_capacity(layer_sizes.len() - 1);
            let mut activation_values = Vec::with_capacity(layer_sizes.len());
            activation_values.push(ColumnVector::new_with_elements(layer_sizes[0], 0.0));
            for (index, &size) in layer_sizes[0..layer_sizes.len() - 1].iter().enumerate() {
                match default_value {
                    Some(value) => {
                        weights.push(Matrix::new_with_elements(layer_sizes[index + 1], size, value));
                        biases.push(ColumnVector::new_with_elements(layer_sizes[index + 1], value));
                    }
                    None => {
                        let element_gen = |_| {
                            let mut rng = rand::thread_rng();
                            rng.gen()
                        };
                        weights.push(Matrix::new_with_number_generate(layer_sizes[index + 1], size, &element_gen));
                        biases.push(ColumnVector::new_with_number_generator(layer_sizes[index + 1], &element_gen));
                    }
                };
                activation_values.push(ColumnVector::new_with_elements(layer_sizes[index + 1], 0.0));
            }
            NeuralNetwork::new_from_vecs(weights, Some(biases), Some(activation_values))
        }
    }

    pub fn new_from_vecs(weights: Vec<Matrix>, biases: Option<Vec<ColumnVector>>, activation_values: Option<Vec<ColumnVector>>) -> NeuralNetwork {
        let amount_of_weight_matrices = weights.len();
        NeuralNetwork {
            biases: match biases {
                Some(value) => value,
                None => {
                    let mut acc: Vec<ColumnVector> = Vec::with_capacity(amount_of_weight_matrices);
                    for matrix in &weights {
                        acc.push(ColumnVector::new_with_elements(matrix.data[0].len(), 0.0));
                    }
                    acc
                }
            },
            activation_values: match activation_values {
                Some(values) => VecDeque::from(values),
                None => {
                    let mut acc: VecDeque<ColumnVector> = VecDeque::with_capacity(weights.len());
                    for matrix in &weights {
                        acc.push_back(ColumnVector::new_with_elements(matrix.data[0].len(), 0.0));
                    }
                    acc
                }
            },
            weights,
        }
    }
    fn serialize_iter(&self) -> SerializerIteratorNN {
        SerializerIteratorNN {
            neural_network: self,
            state: Some(LayerAmount),
        }
    }

    pub fn serialize_to_file(self, file_path: &str) -> () {
        let buffer: Vec<u8> = self.serialize_iter().flat_map(|x| {
            match x {
                NNSerializationValues::Value(v) => v.to_be_bytes().to_vec().into_iter(),
                NNSerializationValues::Size(v) => v.to_be_bytes().to_vec().into_iter()
            }
        }).collect();
        std::fs::File::open(file_path).unwrap().write_all(&buffer).unwrap();
    }

    fn _deserialize_from_file(self, file_path: &str) -> Box<dyn Iterator<Item=NNSerializationValues>> {
        let file = std::fs::File::open(file_path).unwrap();
        let reader = BufReader::new(file);
        // layer_amount_bytes: Vec<u8> = vec::Vec::with_capacity(2);
        let mut byte_iterator = reader.bytes();
        let amount_of_layers = u16::from_be_bytes([byte_iterator.next().unwrap().unwrap(), byte_iterator.next().unwrap().unwrap()]);
        let layer_sizes: Vec<NNSerializationValues> = byte_iterator
            .by_ref()
            .map(|x| x.unwrap())
            .tuples()
            .step_by(2)
            .map(|(w, x)| u16::from_be_bytes([w, x]))
            .take(amount_of_layers as usize)
            .map(|x| NNSerializationValues::Size(x)).collect();
        // .copied();
        let data = byte_iterator
            //     .skip(2 + amount_of_layers as usize * 2)
            .map(|x| x.unwrap())
            .step_by(4)
            .tuple_windows()
            .map(|(w, x, y, z)| f32::from_be_bytes([w, x, y, z]))
            .map(|x| NNSerializationValues::Value(x));
        Box::new([NNSerializationValues::Size(amount_of_layers)].into_iter().chain(layer_sizes.into_iter()).chain(data))


    }

    fn deserialize_from_file(self, file_path: &str) {
        let mut data_iterator = self._deserialize_from_file(file_path);
        let layer_amount = match data_iterator.next().unwrap() {
            NNSerializationValues::Size(v) => v,
            _ => panic!("wrong type")
        };
        let (mut weights, mut biases): (Vec<Matrix>, Vec<ColumnVector>) = data_iterator
            .by_ref()
            .take(layer_amount as usize)
            .tuples()
            .map(|(h, w)| {
                match (h, w) {
                    (NNSerializationValues::Size(h), NNSerializationValues::Size(w)) => {
                        (Matrix::new_with_elements(h as usize, w as usize, 0.0),
                         ColumnVector::new_with_elements(w as usize, 0.0))
                    }
                    _ => panic!("Wrong type")
                }
        }).unzip();

        let mut values = data_iterator.map(|x| {
            match x {
                NNSerializationValues::Value(v) => v,
                _ => panic!("wrong type!")
            }
        });

        let weight_iter = weights.iter_mut().flat_map(|x|{
            x.data.iter_mut()
        }).flat_map(|x| x.iter_mut());
        let bias_iter = biases.iter_mut().flat_map(|x| x.data.iter_mut());

        zip(weight_iter.chain(bias_iter), values).for_each(|(elem, v)|{
            *elem = v
        })
    }
}

//layer amount, layer sizes, weight, biases.
#[derive(Clone)]
enum SerializerIteratorNNState {
    LayerAmount,
    LayerSizes(u16),
    // layer size index (fetch it from the actual weight matrix)
    Weights(u16, u16, u16),
    //weight matrix index, row, height
    Biases(u16, u16), //bias column vector index, elem index
}

#[derive(Clone)]
struct SerializerIteratorNN<'a> {
    neural_network: &'a NeuralNetwork,
    state: Option<SerializerIteratorNNState>,
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum NNSerializationValues {
    Value(f32),
    Size(u16),
}

impl<'a> Iterator for SerializerIteratorNN<'a> {
    type Item = NNSerializationValues;

    fn next(&mut self) -> Option<Self::Item> {
        let current_state = self.state.clone();
        match current_state
        {
            Some(state) => {
                match state {
                    LayerAmount => {
                        self.state = Some(LayerSizes(0));
                        Some(NNSerializationValues::Size((self.neural_network.biases.len() + 1) as u16))
                    }
                    LayerSizes(layer_size_index) if layer_size_index == 0 => {
                        self.state = Some(LayerSizes(1));
                        Some(NNSerializationValues::Size(self.neural_network.weights[0].data[0].len() as u16))
                    }
                    LayerSizes(layer_size_index)
                    // if layer_size_index < self.neural_network.weights.len() && layer_size_index > 0
                    => {
                        if layer_size_index < self.neural_network.weights.len() as u16 {
                            self.state = Some(LayerSizes(layer_size_index + 1));
                        } else {
                            self.state = Some(Weights(0, 0, 0));
                        }
                        Some(NNSerializationValues::Size(self.neural_network.biases[(layer_size_index - 1) as usize].data.len() as u16))
                    }
                    Weights(index, row, col) => {
                        let matrix = &self.neural_network.weights[index as usize].data;
                        let width = matrix[0].len();
                        let height = matrix.len();
                        if col < (width - 1) as u16 {
                            self.state = Some(Weights(index, row, col + 1));
                        } else if row < (height - 1) as u16 {
                            self.state = Some(Weights(index, row + 1, 0));
                        } else if index < (self.neural_network.weights.len() - 1) as u16 {
                            self.state = Some(Weights(index + 1, 0, 0));
                        } else {
                            self.state = Some(Biases(0, 0));
                        }
                        Some(NNSerializationValues::Value(matrix[row as usize][col as usize]))
                    }
                    Biases(index, elem_index) => {
                        let vector = &self.neural_network.biases[index as usize].data;
                        let total = vector.len();
                        if elem_index < (total - 1) as u16 {
                            self.state = Some(Biases(index, elem_index + 1));
                        } else if index < (vector.len() - 1) as u16 {
                            self.state = Some(Biases(index + 1, 0));
                        } else {
                            self.state = None;
                        }
                        Some(NNSerializationValues::Value(vector[elem_index as usize]))
                    }
                }
            }
            None => None
        }
    }
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // self.weights.fmt(f).unwrap();
        self.activation_values.fmt(f)
        // self.biases.fmt(f)
    }
}


#[cfg(test)]
mod tests {
    use matrix::ColumnVector;
    use crate::{NeuralNetwork, NNSerializationValues};
    use super::Matrix;

    #[test]
    fn check_feed_forward() {
        let amount_weight_matrices = 20;
        let matrix_size = 5;
        let mut weights: Vec<Matrix> = Vec::with_capacity(amount_weight_matrices);
        for _ in 0..amount_weight_matrices {
            weights.push(Matrix::identity(matrix_size));
        }
        weights.push(Matrix::zeros(matrix_size, matrix_size));

        let mut test_nn = NeuralNetwork::new_from_vecs(weights, None, None);
        let input_vector = ColumnVector::new_with_elements(matrix_size, 1.0);
        let input_vector2 = input_vector.clone();
        let zero_input = ColumnVector::new_with_elements(matrix_size, 0.0);
        test_nn.calculate_all_activation_values(&input_vector);
        println!("{}", test_nn);
        for (index, value) in test_nn.activation_values.iter().enumerate() {
            assert_eq!(value, if index != test_nn.activation_values.len() - 1 { &input_vector2 } else { &zero_input });
        }
    }

    #[test]
    fn check_serialization_and_deserialization() {
        let m1 = Matrix::identity(2);
        let m2 = Matrix::from_vec(m1.data.clone());
        let b1 = ColumnVector::from_vec(vec![1.0, 1.0]);
        let b2 = ColumnVector::from_vec(b1.data.clone());
        let nn = NeuralNetwork::new_from_vecs(vec![m1, m2], Some(vec![b1, b2]), None);
        let test_match = vec![NNSerializationValues::Size(3), //layers
                              NNSerializationValues::Size(2), //size of first layer
                              NNSerializationValues::Size(2), //size of second layer
                              NNSerializationValues::Size(2), //size of third layer
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(0.0),
                              NNSerializationValues::Value(0.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(0.0),
                              NNSerializationValues::Value(0.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(1.0),
                              NNSerializationValues::Value(1.0)];

        for (index, elem) in nn.serialize_iter().enumerate() {
            assert_eq!(test_match[index], elem);
            let v = match elem {
                NNSerializationValues::Value(v) => v,
                NNSerializationValues::Size(v) => v as f32
            };
            println!("{}", v);
        }
    }
}
