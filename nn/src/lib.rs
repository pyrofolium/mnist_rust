use std::collections::VecDeque;
use std::iter::zip;
use std::ops::Deref;
use rand::Rng;
use matrix::{ColumnVector, Matrix};
use std::fmt;
use std::fmt::Debug;


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

fn squared_error(output_vector: &ColumnVector, desired_output: &ColumnVector) -> f32{
    (output_vector - desired_output).magnitude_squared()
}

struct NeuralNetwork {
    pub weights: Vec<Matrix>,
    pub activation_values: VecDeque<ColumnVector>,
    pub biases: Vec<ColumnVector>,
}


impl NeuralNetwork {
    fn _forward_pass_one_step<'a>(&mut self, layer_index: usize) {
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
        zip(inputs, expected_outputs).map(|(input, expected)|{
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
    use crate::NeuralNetwork;
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
            assert_eq!(value, if index != test_nn.activation_values.len() - 1 {&input_vector2} else {&zero_input});
        }

    }

}
