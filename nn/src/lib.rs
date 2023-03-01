use std::iter::zip;
use matrix::ColumnVector;

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

fn mean_square_error(output_vectors: Vec<ColumnVector>, desired_output_vectors: Vec<ColumnVector>) -> f32 {
    let mut acc: f32 = 0.0;
    for (output, desired) in zip(&output_vectors, &desired_output_vectors) {
        acc += (output - desired).magnitude_squared();
    }
    acc / (2.0 * output_vectors.len() as f32)
}