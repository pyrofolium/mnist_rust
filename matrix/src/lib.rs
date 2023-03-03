extern crate core;

use std::fmt::{Formatter};
use std::iter::zip;
use std::ops::{Add, Sub, Mul, Neg, AddAssign};

//should be used for faster operations with a matrix.
//This exists to allow for matrix multiplication with a vector to happen across
//contiguous data.
pub struct  ColumnVector {
    pub data: Vec<f32>
}

pub struct Matrix {
    data: Vec<Vec<f32>>,
}

impl ColumnVector {
    pub fn from_vec(input: Vec<f32>) -> Self {
        ColumnVector{
            data: input
        }
    }

    pub fn new_with_elements(size: usize, element: f32) -> Self {
        let mut result = Vec::with_capacity(size);
        for _ in 0..size{
            result.push(element);
        }
        ColumnVector::from_vec(result)
    }

    pub fn new_with_number_generator(size:usize, element_gen: &dyn Fn(usize) -> f32) -> Self {
        let mut result = Vec::with_capacity(size);
        for index in 0..size{
            result.push(element_gen(index));
        }
        ColumnVector::from_vec(result)
    }

    pub fn total(&self) -> f32 {
        let mut result = 0.0;
        for elem in &self.data {
            result += elem;
        }
        result
    }

    pub fn average(&self) -> f32 {
        self.total()/(self.data.len() as f32)
    }

    pub fn magnitude_squared(&self) -> f32 {
        let mut acc = 0.0;
        for elem in &self.data {
            acc += elem.powi(2)
        }
        acc
    }

    pub fn _mul_matrix<'a>(&self, matrix: &Matrix, result: &'a mut ColumnVector) -> &'a ColumnVector {
        for (result_elem, matrix_row) in zip(&mut result.data.iter_mut(), &matrix.data) {
            *result_elem = 0.0;
            for (elem_vec, matrix_row_elem) in zip(&self.data, matrix_row) {
                *result_elem += elem_vec * matrix_row_elem;
            }
        }
        result
    }

    pub fn _add<'a>(&self, rhs: &ColumnVector, result: &'a mut ColumnVector) -> &'a ColumnVector {
        for ((lhs_elem, rhs_elem), result_elem) in zip(zip(&self.data, &rhs.data), result.data.iter_mut()) {
            *result_elem = lhs_elem + rhs_elem;
        }
        result
    }

    pub fn _neg<'a>(&self, result: &'a mut ColumnVector) -> &'a ColumnVector {
        for (elem, result_elem) in zip(&self.data, &mut result.data) {
            *result_elem = *elem;
        }
        result
    }

    pub fn _sub<'a>(&self, rhs: &ColumnVector, result: &'a mut ColumnVector) -> &'a ColumnVector {
        for ((lhs_elem, rhs_elem), result_elem) in zip(zip(&self.data, &rhs.data), result.data.iter_mut()) {
            *result_elem = lhs_elem - rhs_elem;
        }
        result
    }

    pub fn _hadamard_product<'a>(self, rhs: &ColumnVector, result: &'a mut ColumnVector) -> &'a ColumnVector {
        for ((elem_lhs, elem_rhs), elem_result) in zip(zip(&self.data, &rhs.data), result.data.iter_mut()){
            *elem_result = elem_lhs * elem_rhs;
        }
        result
    }
}

impl Matrix {
    pub fn from_vec(input: Vec<Vec<f32>>) -> Self {
        Matrix {
            data: input
        }
    }

    pub fn identity(height: usize, width: usize) -> Matrix {
        let mut result = Vec::with_capacity(height);
        if height != height {
            panic!("identity matrix must have equal height and width");
        } else {
            for row_index in 0..height {
                result.push(Vec::with_capacity(width));
                for col_index in 0..width {
                    result.last_mut().unwrap().push(if col_index == row_index {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
            Matrix::from_vec(result)
        }
    }

    pub fn new_with_elements(height: usize, width: usize, element: f32) -> Self {
        let mut result = Vec::with_capacity(height);
        for _ in 0..height {
            result.push(Vec::with_capacity(width));
            for _ in 0..width {
                let row = result.last_mut().unwrap();
                row.push(element);
            }
        }
        Matrix::from_vec(result)
    }

    pub fn new_with_number_generate(height: usize, width: usize, element_gen: &dyn Fn(usize) -> f32) -> Matrix {
        let mut result = Vec::with_capacity(height);
        for index in 0..height {
            result.push(Vec::with_capacity(width));
            for _ in 0..width {
                let row = result.last_mut().unwrap();
                row.push(element_gen(index));
            }
        }
        Matrix::from_vec(result)
    }

    pub fn is_same_shape(&self, other: &Matrix) -> bool {
        !((self.data.len() != other.data.len()) ||
            (self.data.len() > 0 && (self.data[0].len() != other.data[0].len())))
    }

    pub fn is_multipliable(&self, other: &Matrix) -> bool {
        ((other.data.len() > 0) && (self.data.len() == other.data[0].len())) &&
            ((self.data.len() > 0) && (self.data[0].len() == other.data.len())) ||
            (self.data.len() == 0 && other.data.len() == 0)
    }

    pub fn _add<'a>(&self, rhs: &Matrix, result: &'a mut Matrix) -> &'a Matrix {
        if self.is_same_shape(rhs) {
            panic!("For addition both matrices must be the same size")
        } else {
            for ((row1, row2), result_row) in zip(zip(self.data.iter(), rhs.data.iter()), &mut result.data) {
                for ((elem1, elem2), result_elem) in zip(zip(row1, row2), result_row) {
                    *result_elem = elem1 + elem2;
                }
            }
            result
        }
    }

    pub fn _sub<'a>(&self, rhs: &Matrix, result: &'a mut Matrix) -> &'a Matrix {
        if self.is_same_shape(rhs) {
            panic!("For subtraction both matrices must be the same size")
        } else {
            for ((row1, row2), result_row) in zip(zip(self.data.iter(), rhs.data.iter()), &mut result.data) {
                for ((elem1, elem2), result_elem) in zip(zip(row1, row2), result_row) {
                    *result_elem = elem1 - elem2;
                }
            }
            result
        }
    }

    fn _mul_num<'a>(&self, rhs: f32, result: &'a mut Matrix) -> &'a Matrix {
        for (row_original, result_row) in zip(&self.data, result.data.iter_mut()) {
            for (original_elem, result_elem) in zip(row_original, result_row.iter_mut()) {
                *result_elem = original_elem * rhs;
            }
        }
        result
    }

    fn _add_num<'a>(&self, rhs: f32, result: &'a mut Matrix) -> &'a Matrix {
        for (row_original, result_row) in zip(&self.data, result.data.iter_mut()) {
            for (original_elem, result_elem) in zip(row_original, result_row.iter_mut()) {
                *result_elem = original_elem + rhs;
            }
        }
        result
    }

    fn _sub_num<'a>(&self, rhs: f32, result: &'a mut Matrix) -> &'a Matrix {
        for (row_original, result_row) in zip(&self.data, result.data.iter_mut()) {
            for (original_elem, result_elem) in zip(row_original, result_row.iter_mut()) {
                *result_elem = original_elem - rhs;
            }
        }
        result
    }

    fn _neg<'a>(&self, result: &'a mut Matrix) -> &'a Matrix {
        for (row_original, result_row) in zip(&self.data, result.data.iter_mut()) {
            for (&original_elem, result_elem) in zip(row_original, result_row.iter_mut()) {
                *result_elem = -original_elem;
            }
        }
        result
    }


    fn _mul<'a>(&self, rhs: &Matrix, result: &'a mut Matrix) -> &'a Matrix {
        if self.is_multipliable(rhs) {
            for (lhs_row_index, lhs_row) in self.data.iter().enumerate() {
                for rhs_col_index in 0..rhs.data[0].len() {
                    let mut acc = 0.0;
                    for (lhs_row_elem, rhs_row) in zip(lhs_row, &rhs.data) {
                        let rhs_col_elem = rhs_row[rhs_col_index];
                        acc += rhs_col_elem * lhs_row_elem;
                    }
                    result.data[lhs_row_index][rhs_col_index] = acc;
                }
            }
            result
        } else {
            panic!("left hand side matrix must have same amount of rows as right hand side cols in matrix multiplication");
        }
    }
}

impl std::fmt::Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl std::fmt::Debug for ColumnVector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Matrix {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._add(rhs, &mut result);
        result
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._sub(rhs, &mut result);
        result
    }
}

impl Sub<&ColumnVector> for &ColumnVector {
    type Output = ColumnVector;
    fn sub(self, rhs: &ColumnVector) -> Self::Output {
        let mut result = ColumnVector::new_with_elements(rhs.data.len(), 0.0);
        self._sub(rhs, &mut result);
        result
    }
}

impl Neg for &ColumnVector {
    type Output = ColumnVector;
    fn neg(self) -> Self::Output {
        let mut result = ColumnVector::new_with_elements(self.data.len(), 0.0);
        self._neg(&mut result);
        result
    }
}

impl Mul<&ColumnVector> for &Matrix {
    type Output = ColumnVector;

    fn mul(self, rhs: &ColumnVector) -> Self::Output {
        let mut result = ColumnVector::new_with_elements(rhs.data.len(), 0.0);
        rhs._mul_matrix(self, &mut result);
        result
    }
}

impl Mul<&Matrix> for &ColumnVector {
    type Output = ColumnVector;
    fn mul(self, rhs: &Matrix) -> Self::Output{
        let mut result = ColumnVector::new_with_elements(self.data.len(), 0.0);
        self._mul_matrix(rhs, &mut result);
        result
    }
}

impl Add<&ColumnVector> for &ColumnVector {
    type Output = ColumnVector;
    fn add(self, rhs: &ColumnVector) -> Self::Output {
        let mut result = ColumnVector::new_with_elements(self.data.len(), 0.0);
        self._add(rhs, &mut result);
        result
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f32) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._mul_num(rhs, &mut result);
        result
    }
}

impl Mul<&Matrix> for f32 {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new_with_elements(rhs.data.len(), rhs.data[0].len(), 0.0);
        rhs._mul_num(self, &mut result);
        result
    }
}

impl Add<f32> for &Matrix {
    type Output = Matrix;
    fn add(self, rhs: f32) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._add_num(rhs, &mut result);
        result
    }
}

impl Add<&Matrix> for f32 {
    type Output = Matrix;
    fn add(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new_with_elements(rhs.data.len(), rhs.data[0].len(), 0.0);
        rhs._add_num(self, &mut result);
        result
    }
}

impl Sub<f32> for &Matrix {
    type Output = Matrix;
    fn sub(self, rhs: f32) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._sub_num(rhs, &mut result);
        result
    }
}

impl Sub<&Matrix> for f32 {
    type Output = Matrix;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new_with_elements(rhs.data.len(), rhs.data[0].len(), 0.0);
        rhs._sub_num(self, &mut result);
        result
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), rhs.data[0].len(), 0.0);
        self._mul(rhs, &mut result);
        result
    }
}

impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        let mut result = Matrix::new_with_elements(self.data.len(), self.data[0].len(), 0.0);
        self._neg(&mut result);
        result
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.data.len() != other.data.len() || self.data[0].len() != other.data[0].len() {
            false
        } else {
            for (row1, row2) in zip(&self.data, &other.data) {
                for (elem1, elem2) in zip(row1, row2) {
                    if elem1 == elem2 {
                        continue;
                    } else {
                        return false;
                    }
                }
            }
            true
        }
    }
}

impl AddAssign<&ColumnVector> for ColumnVector {
    fn add_assign(&mut self, other: &ColumnVector) {
        for (index, elem) in other.data.iter().enumerate() {
            self.data[index] += elem.clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn equality() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let y = x.clone();
        let mat_x = Matrix::from_vec(x);
        let mut mat_y = Matrix::from_vec(y);
        assert_eq!(mat_y, mat_x);
        mat_y.data[0][0] = 1.0;
        assert_ne!(mat_y, mat_x);
    }

    #[test]
    fn number_arithmetic() {
        let x = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let y = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let mat_x = Matrix::from_vec(x);
        let mat_y = Matrix::from_vec(y);
        let mat_z = &mat_x + 1.0;
        let mat_w = 1.0 as f32 + &mat_x;
        let mat_p = &mat_w - 1.0 as f32;
        let mat_u = &mat_p - 1.0 as f32;
        let mat_a = -&mat_z;
        let mat_0 = 0.0 as f32 * &mat_w;
        let mat_1 = &mat_w * 0.0 as f32;
        assert_eq!(mat_w, mat_y);
        assert_eq!(mat_z, mat_w);
        assert_eq!(mat_p, mat_x);
        assert_eq!(mat_a, mat_u);
        assert_eq!(mat_0, mat_1);
        assert_eq!(mat_1, mat_x);
    }

    #[test]
    fn test_multiplication() {
        let identity = Matrix::identity(4, 4);
        let identity2 = &identity * &identity;
        let mat_x = Matrix::from_vec(
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0],
                vec![11.0, 12.0, 13.0, 14.0],
            ]);
        let mat_prod1 = &mat_x * &identity;
        let mat_prod2 = &identity * &mat_x;
        assert_eq!(identity, identity2);
        assert_eq!(mat_prod1, mat_prod2);
        assert_eq!(mat_prod1, mat_x);
        assert_eq!(mat_prod2, mat_x);
        let row_mat = Matrix::from_vec(vec![vec![1.0, 1.0, 1.0, 1.0]]);
        let col_mat = Matrix::from_vec(vec![vec![1.0], vec![1.0], vec![1.0], vec![1.0]]);
        let scalar = Matrix::from_vec(vec![vec![1.0]]);
        let prod_mat = &(&row_mat * &col_mat) * &scalar;
        assert_eq!(Matrix::from_vec(vec![vec![4.0]]), prod_mat);
    }
}
