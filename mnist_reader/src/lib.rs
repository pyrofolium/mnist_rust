use std::{env, fmt};
use std::fs::File;
use std::path::PathBuf;
use csv::{DeserializeRecordsIter, Reader};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct TrainingDataElement {
    actual_result: u8,
    image: Vec<u8>,
}

impl fmt::Display for TrainingDataElement {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "The image below is a {}\n", self.actual_result).unwrap();
        for (index, elem) in self.image.iter().enumerate() {
            write!(f, "{:>3} ", elem).unwrap();
            write!(f, "{}", if (index + 1) % 28 == 0  { "\n" } else { "" }).unwrap();
        }
        write!(f, "\n")
    }
}

type TrainingData = Vec<TrainingDataElement>;

fn read_csv_data(file_path: &str) -> Reader<File> {
    let mut reader_builder = csv::ReaderBuilder::new();
    reader_builder.has_headers(false).delimiter(',' as u8);
    let reader = reader_builder.from_path(std::path::Path::new(file_path)).unwrap();
    reader
}

fn deserialize_mnist_data(reader: &mut Reader<File>) -> DeserializeRecordsIter<File, TrainingDataElement> {
    reader.deserialize()
}

fn get_current_working_dir() -> std::io::Result<PathBuf> {
    env::current_dir()
}


#[cfg(test)]
mod tests {
    use crate::{get_current_working_dir, deserialize_mnist_data, read_csv_data};

    #[test]
    fn csv_reading() {
        println!("{}", get_current_working_dir().unwrap().as_path().display());
        let mut reader = read_csv_data("../data/mnist_train.csv");
        for elem in deserialize_mnist_data(&mut reader) {
            println!("{}", elem.unwrap());
        }
    }
}