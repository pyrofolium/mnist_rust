use std::intrinsics::exp2f32;

fn sigmoid(x: f32) -> f32{
    1.0/(1.0 + (std::f32::consts::E as f32).powf(-x))
}


