use gedik::board::Position;
use gedik::nn::*;

#[test]
fn test_spatial_planes_dimensions() {
    let pos = Position::start();
    let planes = spatial_planes(&pos);
    assert_eq!(planes.len(), 13 * 81);
}

#[test]
fn test_se_block_forward() {
    let se = SeBlock {
        fc1: Linear { in_f: 32, out_f: 8, weights: vec![0.01; 32 * 8], bias: vec![0.0; 8] },
        fc2: Linear { in_f: 8, out_f: 32, weights: vec![0.01; 8 * 32], bias: vec![0.0; 32] },
    };
    let input = vec![1.0f32; 32 * 81];
    let mut output = vec![0.0f32; 32 * 81];
    se.forward(&input, &mut output, 32);
    assert!(output[0] > 0.0, "SE blok çıktısı pozitif olmalı");
}

