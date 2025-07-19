fn matrix_dot_vector(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, &'static str> {
  if a.is_empty() || a[0].len() != b.len() {
    return Err("Dimension mismatch")
  }
  
  let mut result = Vec::new();
  for row in a {
    let mut total = 0.0;
    for (i, &val) in row.iter().enumerate() {
      total += val * b[i];
    }
    result.push(total);
  }

  Ok(result)
}

// Example usage and test
fn main() {
  let matrix = vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
  ];
  let vector = vec![1.0, 2.0, 3.0];
  
  match matrix_dot_vector(&matrix, &vector) {
      Ok(result) => println!("Result: {:?}", result),
      Err(e) => println!("Error: {}", e),
  }
  
  // Test with mismatched dimensions
  let bad_vector = vec![1.0, 2.0];
  match matrix_dot_vector(&matrix, &bad_vector) {
      Ok(result) => println!("Result: {:?}", result),
      Err(e) => println!("Error: {}", e),
  }
}
