use std::collections::HashMap;

fn calculate_bm25_scores(
    corpus: &[Vec<String>], 
    query: &[String], 
    k1: f64, 
    b: f64
) -> Result<Vec<f64>, &'static str> {
    if corpus.is_empty() || query.is_empty() {
        return Err("Corpus and query cannot be empty");
    }
    
    // Calculate document lengths
    let doc_lengths: Vec<usize> = corpus.iter().map(|doc| doc.len()).collect();
    let avg_doc_length: f64 = doc_lengths.iter().sum::<usize>() as f64 / doc_lengths.len() as f64;
    
    // Count term frequencies in each document
    let doc_term_counts: Vec<HashMap<String, usize>> = corpus
        .iter()
        .map(|doc| {
            let mut counts = HashMap::new();
            for term in doc {
                *counts.entry(term.clone()).or_insert(0) += 1;
            }
            counts
        })
        .collect();
    
    // Calculate document frequencies (how many documents contain each term)
    let mut doc_freqs: HashMap<String, usize> = HashMap::new();
    for doc in corpus {
        // Get unique terms in this document
        let unique_terms: std::collections::HashSet<String> = doc.iter().cloned().collect();
        for term in unique_terms {
            *doc_freqs.entry(term).or_insert(0) += 1;
        }
    }
    
    // Initialize scores array
    let mut scores = vec![0.0; corpus.len()];
    let n = corpus.len() as f64;
    
    // Calculate BM25 score for each query term
    for term in query {
        // Document frequency with Laplace smoothing
        let df = doc_freqs.get(term).unwrap_or(&0) + 1;
        
        // Calculate IDF (Inverse Document Frequency)
        let idf = ((n + 1.0) / df as f64).ln();
        
        // Calculate BM25 component for this term across all documents
        for (idx, term_counts) in doc_term_counts.iter().enumerate() {
            // Term frequency in current document
            let tf = term_counts.get(term).unwrap_or(&0);
            
            if *tf > 0 {  // Only process if term appears in document
                // Document length normalization
                let doc_len = doc_lengths[idx] as f64;
                let norm_factor = 1.0 - b + b * (doc_len / avg_doc_length);
                
                // BM25 formula
                let tf_f64 = *tf as f64;
                let term_score = idf * (tf_f64 * (k1 + 1.0)) / (tf_f64 + k1 * norm_factor);
                scores[idx] += term_score;
            }
        }
    }
    
    Ok(scores)
}

// Helper function to preprocess text (simple tokenization)
fn preprocess_text(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

// Search function that returns top results
fn search_documents(
    documents: &[String], 
    query: &str, 
    k1: f64, 
    b: f64, 
    top_k: usize
) -> Result<Vec<(usize, f64, String)>, &'static str> {
    // Preprocess documents and query
    let corpus: Vec<Vec<String>> = documents
        .iter()
        .map(|doc| preprocess_text(doc))
        .collect();
    
    let query_terms = preprocess_text(query);
    
    // Calculate BM25 scores
    let scores = calculate_bm25_scores(&corpus, &query_terms, k1, b)?;
    
    // Create results with document indices and scores
    let mut results: Vec<(usize, f64, String)> = scores
        .iter()
        .enumerate()
        .map(|(idx, &score)| (idx, score, documents[idx].clone()))
        .collect();
    
    // Sort by score (descending) and return top k
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(top_k);
    
    Ok(results)
}

fn main() {
    // Sample documents
    let documents = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "Python is a powerful programming language for data science".to_string(),
        "Machine learning algorithms can process large datasets efficiently".to_string(),
        "The fox is quick and brown, jumping over dogs lazily".to_string(),
        "Data science requires knowledge of statistics and programming".to_string(),
    ];
    
    let query = "quick brown fox";
    
    // Search with default BM25 parameters
    match search_documents(&documents, query, 1.5, 0.75, 3) {
        Ok(results) => {
            println!("Search results for query: '{}'", query);
            println!("----------------------------------------");
            for (rank, (doc_idx, score, doc_text)) in results.iter().enumerate() {
                println!("{}. Score: {:.4} | Doc {}: {}", 
                    rank + 1, score, doc_idx, doc_text);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
    
    // Test with direct BM25 calculation
    println!("\n--- Direct BM25 calculation test ---");
    let corpus = vec![
        vec!["the".to_string(), "quick".to_string(), "brown".to_string(), "fox".to_string()],
        vec!["the".to_string(), "lazy".to_string(), "dog".to_string()],
        vec!["quick".to_string(), "fox".to_string(), "jumps".to_string()],
    ];
    
    let query = vec!["quick".to_string(), "fox".to_string()];
    
    match calculate_bm25_scores(&corpus, &query, 1.5, 0.75) {
        Ok(scores) => {
            println!("BM25 scores: {:?}", scores);
            for (i, score) in scores.iter().enumerate() {
                println!("Document {}: {:.4}", i, score);
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let empty_corpus: Vec<Vec<String>> = vec![];
        let query = vec!["test".to_string()];
        assert!(calculate_bm25_scores(&empty_corpus, &query, 1.5, 0.75).is_err());
        
        let corpus = vec![vec!["test".to_string()]];
        let empty_query: Vec<String> = vec![];
        assert!(calculate_bm25_scores(&corpus, &empty_query, 1.5, 0.75).is_err());
    }
    
    #[test]
    fn test_basic_scoring() {
        let corpus = vec![
            vec!["cat".to_string(), "dog".to_string()],
            vec!["dog".to_string(), "bird".to_string()],
        ];
        let query = vec!["dog".to_string()];
        
        let result = calculate_bm25_scores(&corpus, &query, 1.5, 0.75);
        assert!(result.is_ok());
        
        let scores = result.unwrap();
        assert_eq!(scores.len(), 2);
        // Both documents contain "dog", so both should have positive scores
        assert!(scores[0] > 0.0);
        assert!(scores[1] > 0.0);
    }
    
    #[test]
    fn test_preprocess_text() {
        let text = "Hello World! This is a TEST.";
        let tokens = preprocess_text(text);
        assert_eq!(tokens, vec!["hello", "world!", "this", "is", "a", "test."]);
    }
}
