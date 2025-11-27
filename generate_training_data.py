from transformers import pipeline
import json

def generate_training_examples(topics):
    """Generate diverse training examples"""
    examples = []
    for topic in topics:
        # Generate detailed context
        context = f"""Technical deep dive into {topic}: [Generated technical content]"""
        
        # Generate challenging MCQ
        mcq = f"""Question: [Technical question about {topic}]
A) [Detailed solution]
B) [Alternative approach]
C) [Common misconception]
D) [Related but incorrect]
Correct Answer: A"""
        
        # Generate analytical questions
        analytical = """1. [Technical analysis question]
2. [Implementation question]"""
        
        examples.append({
            "input_text": context,
            "output_text": f"{mcq}\n\nAnalytical Questions:\n{analytical}"
        })
    
    return examples

# Example usage
topics = [
    "Neural Architecture Search",
    "Attention Mechanisms",
    "Graph Neural Networks",
    "Reinforcement Learning",
    "AutoML Systems",
    # Add more topics...
]

training_data = generate_training_examples(topics)
with open("data.json", "w") as f:
    json.dump(training_data, f, indent=4) 