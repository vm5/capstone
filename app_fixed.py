# Fixed version of the problematic functions

def create_peft_model(content):
    """Create a PEFT model adapted to the current content"""
    global base_model, peft_model, current_context_hash, tokenizer
    
    try:
        # Generate a hash for the current content
        new_context_hash = hash(content) % 10000
        logger.info(f"New context hash: {new_context_hash}")
        
        # Extract concepts to adapt the model
        detected_concepts = extract_ml_concepts(content)
        logger.info(f"Detected concepts for PEFT adaptation: {list(detected_concepts.keys())}")
        
        # Configure LoRA with stronger adaptation
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,  # Increased rank for better adaptation
            lora_alpha=64,  # Increased alpha for stronger updates
            lora_dropout=0.1,
            bias="none",
            target_modules=["c_attn", "c_proj", "mlp"],  # Added MLP layers
            modules_to_save=["wte", "lm_head"]  # Save embeddings and LM head
        )
        
        # Create PEFT model
        peft_model = get_peft_model(base_model, lora_config)
        
        # Create detailed adaptation text focusing on detected concepts
        adaptation_text = []
        
        # Add strong topic markers
        adaptation_text.extend([
            "CURRENT TOPIC AND CONTEXT:",
            "=======================",
            content[:500],  # Include start of content for context
            "=======================",
            ""
        ])
        
        # Add detailed concept information
        for topic, data in detected_concepts.items():
            adaptation_text.extend([
                f"TOPIC: {topic.upper()}",
                "------------------------",
                f"Main Concepts: {', '.join(data['keywords'][:10])}",
                "",
                "Key Definitions:",
                *[f"- {k}: {v}" for k, v in data['definitions'].items()],
                "",
                "Relationships:",
                *[f"- {r}" for r in data['relationships']],
                "",
                # Add practical cases if available
                *(["Practical Applications:",
                   *[f"- {case}: {details['scenario']}\n  Implementation: {details['implementation']}" 
                     for case, details in data.get('practical_cases', {}).items()], ""]
                  if 'practical_cases' in data else []),
                # Add numerical examples if available
                *(["Numerical Examples:",
                   *[f"- {case}: {details['scenario']}\n  Solution: {details['calculation']}" 
                     for case, details in data.get('numerical_examples', {}).items()], ""]
                  if 'numerical_examples' in data else []),
                "------------------------",
                ""
            ])
        
        # Add explicit topic-specific prompts
        adaptation_text.extend([
            "GENERATE QUESTIONS ABOUT:",
            *[f"- {topic.upper()}" for topic in detected_concepts.keys()],
            "",
            "FOCUS ON:",
            "- Technical concepts and terminology",
            "- Practical applications and case studies",
            "- Numerical problems and calculations",
            "- Implementation details and challenges",
            ""
        ])
        
        # Encode and process adaptation text
        logger.info("Encoding adaptation text...")
        inputs = tokenizer("\n".join(adaptation_text), return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(peft_model.device) for k, v in inputs.items()}
        
        # Adapt model with multiple forward passes
        logger.info("Adapting model...")
        with torch.no_grad():
            for _ in range(3):  # Multiple passes for stronger adaptation
                outputs = peft_model(**inputs)
        
        current_context_hash = new_context_hash
        logger.info("PEFT model created and adapted successfully")
        return True
        
    except Exception as e:
        logger.error(f"PEFT model creation failed: {str(e)}")
        return False

def generate_questions_with_peft(content, question_type, num_questions):
    """Generate questions using the PEFT-adapted model"""
    try:
        # Extract ML concepts from content
        ml_concepts = extract_ml_concepts(content)
        logger.info(f"Found concepts for question generation: {list(ml_concepts.keys())}")
        
        # Get the topics from the image for prioritization
        priority_topics = [
            'cnn', 'bagging', 'decision_trees', 'ensemble_learning',
            'map', 'svm', 'hmm', 'em', 'unsupervised'
        ]
        
        # Filter and sort concepts by priority
        available_concepts = []
        
        # First add concepts that match the priority topics
        for topic in priority_topics:
            normalized_topic = topic.replace(' ', '_').lower()
            for concept_key in ml_concepts.keys():
                if normalized_topic in concept_key.lower():
                    available_concepts.append((concept_key, ml_concepts[concept_key]))
                    break
        
        # Then add any remaining detected concepts
        for topic, data in ml_concepts.items():
            if not any(topic.lower() in [t.replace(' ', '_').lower() for t, _ in available_concepts]):
                available_concepts.append((topic, data))
        
        if not available_concepts:
            logger.warning("No specific ML concepts found, falling back to traditional generation")
            return generate_questions_without_peft(content, question_type, num_questions)
        
        questions = []
        concepts_used = set()
        
        # Generate questions ensuring topic diversity
        while len(questions) < num_questions and available_concepts:
            # Rotate through available concepts
            for topic, concept_data in available_concepts:
                if len(questions) >= num_questions:
                    break
                    
                if topic in concepts_used and len(available_concepts) > 1:
                    continue
                
                logger.info(f"Generating {question_type} question for topic: {topic}")
                
                # Get template based on question type and topic
                if question_type == 'mcq':
                    question = generate_topic_specific_mcq(topic, concept_data)
                elif question_type == 'descriptive':
                    question = generate_topic_specific_descriptive(topic, concept_data)
                elif question_type == 'true_false':
                    question = generate_topic_specific_true_false(topic, concept_data)
                else:  # fill_blank
                    question = generate_topic_specific_fill_blank(topic, concept_data)
                
                if question:
                    questions.append(question)
                    concepts_used.add(topic)
            
            # If we still need more questions, allow reusing topics
            if len(questions) < num_questions:
                concepts_used.clear()
        
        return questions
        
    except Exception as e:
        logger.error(f"Error in PEFT question generation: {str(e)}")
        return generate_questions_without_peft(content, question_type, num_questions)

def generate_topic_specific_mcq(topic, concept_data):
    try:
        concept = random.choice(concept_data['keywords'])
        definition = concept_data['definitions'].get(concept, f"Main functionality of {concept}")
        
        question = {
            'id': f'mcq-{topic}-{hash(concept) % 1000:03d}',
            'question': f"What is {concept} in {topic.replace('_', ' ')}?",
            'options': [
                definition,
                f"The opposite of {concept}",
                f"A different concept from {topic}",
                f"None of the above"
            ],
            'correct_answer': definition,
            'explanation': f"This is the correct definition of {concept}.",
            'difficulty': 0.7,
            'topic': topic
        }
        return question
    except Exception as e:
        logger.error(f"Error generating MCQ: {str(e)}")
        return None

def generate_topic_specific_descriptive(topic, concept_data):
    try:
        concepts = random.sample(concept_data['keywords'], min(3, len(concept_data['keywords'])))
        question_text = f"Explain the following concepts in {topic.replace('_', ' ')}:\n" + \
                      "\n".join(f"{i+1}. {concept}" for i, concept in enumerate(concepts))
        
        question = {
            'id': f'desc-{topic}-{hash(question_text) % 1000:03d}',
            'question': question_text,
            'answer_outline': [
                "Definition and basic concepts",
                "Technical details and implementation",
                "Practical applications and use cases",
                "Advantages and limitations"
            ],
            'explanation': f"Tests understanding of key concepts in {topic.replace('_', ' ')}",
            'difficulty': 0.8,
            'topic': topic
        }
        return question
    except Exception as e:
        logger.error(f"Error generating descriptive question: {str(e)}")
        return None

def generate_topic_specific_true_false(topic, concept_data):
    try:
        concept = random.choice(concept_data['keywords'])
        relationship = random.choice(concept_data['relationships'])
        statement = f"{concept} is primarily used for {relationship} in {topic.replace('_', ' ')}."
        answer = True
        
        question = {
            'id': f'tf-{topic}-{hash(statement) % 1000:03d}',
            'statement': statement,
            'answer': answer,
            'explanation': f"This statement about {topic.replace('_', ' ')} is {answer}.",
            'difficulty': 0.6,
            'topic': topic
        }
        return question
    except Exception as e:
        logger.error(f"Error generating true/false question: {str(e)}")
        return None 