# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load('en_core_web_sm')


def get_ner_and_verbs(text):
    doc = nlp(text)

    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    entities = []
    # Find named entities, phrases and concepts
    for entity in doc.ents:
        entities.append({"entity": entity.text, "type": entity.label_})

    return {
        "entities": entities,
        "verbs": verbs
    }
