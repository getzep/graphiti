"""
Module for coreference resolution using FastCoref.
Handles entity extraction and pronoun resolution for Polish and English text.
"""
import logging
import signal
import threading
import re
from typing import List, Dict, Any, Tuple, Optional
from fastcoref import FCoref
import spacy

logger = logging.getLogger(__name__)

def _predict_with_timeout(model, texts, timeout_seconds=30):
    """
    Run FastCoref prediction with timeout protection.
    
    Args:
        model: FastCoref model instance
        texts: List of texts to process
        timeout_seconds: Timeout in seconds
        
    Returns:
        Predictions or None if timeout/error
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = model.predict(
                texts=texts,
                output_file=None,
                is_split_into_words=False
            )
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        logger.warning(f"FastCoref prediction timed out after {timeout_seconds} seconds")
        return None
    
    if exception[0]:
        logger.warning(f"FastCoref prediction failed: {exception[0]}")
        return None
        
    return result[0]

class CoreferenceResolver:
    """
    FastCoref-based coreference resolver that extracts entities and resolves pronouns.
    """
    
    def __init__(self, model_name: str = "biu-nlp/f-coref"):
        """
        Initialize the coreference resolver.
        
        Args:
            model_name: FastCoref model to use
        """
        self.model = None
        self.nlp = None
        self.model_name = model_name
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize FastCoref and spaCy models."""
        try:
            # Initialize FastCoref
            self.model = FCoref(device='cpu')  # Use CPU for stability
            
            # Try to load Polish model first, fallback to English
            try:
                self.nlp = spacy.load("pl_core_news_sm")
                logger.info("Loaded Polish spaCy model")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded English spaCy model")
                except OSError:
                    logger.warning("No spaCy model found. Download with: python -m spacy download en_core_web_sm")
                    self.nlp = None
                    
        except Exception as e:
            logger.error(f"Error initializing coreference resolver: {e}")
            self.model = None
            self.nlp = None
    
    def _extract_entities_with_spacy(self, text: str) -> List[str]:
        """
        Extract entities using spaCy as fallback.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            return []
            
        try:
            doc = self.nlp(text)
            entities = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                    entities.append(ent.text.strip())
            
            # Extract noun phrases that might be entities
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Limit to short phrases
                    entities.append(chunk.text.strip())
                    
            return list(set(entities))
            
        except Exception as e:
            logger.error(f"Error extracting entities with spaCy: {e}")
            return []
    
    def resolve_coreferences_and_extract_entities(
        self, 
        text: str, 
        context_history: List[str] = None,
        existing_entities: List[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve coreferences and extract entities from text.
        
        Args:
            text: Current message text
            context_history: List of previous messages for context
            existing_entities: Known entities from previous conversations
            
        Returns:
            Dictionary containing:
            - resolved_text: Text with pronouns replaced
            - entities: List of extracted entities
            - coreference_clusters: Information about found coreferences
        """
        if not text or not text.strip():            return {
                "resolved_text": text,
                "entities": [],
                "coreference_clusters": []
            }

        # Prepare full context for coreference resolution
        full_context = ""
        if context_history:
            full_context = " ".join(context_history) + " " + text
        else:
            full_context = text
            
        resolved_text = text
        entities = []
        coreference_clusters = []
        
        try:
            if self.model:
                logger.debug(f"FastCoref processing text: '{full_context[:100]}...'")
                
                # Use FastCoref for coreference resolution with timeout protection
                try:
                    predictions = _predict_with_timeout(
                        self.model,
                        [full_context],
                        timeout_seconds=30
                    )
                    logger.debug(f"FastCoref prediction completed, results: {len(predictions) if predictions else 0}")
                except Exception as predict_error:
                    logger.warning(f"FastCoref prediction failed: {predict_error}")
                    predictions = None
                
                if predictions and len(predictions) > 0:
                    # Get the resolved text (with coreferences replaced)
                    result = predictions[0]
                    resolved_full_text = getattr(result, 'text', full_context)
                    
                    # Extract only the current message part from resolved text
                    if context_history:
                        # Approximate extraction of current message from resolved context
                        context_length = len(" ".join(context_history) + " ")
                        resolved_text = resolved_full_text[context_length:] if len(resolved_full_text) > context_length else text
                    else:
                        resolved_text = resolved_full_text
                      # Get coreference clusters information - use correct method
                    try:
                        coreference_clusters = result.get_clusters()
                        logger.debug(f"Found {len(coreference_clusters)} coreference clusters")
                    except Exception as cluster_error:
                        logger.warning(f"Error getting clusters: {cluster_error}")
                        coreference_clusters = []
                    
                    # Apply pronoun replacement using clusters
                    if coreference_clusters:
                        if context_history:
                            # Replace pronouns in the full context, then extract current message part
                            resolved_full_text = self._replace_pronouns_with_entities(resolved_full_text, coreference_clusters)
                            context_length = len(" ".join(context_history) + " ")
                            resolved_text = resolved_full_text[context_length:] if len(resolved_full_text) > context_length else resolved_full_text
                        else:
                            # Replace pronouns directly in the text
                            resolved_text = self._replace_pronouns_with_entities(resolved_text, coreference_clusters)
                    
                    # Extract entities from resolved text
                    entities = self._extract_entities_from_clusters(coreference_clusters, resolved_text)
                else:
                    logger.debug("No FastCoref predictions available, using fallback")
                    
            # Fallback to spaCy if FastCoref fails or unavailable
            if not entities:
                entities = self._extract_entities_with_spacy(resolved_text)
                
            # Filter and merge with existing entities
            entities = self._filter_and_merge_entities(entities, existing_entities)
            
        except Exception as e:
            logger.error(f"Error in coreference resolution: {e}")
            # Fallback to original text and spaCy extraction
            resolved_text = text
            entities = self._extract_entities_with_spacy(text)
            entities = self._filter_and_merge_entities(entities, existing_entities)
        
        return {
            "resolved_text": resolved_text.strip(),
            "entities": entities,
            "coreference_clusters": coreference_clusters
        }
    
    def _extract_entities_from_clusters(self, clusters: List[List], text: str) -> List[str]:
        """
        Extract entities from coreference clusters.
        
        Args:
            clusters: Coreference clusters from FastCoref
            text: Resolved text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            for cluster in clusters:
                # Find the most specific mention in each cluster (usually the first full name)
                best_mention = None
                best_score = 0
                
                for mention in cluster:
                    mention_text = mention.strip() if isinstance(mention, str) else str(mention).strip()
                    
                    # Score mentions (prefer longer, non-pronoun mentions)
                    score = len(mention_text.split())
                    if not self._is_pronoun(mention_text):
                        score += 10
                    if mention_text and mention_text[0].isupper():  # Proper noun
                        score += 5
                        
                    if score > best_score:
                        best_score = score
                        best_mention = mention_text
                
                if best_mention and not self._is_pronoun(best_mention):
                    entities.append(best_mention)
                    
            # Also extract entities using spaCy for additional coverage
            spacy_entities = self._extract_entities_with_spacy(text)
            entities.extend(spacy_entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities from clusters: {e}")
            
        return list(set(entities))
    
    def _is_pronoun(self, text: str) -> bool:
        """
        Check if text is a pronoun.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a pronoun
        """
        if not text:
            return False
            
        pronouns = {
            # English pronouns
            'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their',
            'i', 'you', 'we', 'me', 'us', 'my', 'your', 'our',
            # Polish pronouns
            'on', 'ona', 'ono', 'oni', 'one', 'go', 'ją', 'je', 'ich', 'im', 'nimi',
            'ja', 'ty', 'my', 'wy', 'mnie', 'cię', 'nas', 'was', 'mój', 'twój', 'nasz', 'wasz'
        }
        return text.lower().strip() in pronouns
    
    def _filter_and_merge_entities(self, new_entities: List[str], existing_entities: List[str] = None) -> List[str]:
        """
        Filter and merge entities with existing ones.
        
        Args:
            new_entities: Newly extracted entities
            existing_entities: Previously known entities
            
        Returns:
            Filtered and merged list of entities
        """
        if not new_entities:
            return []
            
        existing_entities = existing_entities or []
        
        # Filter out pronouns, very short words, and common words
        filtered_entities = []
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                       'i', 'jest', 'to', 'w', 'na', 'z', 'do', 'od', 'dla', 'przez', 'o'}
        
        for entity in new_entities:
            entity = entity.strip()
            if (len(entity) > 1 and 
                not self._is_pronoun(entity) and 
                entity.lower() not in common_words and
                not entity.isdigit()):
                filtered_entities.append(entity)
        
        # Merge with existing entities (avoid duplicates based on similarity)
        merged_entities = list(existing_entities)
        
        for new_entity in filtered_entities:
            # Check if entity is similar to existing ones
            is_duplicate = False
            for existing in existing_entities:
                if (new_entity.lower() == existing.lower() or 
                    new_entity.lower() in existing.lower() or 
                    existing.lower() in new_entity.lower()):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged_entities.append(new_entity)
        
        return list(set(merged_entities))
    
    def _replace_pronouns_with_entities(self, text: str, clusters) -> str:
        """
        Replace pronouns with their resolved entities based on coreference clusters.
        
        Args:
            text: Original text
            clusters: Coreference clusters from FastCoref
            
        Returns:
            Text with pronouns replaced
        """
        result_text = text
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            # Find the best entity (usually the first full name/proper noun)
            entity = None
            pronouns = []
            
            for mention in cluster:
                mention_lower = mention.lower()
                if mention_lower in ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']:
                    pronouns.append(mention)
                else:
                    # This looks like a proper entity
                    if entity is None or len(mention) > len(entity):
                        entity = mention
            
            # Replace pronouns with the entity
            if entity and pronouns:
                for pronoun in pronouns:
                    # Case-sensitive replacement with word boundaries
                    import re
                    
                    # Replace with word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(pronoun) + r'\b'
                    result_text = re.sub(pattern, entity, result_text)
        
        return result_text

    def is_available(self) -> bool:
        """
        Check if the coreference resolver is available.
        
        Returns:
            True if models are loaded and ready
        """
        return self.model is not None

# Global instance for reuse
_resolver_instance = None

def get_coreference_resolver() -> CoreferenceResolver:
    """
    Get singleton instance of coreference resolver.
    
    Returns:
        CoreferenceResolver instance
    """
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = CoreferenceResolver()
    return _resolver_instance
