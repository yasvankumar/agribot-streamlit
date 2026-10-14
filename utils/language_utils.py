import re
import string

class LanguageUtils:
    def __init__(self):
        # Tamil Unicode range
        self.tamil_range = range(0x0B80, 0x0BFF)
        
        # Common Tamil words for detection
        self.tamil_words = [
            'வணக்கம்', 'நமஸ்கார்', 'எப்படி', 'எங்கே', 'என்ன', 'யார்', 'எப்போது',
            'பயிர்', 'விவசாயம்', 'நோய்', 'மருந்து', 'உரம்', 'தண்ணீர்', 'மழை',
            'வானிலை', 'சந்தை', 'விலை', 'விற்பனை', 'கள்', 'இலை', 'பூ', 'காய்'
        ]
        
        # Common English agricultural terms
        self.english_agri_words = [
            'crop', 'farming', 'disease', 'pest', 'fertilizer', 'irrigation',
            'weather', 'market', 'price', 'yield', 'harvest', 'soil', 'seed'
        ]
        
        # Tanglish patterns (English words with Tamil script mixing)
        self.tanglish_patterns = [
            r'[a-zA-Z]+[\u0B80-\u0BFF]+',  # English + Tamil
            r'[\u0B80-\u0BFF]+[a-zA-Z]+',  # Tamil + English
        ]
    
    def detect_language(self, text):
        """Detect language of the input text"""
        if not text:
            return 'en'
        
        # Remove punctuation and convert to lowercase
        clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        
        # Count Tamil characters
        tamil_chars = sum(1 for char in text if ord(char) in self.tamil_range)
        total_chars = len(text.replace(' ', ''))
        
        # Check for Tanglish patterns
        tanglish_match = any(re.search(pattern, text) for pattern in self.tanglish_patterns)
        
        # Check for mixed language indicators
        has_english_words = any(word in clean_text for word in self.english_agri_words)
        has_tamil_words = any(word in text for word in self.tamil_words)
        
        # Language detection logic
        if tanglish_match or (has_english_words and has_tamil_words):
            return 'tanglish'
        elif tamil_chars > 0 and (tamil_chars / max(total_chars, 1)) > 0.3:
            return 'ta'
        elif has_tamil_words:
            return 'ta'
        else:
            return 'en'
    
    def translate_key_terms(self, text, target_language):
        """Translate key agricultural terms"""
        # Agricultural term translations
        translations = {
            'en_to_ta': {
                'crop': 'பயிர்',
                'disease': 'நோய்',
                'pest': 'பூச்சி',
                'fertilizer': 'உரம்',
                'irrigation': 'நீர்ப்பாசனம்',
                'weather': 'வானிலை',
                'market': 'சந்தை',
                'price': 'விலை',
                'yield': 'விளைச்சல்',
                'harvest': 'அறுவடை',
                'soil': 'மண்',
                'seed': 'விதை',
                'water': 'தண்ணீர்',
                'rain': 'மழை',
                'temperature': 'வெப்பநிலை',
                'humidity': 'ஈரப்பதம்',
                'treatment': 'சிகிச்சை',
                'prevention': 'தடுப்பு'
            },
            'ta_to_en': {
                'பயிர்': 'crop',
                'நோய்': 'disease',
                'பூச்சி': 'pest',
                'உரம்': 'fertilizer',
                'நீர்ப்பாசனம்': 'irrigation',
                'வானிலை': 'weather',
                'சந்தை': 'market',
                'விலை': 'price',
                'விளைச்சல்': 'yield',
                'அறுவடை': 'harvest',
                'மண்': 'soil',
                'விதை': 'seed',
                'தண்ணீர்': 'water',
                'மழை': 'rain',
                'வெப்பநிலை': 'temperature',
                'ஈரப்பதம்': 'humidity',
                'சிகிச்சை': 'treatment',
                'தடுப்பு': 'prevention'
            }
        }
        
        if target_language == 'ta':
            trans_dict = translations['en_to_ta']
        elif target_language == 'en':
            trans_dict = translations['ta_to_en']
        else:
            return text
        
        # Perform translation
        translated_text = text
        for source, target in trans_dict.items():
            # Case-insensitive replacement for English words
            if target_language == 'ta':
                translated_text = re.sub(r'\b' + re.escape(source) + r'\b', 
                                       target, translated_text, flags=re.IGNORECASE)
            else:
                translated_text = translated_text.replace(source, target)
        
        return translated_text
    
    def format_bilingual_response(self, english_text, tamil_text=None):
        """Format response with both languages"""
        if tamil_text:
            return f"{english_text}\n\n**தமிழ்:** {tamil_text}"
        else:
            # Auto-translate key terms
            translated = self.translate_key_terms(english_text, 'ta')
            if translated != english_text:
                return f"{english_text}\n\n**Key terms in Tamil:** {translated}"
        
        return english_text
    
    def extract_keywords(self, text, language=None):
        """Extract relevant agricultural keywords from text"""
        if language is None:
            language = self.detect_language(text)
        
        # Convert to lowercase and remove punctuation
        clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
        words = clean_text.split()
        
        keywords = []
        
        if language == 'ta':
            # Extract Tamil agricultural terms
            for word in words:
                if any(tamil_word in word for tamil_word in self.tamil_words):
                    keywords.append(word)
        else:
            # Extract English agricultural terms
            for word in words:
                if word in self.english_agri_words:
                    keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def is_agricultural_query(self, text):
        """Determine if the text is related to agriculture"""
        detected_lang = self.detect_language(text)
        keywords = self.extract_keywords(text, detected_lang)
        
        # Agricultural indicators
        agri_indicators_en = [
            'farm', 'farming', 'crop', 'plant', 'grow', 'agriculture', 'cultivation',
            'harvest', 'yield', 'fertilizer', 'pesticide', 'irrigation', 'soil',
            'seed', 'disease', 'pest', 'weather', 'rain', 'drought', 'market'
        ]
        
        agri_indicators_ta = [
            'விவசாய', 'பயிர்', 'வேளாண்', 'சாகுபடி', 'அறுவடை', 'விளைச்சல்',
            'உரம்', 'மருந்து', 'நீர்', 'மண்', 'விதை', 'நோய்', 'பூச்சி'
        ]
        
        text_lower = text.lower()
        
        # Check for agricultural terms
        has_agri_terms = (
            any(term in text_lower for term in agri_indicators_en) or
            any(term in text for term in agri_indicators_ta) or
            len(keywords) > 0
        )
        
        return has_agri_terms
    
    def get_greeting_message(self, language='en'):
        """Get appropriate greeting message based on language"""
        greetings = {
            'en': "Hello! I'm AgriBot, your agricultural assistant. How can I help you with farming today?",
            'ta': "வணக்கம்! நான் AgriBot, உங்கள் விவசாய உதவியாளர். இன்று விவசாயத்தில் எப்படி உதவ முடியும்?",
            'tanglish': "Vanakkam! I'm AgriBot, your விவசாய assistant. Eppadi help பண்ணலாம் today?"
        }
        
        return greetings.get(language, greetings['en'])
    
    def format_error_message(self, error_message, language='en'):
        """Format error messages in appropriate language"""
        error_messages = {
            'en': f"I'm sorry, I encountered an error: {error_message}. Please try again or contact support.",
            'ta': f"மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது: {error_message}. மீண்டும் முயற்சிக்கவும் அல்லது உதவியை தொடர்பு கொள்ளுங்கள்.",
            'tanglish': f"Sorry, oru error ஏற்பட்டது: {error_message}. Please try again அல்லது support-ஐ contact பண்ணுங்கள்."
        }
        
        return error_messages.get(language, error_messages['en'])
