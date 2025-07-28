import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from pathlib import Path
import sys
import traceback
import re
from collections import Counter
import statistics

# Initialize page config FIRST - before any other streamlit commands
import streamlit as st

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="PPT Quality Checker Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Core imports with error handling - using session state to track errors
def check_dependencies():
    """Check and report missing dependencies without using st.error during import"""
    missing_deps = []

    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from wordcloud import WordCloud
    except ImportError as e:
        missing_deps.append(f"Core dependencies: {e}")

    # NLTK with safe initialization
    try:
        import nltk
        from nltk.data import find
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
    except ImportError:
        missing_deps.append("NLTK not installed. Please run: pip install nltk")

    # LangChain imports with error handling
    try:
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        from langchain_groq import ChatGroq
        from langchain_core.prompts import (
            SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
        )
        from langchain_core.output_parsers import StrOutputParser
    except ImportError as e:
        missing_deps.append(f"LangChain dependencies missing: {e}")

    # Document processing imports
    try:
        from unstructured.partition.pptx import partition_pptx
        global UNSTRUCTURED_AVAILABLE
        UNSTRUCTURED_AVAILABLE = True
    except ImportError:
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE
            UNSTRUCTURED_AVAILABLE = False
        except ImportError:
            missing_deps.append("No PowerPoint processing library available. Please install: pip install python-pptx")

    return missing_deps


# Store dependency check results
if 'dependency_check_done' not in st.session_state:
    st.session_state.dependency_errors = check_dependencies()
    st.session_state.dependency_check_done = True

# Now import the rest after page config is set
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud

    import nltk
    from nltk.data import find
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords

    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    from langchain_groq import ChatGroq
    from langchain_core.prompts import (
        SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
    )
    from langchain_core.output_parsers import StrOutputParser

    # Document processing imports
    try:
        from unstructured.partition.pptx import partition_pptx

        UNSTRUCTURED_AVAILABLE = True
    except ImportError:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE

        UNSTRUCTURED_AVAILABLE = False

except ImportError:
    # We'll handle this in the main function
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def safe_nltk_download():
    """Safely download NLTK data with error handling"""
    required_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    success = True
    messages = []

    for data_item in required_data:
        try:
            if data_item == 'punkt':
                find('tokenizers/punkt')
            elif data_item == 'stopwords':
                find('corpora/stopwords')
            elif data_item == 'averaged_perceptron_tagger':
                find('taggers/averaged_perceptron_tagger')
        except LookupError:
            try:
                nltk.download(data_item, quiet=True)
                messages.append(f"✅ NLTK {data_item} downloaded")
            except Exception as e:
                messages.append(f"NLTK download warning for {data_item}: {e}")
                success = False

    return success, messages


class PPTQualityChecker:
    """Comprehensive PowerPoint Quality Analysis Tool"""

    SUPPORTED_MODELS = {
        "mixtral-8x7b-32768": "Mixtral 8x7B (Recommended)",
        "llama3-70b-8192": "Llama 3 70B",
        "llama3-8b-8192": "Llama 3 8B (Fast)",
        "gemma-7b-it": "Gemma 7B"
    }

    QUALITY_CATEGORIES = {
        "content": "Content Quality & Structure",
        "design": "Visual Design & Layout",
        "readability": "Text Readability & Clarity",
        "engagement": "Audience Engagement",
        "accessibility": "Accessibility Standards",
        "professional": "Professional Standards"
    }

    def __init__(self):
        """Initialize the quality checker with safe setup"""
        try:
            self.session_id = self._generate_session_id()
            self.data_dir = Path("data")
            self.cache_dir = Path("cache")
            self.reports_dir = Path("quality_reports")
            self._ensure_directories()
            self._initialize_nltk()
            logger.info(f"PPTQualityChecker initialized with session ID: {self.session_id}")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return hashlib.md5(f"{datetime.now().isoformat()}".encode()).hexdigest()[:8]

    def _ensure_directories(self):
        """Create necessary directories with error handling"""
        try:
            for directory in [self.data_dir, self.cache_dir, self.reports_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            # Fallback to current directory
            self.data_dir = Path(".")
            self.cache_dir = Path(".")
            self.reports_dir = Path(".")

    def _initialize_nltk(self):
        """Initialize NLTK with comprehensive error handling"""
        try:
            success, messages = safe_nltk_download()
            if not success:
                logger.warning("NLTK initialization had warnings but continuing...")
            # Store messages for display in UI
            if 'nltk_messages' not in st.session_state:
                st.session_state.nltk_messages = messages
        except Exception as e:
            logger.error(f"NLTK initialization failed: {str(e)}")
            if 'nltk_error' not in st.session_state:
                st.session_state.nltk_error = str(e)

    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save uploaded file with comprehensive error handling"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in (' ', '.', '_')).rstrip()
            filename = f"{timestamp}_{clean_name}"
            save_path = self.data_dir / filename

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info(f"File saved successfully: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise Exception(f"Could not save uploaded file: {str(e)}")

    def get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file caching with error handling"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate file hash: {str(e)}")
            return str(int(time.time()))

    def _extract_detailed_content(self, file_path: Path) -> Dict[str, Any]:
        """Extract detailed content and metadata from PowerPoint"""
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE

            prs = Presentation(str(file_path))
            detailed_data = {
                'slides': [],
                'total_slides': len(prs.slides),
                'slide_layouts': [],
                'fonts_used': set(),
                'colors_used': set(),
                'images_count': 0,
                'charts_count': 0,
                'tables_count': 0,
                'shapes_count': 0
            }

            for i, slide in enumerate(prs.slides, 1):
                slide_data = {
                    'slide_number': i,
                    'layout_name': slide.slide_layout.name if slide.slide_layout else "Unknown",
                    'text_content': [],
                    'word_count': 0,
                    'sentence_count': 0,
                    'bullet_points': 0,
                    'images': 0,
                    'charts': 0,
                    'tables': 0,
                    'shapes': 0,
                    'text_boxes': 0
                }

                detailed_data['slide_layouts'].append(slide.slide_layout.name if slide.slide_layout else "Unknown")

                for shape in slide.shapes:
                    detailed_data['shapes_count'] += 1
                    slide_data['shapes'] += 1

                    if hasattr(shape, "text") and shape.text.strip():
                        text_content = shape.text.strip()
                        slide_data['text_content'].append(text_content)
                        slide_data['word_count'] += len(text_content.split())

                        # Count sentences
                        try:
                            sentences = sent_tokenize(text_content)
                            slide_data['sentence_count'] += len(sentences)
                        except:
                            slide_data['sentence_count'] += text_content.count('.') + text_content.count(
                                '!') + text_content.count('?')

                        # Count bullet points
                        slide_data['bullet_points'] += text_content.count('•') + text_content.count(
                            '*') + text_content.count('-')
                        slide_data['text_boxes'] += 1

                    # Count different shape types
                    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                        slide_data['images'] += 1
                        detailed_data['images_count'] += 1
                    elif shape.shape_type == MSO_SHAPE_TYPE.CHART:
                        slide_data['charts'] += 1
                        detailed_data['charts_count'] += 1
                    elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                        slide_data['tables'] += 1
                        detailed_data['tables_count'] += 1

                detailed_data['slides'].append(slide_data)

            return detailed_data

        except Exception as e:
            logger.error(f"Detailed content extraction failed: {str(e)}")
            return {'slides': [], 'total_slides': 0, 'error': str(e)}

    def _analyze_content_quality(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content quality metrics"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            # Word count analysis
            word_counts = [slide['word_count'] for slide in slides]
            avg_words = statistics.mean(word_counts) if word_counts else 0

            analysis['metrics']['average_words_per_slide'] = round(avg_words, 1)
            analysis['metrics']['total_words'] = sum(word_counts)
            analysis['metrics']['word_count_consistency'] = round(statistics.stdev(word_counts), 1) if len(
                word_counts) > 1 else 0

            # Content distribution
            text_heavy_slides = len([w for w in word_counts if w > 50])
            analysis['metrics']['text_heavy_slides'] = text_heavy_slides
            analysis['metrics']['text_heavy_percentage'] = round((text_heavy_slides / len(slides)) * 100, 1)

            # Bullet points analysis
            bullet_counts = [slide['bullet_points'] for slide in slides]
            analysis['metrics']['average_bullets_per_slide'] = round(statistics.mean(bullet_counts),
                                                                     1) if bullet_counts else 0

            # Score calculation
            score = 100

            # Penalize for too many words per slide
            if avg_words > 40:
                score -= min(30, (avg_words - 40) * 0.5)
                analysis['issues'].append(f"High word count per slide ({avg_words:.1f} avg)")
                analysis['recommendations'].append("Reduce text per slide to improve readability")

            # Penalize for inconsistent content
            if len(word_counts) > 1 and statistics.stdev(word_counts) > 30:
                score -= 15
                analysis['issues'].append("Inconsistent content distribution across slides")
                analysis['recommendations'].append("Balance content more evenly across slides")

            # Reward for good structure
            if 5 <= avg_words <= 30:
                score += 10
                analysis['recommendations'].append("Good balance of text content")

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Content quality analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def _analyze_design_quality(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze design and visual quality"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            # Layout consistency
            layouts = detailed_data.get('slide_layouts', [])
            unique_layouts = len(set(layouts))
            layout_consistency = (1 - (unique_layouts / len(layouts))) * 100 if layouts else 0

            analysis['metrics']['unique_layouts'] = unique_layouts
            analysis['metrics']['layout_consistency'] = round(layout_consistency, 1)

            # Visual elements distribution
            total_images = sum(slide['images'] for slide in slides)
            total_charts = sum(slide['charts'] for slide in slides)
            total_tables = sum(slide['tables'] for slide in slides)

            analysis['metrics']['total_images'] = total_images
            analysis['metrics']['total_charts'] = total_charts
            analysis['metrics']['total_tables'] = total_tables
            analysis['metrics']['visual_elements_per_slide'] = round(
                (total_images + total_charts + total_tables) / len(slides), 1)

            # Text-only slides
            text_only_slides = len(
                [slide for slide in slides if slide['images'] == 0 and slide['charts'] == 0 and slide['tables'] == 0])
            analysis['metrics']['text_only_slides'] = text_only_slides
            analysis['metrics']['text_only_percentage'] = round((text_only_slides / len(slides)) * 100, 1)

            # Score calculation
            score = 100

            # Penalize for too many text-only slides
            if text_only_slides / len(slides) > 0.7:
                score -= 25
                analysis['issues'].append(f"{text_only_slides} slides are text-only")
                analysis['recommendations'].append("Add visual elements to improve engagement")

            # Penalize for layout inconsistency
            if unique_layouts > len(slides) * 0.5:
                score -= 20
                analysis['issues'].append("Too many different layouts used")
                analysis['recommendations'].append("Use consistent slide layouts for better flow")

            # Reward for good visual balance
            visual_ratio = (total_images + total_charts + total_tables) / len(slides)
            if 0.3 <= visual_ratio <= 1.5:
                score += 15
                analysis['recommendations'].append("Good balance of visual elements")

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Design quality analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def _analyze_readability(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text readability and clarity"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            all_text = []
            sentence_lengths = []

            for slide in slides:
                for text in slide['text_content']:
                    all_text.append(text)
                    try:
                        sentences = sent_tokenize(text)
                        for sentence in sentences:
                            words = word_tokenize(sentence)
                            sentence_lengths.append(len(words))
                    except:
                        # Fallback sentence splitting
                        sentences = re.split(r'[.!?]+', text)
                        for sentence in sentences:
                            if sentence.strip():
                                sentence_lengths.append(len(sentence.split()))

            if not sentence_lengths:
                return analysis

            # Calculate metrics
            avg_sentence_length = statistics.mean(sentence_lengths)
            analysis['metrics']['average_sentence_length'] = round(avg_sentence_length, 1)
            analysis['metrics']['total_sentences'] = len(sentence_lengths)

            # Word complexity (count of long words)
            all_words = []
            for text in all_text:
                try:
                    words = word_tokenize(text.lower()) if text else []
                except:
                    words = text.lower().split() if text else []
                all_words.extend(words)

            long_words = [word for word in all_words if len(word) > 6]
            complexity_ratio = len(long_words) / len(all_words) if all_words else 0
            analysis['metrics']['complexity_ratio'] = round(complexity_ratio * 100, 1)

            # Score calculation
            score = 100

            # Penalize for long sentences
            if avg_sentence_length > 20:
                score -= min(30, (avg_sentence_length - 20) * 1.5)
                analysis['issues'].append(f"Long sentences (avg {avg_sentence_length:.1f} words)")
                analysis['recommendations'].append("Break down complex sentences for better readability")

            # Penalize for high complexity
            if complexity_ratio > 0.3:
                score -= 20
                analysis['issues'].append(f"High word complexity ({complexity_ratio * 100:.1f}%)")
                analysis['recommendations'].append("Use simpler vocabulary for better understanding")

            # Reward for good readability
            if 8 <= avg_sentence_length <= 15:
                score += 15
                analysis['recommendations'].append("Good sentence length for presentations")

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Readability analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def _analyze_engagement(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audience engagement potential"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            # Question analysis
            all_text = ' '.join([' '.join(slide['text_content']) for slide in slides])
            question_count = all_text.count('?')
            analysis['metrics']['question_count'] = question_count

            # Interactive elements
            interactive_slides = 0
            for slide in slides:
                if slide['charts'] > 0 or slide['images'] > 0 or slide['tables'] > 0:
                    interactive_slides += 1

            analysis['metrics']['interactive_slides'] = interactive_slides
            analysis['metrics']['interactive_percentage'] = round((interactive_slides / len(slides)) * 100, 1)

            # Call-to-action detection
            cta_keywords = ['action', 'next steps', 'implement', 'apply', 'try', 'start', 'begin']
            cta_count = sum(all_text.lower().count(keyword) for keyword in cta_keywords)
            analysis['metrics']['call_to_action_indicators'] = cta_count

            # Score calculation
            score = 100

            # Questions boost engagement
            if question_count > 0:
                score += min(20, question_count * 5)
                analysis['recommendations'].append(f"Good use of questions ({question_count})")
            else:
                score -= 15
                analysis['issues'].append("No questions found to engage audience")
                analysis['recommendations'].append("Add rhetorical or direct questions to engage audience")

            # Interactive elements
            if interactive_slides / len(slides) < 0.3:
                score -= 20
                analysis['issues'].append("Limited visual/interactive elements")
                analysis['recommendations'].append("Add more charts, images, or interactive elements")

            # Call-to-action
            if cta_count > 0:
                score += 10
                analysis['recommendations'].append("Good use of action-oriented language")
            else:
                analysis['recommendations'].append("Consider adding clear call-to-action statements")

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Engagement analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def _analyze_accessibility(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze accessibility standards compliance"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            # Text-heavy slides (accessibility concern)
            heavy_text_slides = len([slide for slide in slides if slide['word_count'] > 50])
            analysis['metrics']['heavy_text_slides'] = heavy_text_slides

            # Images without alt text (assumption - would need deeper analysis)
            total_images = sum(slide['images'] for slide in slides)
            analysis['metrics']['total_images'] = total_images

            # Font readability indicators
            all_text = ' '.join([' '.join(slide['text_content']) for slide in slides])

            # Check for potential readability issues
            all_caps_ratio = len(re.findall(r'\b[A-Z]{3,}\b', all_text)) / len(all_text.split()) if all_text else 0
            analysis['metrics']['all_caps_ratio'] = round(all_caps_ratio * 100, 1)

            # Score calculation
            score = 100

            # Penalize for accessibility issues
            if heavy_text_slides > 0:
                score -= heavy_text_slides * 5
                analysis['issues'].append(f"{heavy_text_slides} slides have excessive text")
                analysis['recommendations'].append("Reduce text density for better accessibility")

            if all_caps_ratio > 0.1:
                score -= 15
                analysis['issues'].append("Excessive use of ALL CAPS text")
                analysis['recommendations'].append("Avoid ALL CAPS for better readability")

            # Accessibility best practices
            analysis['recommendations'].extend([
                "Ensure sufficient color contrast for text",
                "Provide alt text for images",
                "Use clear, readable fonts",
                "Maintain consistent navigation"
            ])

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Accessibility analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def _analyze_professional_standards(self, detailed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze professional presentation standards"""
        try:
            analysis = {
                'overall_score': 0,
                'metrics': {},
                'issues': [],
                'recommendations': []
            }

            slides = detailed_data.get('slides', [])
            if not slides:
                return analysis

            # Slide count analysis
            slide_count = len(slides)
            analysis['metrics']['total_slides'] = slide_count

            # Content structure
            title_slides = len([slide for slide in slides if slide['word_count'] < 10])
            content_distribution = [slide['word_count'] for slide in slides]

            analysis['metrics']['title_slides'] = title_slides
            analysis['metrics']['content_variance'] = round(statistics.stdev(content_distribution), 1) if len(
                content_distribution) > 1 else 0

            # Professional indicators
            all_text = ' '.join([' '.join(slide['text_content']) for slide in slides])

            # Check for professional language patterns
            informal_words = ['gonna', 'wanna', 'kinda', 'yeah', 'ok', 'awesome', 'cool']
            informal_count = sum(all_text.lower().count(word) for word in informal_words)
            analysis['metrics']['informal_language_count'] = informal_count

            # Score calculation
            score = 100

            # Optimal slide count (10-20 slides for most presentations)
            if slide_count < 5:
                score -= 20
                analysis['issues'].append("Very short presentation")
                analysis['recommendations'].append("Consider expanding content for more comprehensive coverage")
            elif slide_count > 30:
                score -= 15
                analysis['issues'].append("Very long presentation")
                analysis['recommendations'].append(
                    "Consider condensing content or splitting into multiple presentations")

            # Professional language
            if informal_count > 0:
                score -= informal_count * 5
                analysis['issues'].append(f"Informal language detected ({informal_count} instances)")
                analysis['recommendations'].append("Use more formal, professional language")

            # Structure consistency
            if analysis['metrics']['content_variance'] > 40:
                score -= 10
                analysis['issues'].append("Inconsistent slide content structure")
                analysis['recommendations'].append("Maintain consistent content structure across slides")

            analysis['overall_score'] = max(0, min(100, score))
            return analysis

        except Exception as e:
            logger.error(f"Professional standards analysis failed: {str(e)}")
            return {'overall_score': 0, 'metrics': {}, 'issues': [str(e)], 'recommendations': []}

    def generate_ai_feedback(self, detailed_data: Dict[str, Any], quality_analysis: Dict[str, Any],
                             api_key: str, model: str) -> str:
        """Generate AI-powered comprehensive feedback"""
        try:
            # Prepare context for AI
            context = {
                'total_slides': detailed_data.get('total_slides', 0),
                'overall_scores': {category: analysis.get('overall_score', 0)
                                   for category, analysis in quality_analysis.items()},
                'key_issues': [],
                'key_recommendations': []
            }

            for category, analysis in quality_analysis.items():
                context['key_issues'].extend(analysis.get('issues', []))
                context['key_recommendations'].extend(analysis.get('recommendations', []))

            # Create AI prompt
            system_prompt = """You are an expert presentation consultant with years of experience in 
            analyzing and improving PowerPoint presentations. You provide constructive, actionable 
            feedback that helps users create more effective presentations."""

            user_prompt = f"""
            Analyze this PowerPoint presentation and provide comprehensive feedback:

            Presentation Overview:
            - Total Slides: {context['total_slides']}
            - Content Quality Score: {context['overall_scores'].get('content', 0)}/100
            - Design Quality Score: {context['overall_scores'].get('design', 0)}/100
            - Readability Score: {context['overall_scores'].get('readability', 0)}/100
            - Engagement Score: {context['overall_scores'].get('engagement', 0)}/100
            - Accessibility Score: {context['overall_scores'].get('accessibility', 0)}/100
            - Professional Standards Score: {context['overall_scores'].get('professional', 0)}/100

            Key Issues Identified:
            {chr(10).join('- ' + issue for issue in context['key_issues'][:10])}

            Key Recommendations:
            {chr(10).join('- ' + rec for rec in context['key_recommendations'][:10])}

            Please provide:
            1. Overall Assessment (2-3 sentences)
            2. Top 3 Strengths
            3. Top 3 Areas for Improvement
            4. Specific Action Items (5-7 concrete steps)
            5. Professional Rating (Excellent/Good/Fair/Needs Improvement)

            Keep your response practical and actionable.
            """

            # Initialize AI model
            llm = ChatGroq(
                model=model,
                api_key=api_key,
                temperature=0.3,
                max_tokens=2048,
                timeout=30
            )

            # Create prompt template
            system_message = SystemMessagePromptTemplate.from_template(system_prompt)
            human_message = HumanMessagePromptTemplate.from_template(user_prompt)

            template = ChatPromptTemplate([system_message, human_message])
            chain = template | llm | StrOutputParser()

            # Generate response
            response = chain.invoke({})

            return response if response else "AI feedback generation failed - please check your API key and try again."

        except Exception as e:
            logger.error(f"AI feedback generation failed: {str(e)}")
            return f"AI feedback unavailable: {str(e)}"

    def run_comprehensive_analysis(self, file_path: Path, use_ai: bool = False,
                                   api_key: str = "", model: str = "") -> Dict[str, Any]:
        """Run comprehensive quality analysis"""
        try:
            logger.info("Starting comprehensive PPT analysis")

            # Extract detailed content
            detailed_data = self._extract_detailed_content(file_path)

            if 'error' in detailed_data:
                raise Exception(detailed_data['error'])

            # Run all quality analyses
            quality_analysis = {
                'content': self._analyze_content_quality(detailed_data),
                'design': self._analyze_design_quality(detailed_data),
                'readability': self._analyze_readability(detailed_data),
                'engagement': self._analyze_engagement(detailed_data),
                'accessibility': self._analyze_accessibility(detailed_data),
                'professional': self._analyze_professional_standards(detailed_data)
            }

            # Calculate overall score
            scores = [analysis['overall_score'] for analysis in quality_analysis.values()]
            overall_score = statistics.mean(scores) if scores else 0

            # Generate AI feedback if requested
            ai_feedback = ""
            if use_ai and api_key:
                ai_feedback = self.generate_ai_feedback(detailed_data, quality_analysis, api_key, model)

            # Compile results
            results = {
                'overall_score': round(overall_score, 1),
                'detailed_data': detailed_data,
                'quality_analysis': quality_analysis,
                'ai_feedback': ai_feedback,
                'analysis_timestamp': datetime.now().isoformat(),
                'file_name': file_path.name
            }

            logger.info(f"Analysis completed with overall score: {overall_score}")
            return results

        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {str(e)}")
            raise

    def save_quality_report(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save comprehensive quality report"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"quality_report_{timestamp}.md"

            report_path = self.reports_dir / filename

            # Generate markdown report
            report_content = f"""# PowerPoint Quality Analysis Report

**File:** {results['file_name']}  
**Analysis Date:** {datetime.fromisoformat(results['analysis_timestamp']).strftime('%Y-%m-%d %H:%M:%S')}  
**Overall Quality Score:** {results['overall_score']}/100

## Executive Summary

{'🟢 Excellent' if results['overall_score'] >= 85 else '🟡 Good' if results['overall_score'] >= 70 else '🟠 Fair' if results['overall_score'] >= 50 else '🔴 Needs Improvement'} - This presentation scored {results['overall_score']}/100 across all quality dimensions.

## Detailed Scores

| Category | Score | Status |
|----------|-------|--------|
| Content Quality | {results['quality_analysis']['content']['overall_score']}/100 | {'✅' if results['quality_analysis']['content']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['content']['overall_score'] >= 50 else '❌'} |
| Design Quality | {results['quality_analysis']['design']['overall_score']}/100 | {'✅' if results['quality_analysis']['design']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['design']['overall_score'] >= 50 else '❌'} |
| Readability | {results['quality_analysis']['readability']['overall_score']}/100 | {'✅' if results['quality_analysis']['readability']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['readability']['overall_score'] >= 50 else '❌'} |
| Engagement | {results['quality_analysis']['engagement']['overall_score']}/100 | {'✅' if results['quality_analysis']['engagement']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['engagement']['overall_score'] >= 50 else '❌'} |
| Accessibility | {results['quality_analysis']['accessibility']['overall_score']}/100 | {'✅' if results['quality_analysis']['accessibility']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['accessibility']['overall_score'] >= 50 else '❌'} |
| Professional Standards | {results['quality_analysis']['professional']['overall_score']}/100 | {'✅' if results['quality_analysis']['professional']['overall_score'] >= 70 else '⚠️' if results['quality_analysis']['professional']['overall_score'] >= 50 else '❌'} |

## Key Metrics

### Presentation Overview
- **Total Slides:** {results['detailed_data']['total_slides']}
- **Total Words:** {sum(slide['word_count'] for slide in results['detailed_data']['slides'])}
- **Visual Elements:** {results['detailed_data']['images_count']} images, {results['detailed_data']['charts_count']} charts, {results['detailed_data']['tables_count']} tables

### Content Analysis
"""

            # Add category-specific details
            for category, analysis in results['quality_analysis'].items():
                report_content += f"\n### {self.QUALITY_CATEGORIES[category]}\n\n"

                if analysis['issues']:
                    report_content += "**Issues Identified:**\n"
                    for issue in analysis['issues']:
                        report_content += f"- ❌ {issue}\n"
                    report_content += "\n"

                if analysis['recommendations']:
                    report_content += "**Recommendations:**\n"
                    for rec in analysis['recommendations']:
                        report_content += f"- 💡 {rec}\n"
                    report_content += "\n"

            # Add AI feedback if available
            if results['ai_feedback']:
                report_content += f"\n## AI-Powered Expert Analysis\n\n{results['ai_feedback']}\n"

            # Add detailed metrics
            report_content += "\n## Detailed Metrics\n\n"
            for category, analysis in results['quality_analysis'].items():
                if analysis['metrics']:
                    report_content += f"### {self.QUALITY_CATEGORIES[category]} Metrics\n\n"
                    for metric, value in analysis['metrics'].items():
                        report_content += f"- **{metric.replace('_', ' ').title()}:** {value}\n"
                    report_content += "\n"

            report_content += f"\n---\n*Report generated by PPT Quality Checker v2.0*"

            # Save report
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"Quality report saved: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Failed to save quality report: {str(e)}")
            raise Exception(f"Could not save quality report: {str(e)}")


def create_modern_ui():
    """Create modern Streamlit UI for quality checker"""
    try:
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .quality-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-left: 4px solid #2E86AB;
            margin: 1rem 0;
            transition: transform 0.2s ease;
        }

        .quality-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        .score-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
        }

        .status-excellent { color: #2E7D32; font-weight: bold; }
        .status-good { color: #F57C00; font-weight: bold; }
        .status-fair { color: #FF9800; font-weight: bold; }
        .status-poor { color: #D32F2F; font-weight: bold; }

        .stButton > button {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"UI styling failed: {str(e)}")


def display_quality_scores(results: Dict[str, Any]):
    """Display quality scores with visual indicators"""
    try:
        st.markdown("### 📊 Quality Assessment Dashboard")

        # Overall score
        overall_score = results['overall_score']
        status_class = (
            "status-excellent" if overall_score >= 85 else
            "status-good" if overall_score >= 70 else
            "status-fair" if overall_score >= 50 else
            "status-poor"
        )

        st.markdown(f"""
        <div class="score-card">
            <h2>Overall Quality Score</h2>
            <h1 class="{status_class}">{overall_score}/100</h1>
        </div>
        """, unsafe_allow_html=True)

        # Category scores
        col1, col2, col3 = st.columns(3)
        categories = list(results['quality_analysis'].keys())

        for i, (category, analysis) in enumerate(results['quality_analysis'].items()):
            score = analysis['overall_score']
            col = [col1, col2, col3][i % 3]

            status_emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"

            with col:
                st.markdown(f"""
                <div class="quality-card">
                    <h4>{status_emoji} {PPTQualityChecker.QUALITY_CATEGORIES[category]}</h4>
                    <h2>{score}/100</h2>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying scores: {str(e)}")


def display_detailed_analysis(results: Dict[str, Any]):
    """Display detailed analysis results"""
    try:
        st.markdown("### 🔍 Detailed Analysis")

        # Presentation overview
        detailed_data = results['detailed_data']
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Slides", detailed_data['total_slides'])
        with col2:
            total_words = sum(slide['word_count'] for slide in detailed_data['slides'])
            st.metric("Total Words", f"{total_words:,}")
        with col3:
            st.metric("Images", detailed_data['images_count'])
        with col4:
            st.metric("Charts & Tables", detailed_data['charts_count'] + detailed_data['tables_count'])

        # Category-wise analysis
        for category, analysis in results['quality_analysis'].items():
            with st.expander(f"📋 {PPTQualityChecker.QUALITY_CATEGORIES[category]} - {analysis['overall_score']}/100"):

                # Issues
                if analysis['issues']:
                    st.markdown("**⚠️ Issues Identified:**")
                    for issue in analysis['issues']:
                        st.markdown(f"- {issue}")

                # Recommendations
                if analysis['recommendations']:
                    st.markdown("**💡 Recommendations:**")
                    for rec in analysis['recommendations']:
                        st.markdown(f"- {rec}")

                # Metrics
                if analysis['metrics']:
                    st.markdown("**📊 Detailed Metrics:**")
                    metrics_df = pd.DataFrame([
                        {"Metric": k.replace('_', ' ').title(), "Value": v}
                        for k, v in analysis['metrics'].items()
                    ])
                    st.dataframe(metrics_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying detailed analysis: {str(e)}")


def create_visualizations(results: Dict[str, Any]):
    """Create visualizations for the analysis"""
    try:
        st.markdown("### 📈 Visual Analysis")

        # Score radar chart
        categories = list(results['quality_analysis'].keys())
        scores = [results['quality_analysis'][cat]['overall_score'] for cat in categories]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar chart of scores
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
        bars = ax1.bar(range(len(categories)), scores, color=colors[:len(categories)])
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels([cat.title() for cat in categories], rotation=45, ha='right')
        ax1.set_ylabel('Quality Score')
        ax1.set_title('Quality Scores by Category')
        ax1.set_ylim(0, 100)

        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{score:.0f}', ha='center', va='bottom', fontweight='bold')

        # Slide-wise word count distribution
        detailed_data = results['detailed_data']
        word_counts = [slide['word_count'] for slide in detailed_data['slides']]
        slide_numbers = list(range(1, len(word_counts) + 1))

        ax2.plot(slide_numbers, word_counts, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax2.fill_between(slide_numbers, word_counts, alpha=0.3, color='#2E86AB')
        ax2.set_xlabel('Slide Number')
        ax2.set_ylabel('Word Count')
        ax2.set_title('Content Distribution Across Slides')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Word cloud of content (if text available)
        if detailed_data['slides']:
            all_text = ' '.join([' '.join(slide['text_content']) for slide in detailed_data['slides']])
            if len(all_text) > 50:
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white',
                                          colormap='viridis').generate(all_text)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title('Content Word Cloud', fontsize=16, fontweight='bold')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate word cloud: {str(e)}")

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")


def main():
    """Main application function"""
    try:
        # Check for dependency errors first
        if st.session_state.dependency_errors:
            st.error("❌ Missing Dependencies")
            for error in st.session_state.dependency_errors:
                st.error(error)
            st.info("Please install missing dependencies and restart the application.")
            st.stop()

        create_modern_ui()

        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🎯 PowerPoint Quality Checker Pro</h1>
            <p>Comprehensive analysis and improvement recommendations for your presentations</p>
        </div>
        """, unsafe_allow_html=True)

        # Display NLTK messages if any
        if hasattr(st.session_state, 'nltk_messages') and st.session_state.nltk_messages:
            with st.expander("🔧 System Setup Messages"):
                for msg in st.session_state.nltk_messages:
                    st.info(msg)

        # Initialize checker
        try:
            if 'checker' not in st.session_state:
                with st.spinner("🚀 Initializing Quality Checker..."):
                    st.session_state.checker = PPTQualityChecker()
            checker = st.session_state.checker
        except Exception as e:
            st.error(f"❌ Application initialization failed: {str(e)}")
            st.stop()

        # Sidebar Configuration
        with st.sidebar:
            st.header("⚙️ Analysis Configuration")

            # Quality Categories Selection
            st.subheader("📋 Analysis Categories")
            selected_categories = st.multiselect(
                "Select quality dimensions to analyze:",
                options=list(checker.QUALITY_CATEGORIES.keys()),
                default=list(checker.QUALITY_CATEGORIES.keys()),
                format_func=lambda x: checker.QUALITY_CATEGORIES[x]
            )

            # AI Analysis Settings
            st.subheader("🤖 AI-Powered Analysis")
            enable_ai = st.checkbox("Enable AI Expert Feedback", value=False)

            if enable_ai:
                groq_api_key = st.text_input(
                    "GROQ API Key",
                    type="password",
                    help="Required for AI analysis"
                )

                ai_model = st.selectbox(
                    "AI Model",
                    options=list(checker.SUPPORTED_MODELS.keys()),
                    format_func=lambda x: checker.SUPPORTED_MODELS[x]
                )

                if groq_api_key:
                    st.success("✅ AI Analysis Ready")
            else:
                groq_api_key = ""
                ai_model = "mixtral-8x7b-32768"

            # Analysis Options
            st.subheader("🔧 Options")
            generate_report = st.checkbox("Generate Detailed Report", value=True)
            create_visuals = st.checkbox("Create Visualizations", value=True)

        # Main Content
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            <div class="quality-card">
                <h3>📁 Upload Your Presentation</h3>
                <p>Upload your PowerPoint file (.pptx) to start quality analysis</p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Choose a PowerPoint file",
                type=["pptx"],
                help="Only .pptx files are supported"
            )

            if uploaded_file is not None:
                # File info
                file_info = {
                    "📄 Name": uploaded_file.name,
                    "📏 Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "🏷️ Type": uploaded_file.type
                }

                st.markdown("**File Information:**")
                for key, value in file_info.items():
                    st.text(f"{key}: {value}")

                # Analysis button
                if st.button("🎯 Start Quality Analysis", type="primary"):
                    try:
                        # Save file
                        with st.spinner("📤 Processing file..."):
                            save_path = checker.save_uploaded_file(uploaded_file)

                        # Run analysis
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        status_text.text("🔍 Analyzing presentation quality...")
                        progress_bar.progress(25)

                        # Perform analysis
                        results = checker.run_comprehensive_analysis(
                            save_path,
                            use_ai=enable_ai and bool(groq_api_key),
                            api_key=groq_api_key,
                            model=ai_model
                        )

                        progress_bar.progress(75)
                        status_text.text("📊 Generating insights...")

                        # Display results
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")

                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()

                        # Store results in session state
                        st.session_state.analysis_results = results

                        st.success("🎉 Quality analysis completed successfully!")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}")

        with col2:
            # Info panels
            st.markdown("""
            <div class="quality-card">
                <h3>🎯 What We Analyze</h3>
                <ul>
                    <li><strong>Content Quality:</strong> Structure, clarity, word count</li>
                    <li><strong>Design Quality:</strong> Visual elements, layout consistency</li>
                    <li><strong>Readability:</strong> Text complexity, sentence length</li>
                    <li><strong>Engagement:</strong> Interactive elements, questions</li>
                    <li><strong>Accessibility:</strong> Inclusive design standards</li>
                    <li><strong>Professional Standards:</strong> Best practices compliance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="quality-card">
                <h3>💡 Pro Tips</h3>
                <ul>
                    <li>Keep slides concise (20-40 words)</li>
                    <li>Use consistent layouts</li>
                    <li>Include visual elements</li>
                    <li>Add engaging questions</li>
                    <li>Ensure accessibility</li>
                    <li>Maintain professional tone</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Display results if available
        if 'analysis_results' in st.session_state:
            results = st.session_state.analysis_results

            st.markdown("---")

            # Quality scores dashboard
            display_quality_scores(results)

            # AI feedback
            if results.get('ai_feedback'):
                st.markdown("### 🤖 AI Expert Analysis")
                st.markdown(results['ai_feedback'])

            # Detailed analysis
            display_detailed_analysis(results)

            # Visualizations
            if create_visuals:
                create_visualizations(results)

            # Generate and download report
            if generate_report:
                try:
                    report_path = checker.save_quality_report(results)

                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()

                    st.download_button(
                        label="📥 Download Quality Report",
                        data=report_content,
                        file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        type="primary"
                    )
                except Exception as e:
                    st.warning(f"Report generation failed: {str(e)}")

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>🎯 PowerPoint Quality Checker Pro v2.0</p>
            <p>Built for comprehensive presentation analysis with ❤️</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Critical application error: {str(e)}")
        logger.error(f"Critical error in main: {str(e)}")


if __name__ == "__main__":
    main()