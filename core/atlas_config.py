import os
import json
import re
import hashlib
import inspect
import logging
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').setLevel(logging.ERROR)
logging.getLogger('torch.utils._pytree').setLevel(logging.ERROR)

try:
    import orjson
    def json_loads(data): return orjson.loads(data)
    def json_dumps(obj): return orjson.dumps(obj).decode('utf-8')
    JSON_LIB = 'orjson'
except ImportError:
    json_loads = json.loads
    json_dumps = json.dumps
    JSON_LIB = 'json'

import numpy as np
import csv
import webbrowser
import math
import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict, deque
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import argparse
import warnings
import lzma
from lexicalrichness import LexicalRichness
import umap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import enchant
    ENCHANT_AVAILABLE = True
except ImportError:
    ENCHANT_AVAILABLE = False

try:
    from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
except ImportError:
    SklearnHDBSCAN = None

warnings.filterwarnings('ignore')
os.environ.update({'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})

def sanitize_filename(name: str) -> str:
    name = name.replace('..', '').replace('/', '').replace('\\', '')
    name = name.replace('\0', '')
    return ''.join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.')).strip()

def validate_text_name(text: str) -> bool:
    if not text or len(text) > 255:
        return False
    if '..' in text or '/' in text or '\\' in text:
        return False
    if '\0' in text:
        return False
    return True

def escape_html_attribute(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return html.escape(text, quote=True)

def escape_for_javascript(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return json.dumps(text)

def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=True)

CACHE_FILTER_KEYS = {
    'date_from', 'date_to',
    'duration_from', 'duration_to',
    'exclude_live',
    'min_tokens_per_file',
    'token_limit',
    'text_token_limit',
    'per_video_token_limit',
    'centroid_mode',
    'centroid_videos',
    'centroid_words',
    'stat_word_count',
}

SBERT_MODEL = os.getenv('UMAPPER_EMBEDDING_MODEL', 'all-MiniLM-L12-v2')

EMBEDDING_MODEL_PROFILES: Dict[str, Dict[str, int]] = {
    'thenlper/gte-small': {'context_window': 512, 'chunk_target': 384, 'min_chunk': 192, 'batch_size': 128, 'practical_max_tokens_cpu': 384, 'practical_max_tokens_mps': 512, 'practical_max_tokens_cuda': 512},
    'all-MiniLM-L12-v2': {'context_window': 256, 'chunk_target': 220, 'min_chunk': 160, 'batch_size': 256},
    'my_finetuned_sbert': {'context_window': 256, 'chunk_target': 220, 'min_chunk': 160, 'batch_size': 256},
    'all-MiniLM-L6-v2': {'context_window': 256, 'chunk_target': 220, 'min_chunk': 160, 'batch_size': 256},
    'all-mpnet-base-v2': {'context_window': 384, 'chunk_target': 320, 'min_chunk': 220, 'batch_size': 192},
    'Alibaba-NLP/gte-modernbert-base': {'context_window': 8192, 'chunk_target': 768, 'min_chunk': 256, 'batch_size': 4, 'practical_max_tokens_cpu': 768, 'practical_max_tokens_mps': 1024, 'practical_max_tokens_cuda': 2048},
}

MODEL_NAME_ALIASES: Dict[str, str] = {
    'gte-modernbert-base': 'Alibaba-NLP/gte-modernbert-base',
    'modernbert': 'Alibaba-NLP/gte-modernbert-base',
    'gte-small': 'thenlper/gte-small',
    'gte_small': 'thenlper/gte-small',
}

def resolve_embedding_model_name(model_name: Optional[str]) -> str:
    requested = (model_name or SBERT_MODEL).strip()
    return MODEL_NAME_ALIASES.get(requested, requested)

def get_embedding_model_profile(model_name: str) -> Dict[str, int]:
    profile = EMBEDDING_MODEL_PROFILES.get(model_name)
    if profile is not None:
        return profile
    return {'context_window': 512, 'chunk_target': 384, 'min_chunk': 256, 'batch_size': 128, 'practical_max_tokens_cpu': 384, 'practical_max_tokens_mps': 512, 'practical_max_tokens_cuda': 1024}

perplexity_model = None
perplexity_tokenizer = None
vader_analyzer = SentimentIntensityAnalyzer()
spacy_nlp = None
enchant_dict = None
enchant_word_check_cache: Dict[str, bool] = {}
spacy_embedding_cache = {}
sbert_token_embedding_cache = {}

TOKEN_REGEX = re.compile(r"([^\w\s']|(?<!\w)'|'(?!\w)|\s+|'+\s)")

INSULT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bstupid\w*\b', r'\bdumb\w*\b', r'\bidiot\w*\b', r'\bnerd\w*\b', r'\bdork\w*\b', r'\bdummy\w*\b',
    r'\bjerks?\b', r'\bdingle\w*\b', r'\bdoofus\w*\b', r'\bcoward\w*\b', r'\bdummies\w*\b',
    r'\bbuffoon\w*\b', r'\blame\w*\b', r'\bgoofball\w*\b', r'\bnoob\w*\b', r'\b(?:these|this|that|another|a|the)\s+fools?\b',
    r'\bnincompoop\w*\b', r'\bcretin\w*\b', r'\bmoron\w*\b', r'\b(?<!the )loser\w*\b', r'\brascal\w*\b', r'\bhooligan\w*\b',
    r"\byou(?:\s+\w+)?\s+fool\b", r"\byou(?:\s+\w+)?\s+prick\b", r'\bskank\w*\b', r'\bugly\b',
    r'\b(?<!cyber)(?<!cyber\s)(?<!cm)(?<!cm\s)punk(?:s)?\b(?!\s+(?:music|rock|disc|discs|record|album|band|song|genre))'
]]

DEV_SPECIALTY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bthis\s+is\s+a\b']]
CUSTOM_MARKER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bi\s+think\b', r'\bprobably\b', r'\bi\s+feel\b', r'\bmaybe\b'
]]
SMORE_MARKERS_2_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\balso\b', r'\btoo\b']]
HYPO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b\w*hypo\w*\b']]

HEDGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b', r'\bprobably\b', r'\bseems?\b', r'\bappears?\b',
    r'\bmight\b', r'\bcould\b', r'\bsort of\b', r'\bkind of\b', r'\bkinda\b', r'\bsorta\b',
    r'\blike\b', r'\bbasically\b', r'\bactually\b', r'\bessentially\b', r'\bpretty much\b',
    r'\bmore or less\b', r'\bI think\b', r'\bI guess\b', r'\bI believe\b', r'\bI suppose\b',
    r'\bapparently\b', r'\bpresumably\b'
]]
DISCUSSION_COMMENTARY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bdiscuss\w*\b', r'\bdiscussion\b', r'\bconvers\w*\b', r'\bchat(?:ting)?\b', r'\bchat about\b',
    r'\btalk(?:ing)? about\b', r'\btalk (?:with|to)\b', r"\blet's talk\b", r'\bcommentary\b',
    r'\bwhat do you think\b', r'\blet me know\b', r'\bshare your\b', r'\bwe need to talk\b',
    r'\bquestion\w*\b', r'\bask\w*\b', r'\banswer\w*\b', r'\bopinion\w*\b', r'\bthoughts?\b',
    r'\bq\s*a\b', r'\bq and a\b', r'\bask me anything\b', r'\bama\b',
    r'\binterview\w*\b', r'\binterview with\b', r'\bsit[- ]down\b', r'\bfireside chat\b',
    r'\bguest\b', r'\bguest host\b', r'\bco[- ]?host\b', r'\bhosted by\b', r'\bmoderator\b', r'\bmoderated by\b',
    r'\bpanel discussion\b', r'\bpanel\b', r'\broundtable\b', r'\bpodcast\b'
]]
ARTICLE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bthe\b', r'\ba\b', r'\ban\b']]
DEMONSTRATIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b', r'\bthere\b', r'\bhere\b']]
POSSESSIVE_TOKENS = {'my', 'your', 'yours', 'his', 'her', 'hers', 'its', 'our', 'ours', 'their', 'theirs', 'whose'}
QUANTIFIER_TOKENS = {
    'some', 'many', 'few', 'several', 'much', 'little', 'plenty', 'numerous', 'countless', 'various',
    'multiple', 'any', 'each', 'every', 'either', 'neither', 'both', 'all', 'enough', 'lot', 'lots',
    'more', 'most', 'less', 'least', 'fewer'
}
PRONOUN_DEICTIC_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bthis\b', r'\bthat\b', r'\bit\b', r'\byou\b', r'\bwe\b', r'\bi\b', r'\bme\b',
    r'\bmy\b', r'\bmine\b', r'\bour\b', r'\byour\b', r'\byours\b'
]]
PRONOUN_VOLATILITY_GROUPS = {
    'i': {'i', 'me', 'my', 'mine', 'myself'},
    'you': {'you', 'your', 'yours', 'yourself', 'yourselves'},
    'we': {'we', 'us', 'our', 'ours', 'ourselves'}
}
PRONOUN_VOLATILITY_TOKEN_TO_GROUP = {token: group for group, tokens in PRONOUN_VOLATILITY_GROUPS.items() for token in tokens}
EXCLAMATION_INTERJECTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\boh\b', r'\bwow\b', r'\bwoww+\b', r'\bwowza\b', r'\bamazing\b', r'\bincredible\b',
    r'\bawesome\b', r'\binsane\b', r'\bepic\b', r'\bcrazy\b', r'\bwild\b', r'\bdevil\b',
    r'\bdevils?\b', r'\boh my\b', r'\boh no\b', r'\bno way\b', r'\bholy\b',
    r'\byeah(?:h+)?\b', r'\byess?\b', r'\blet\'s go\b', r'\blet\'s goo+\b',
    r'\bcome on\b', r'\bbruh\b', r'\bbroo\b', r'\bmate\b', r'\bbro\b', r'\bman\b',
    r'\bdude\b', r'\bguys?\b'
]]
DEICTIC_SPATIAL_TEMPORAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bhere\b', r'\bthere\b', r'\bthis way\b', r'\bthat way\b', r'\bleft\b', r'\bright\b',
    r'\bup\b', r'\bdown\b', r'\bover\b', r'\baround\b', r'\bin\b', r'\bout\b',
    r'\bback\b', r'\bfront\b', r'\btop\b', r'\bbottom\b', r'\bnow\b', r'\bthen\b',
    r'\btoday\b', r'\btomorrow\b', r'\byesterday\b', r'\brecently\b', r'\bsoon\b',
    r'\blater\b', r'\blook\b', r'\bwatch\b', r'\bsee\b', r'\bcheck\b', r'\bgo\b',
    r'\bcome\b', r'\bcome back\b'
]]
ELABORATION_EXPLANATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bfor (example|instance)\b', r'\bspecifically\b', r'\bin (other )?words\b',
    r'\bthat is\b', r'\bi\.e\.\b', r'\bin fact\b', r'\bactually\b',
    r'\bmore (specifically|precisely)\b', r'\bin detail\b', r'\bas follows\b',
    r'\bnotably\b', r'\bparticularly\b', r'\bespecially\b', r'\bbecause\b',
    r'\bsince\b', r'\bdue to\b', r'\bas a result\b', r'\btherefore\b', r'\bthus\b',
    r'\bconsequently\b', r'\baccordingly\b', r'\bthe reason\b', r'\bcaused by\b',
    r'\bso that\b', r'\bin order to\b', r'\bso\b', r'\bwhich explains?\b', r'\bwhich means?\b'
]]
NARRATION_CONTINUATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bthen\b', r'\bafter\b', r'\bafterward\b', r'\bnext\b', r'\bfirst\b',
    r'\bsecond\b', r'\bfollowed by\b', r'\bwhile\b', r'\bduring\b', r'\bas\b',
    r'\bonce\b', r'\bsuddenly\b', r'\bnow\b', r'\bimmediately\b', r'\brealizing\b',
    r'\bstarting\b', r'\bbeginning\b', r'\band\b', r'\balso\b', r'\btoo\b',
    r'\bas well\b', r'\bplus\b', r'\bmoreover\b', r'\bfurthermore\b',
    r'\badditionally\b', r'\bsimilarly\b', r'\blikewise\b', r'\bor\b',
    r'\neither\b', r'\bneither\b', r'\bboth\b', r'\beither way\b'
]]
FAMILY_EXCLAMATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\boh my (god|gosh|goodness|word)\b', r'\bgosh\b', r'\bjeez\b', r'\bgeez\b',
    r'\bheck\b', r'\bdang\b', r'\bdarn\b', r'\bshoot\b', r'\boh boy\b', r'\bow\b',
    r'\boh man\b', r'\bgoodness\b', r'\bholy (cow|moly|smokes)\b', r'\bwowza\b',
    r'\byikes\b', r'\byowza\b', r'\buh oh\b', r'\boopsie\b', r'\bwhoops\b'
]]
SIMILE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\blike a\b', r'\blike an\b', r'\blike the\b', r'\blike some\b', r'\bas\s+\w+\s+as\b',
    r'\bseems like\b', r'\blooks like\b', r'\bfeels like\b', r'\bsounds like\b', r'\bacts like\b',
    r'\bworks like\b', r'\bruns like\b', r'\bsmells like\b', r'\bjust like\b', r'\bexactly like\b',
    r'\bkinda like\b', r'\bkind of like\b', r'\bsort of like\b', r'\ba bit like\b', r'\bmore like\b',
    r'\bless like\b', r'\bcompared to\b', r'\bremind\w* me of\b', r'\bsimilar to\b',
    r'\bresemble\w*\b', r'\bakin to\b'
]]
COLOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bred\b', r'\bredder\b', r'\breddest\b', r'\breddish\b',
    r'\bblue\b', r'\bbluer\b', r'\bbluest\b', r'\bbluish\b',
    r'\byellow\b', r'\byellower\b', r'\byellowest\b', r'\byellowish\b',
    r'\bgreen\b', r'\bgreener\b', r'\bgreenest\b', r'\bgreenish\b',
    r'\borange\b', r'\borangish\b', r'\bpurple\b', r'\bpurplish\b', r'\bpurpler\b',
    r'\bpink\b', r'\bpinkish\b', r'\bpinker\b', r'\bblack\b', r'\bblacker\b',
    r'\bblackest\b', r'\bblackish\b', r'\bwhite\b', r'\bwhiter\b', r'\bwhitest\b',
    r'\bwhitish\b', r'\bgray\b', r'\bgrey\b', r'\bgrayer\b', r'\bgreyer\b',
    r'\bgrayish\b', r'\bgreyish\b', r'\bbrown\b', r'\bbrowner\b', r'\bbrownest\b',
    r'\bbrownish\b', r'\bdark\b', r'\bdarker\b', r'\bdarkest\b', r'\blight\b',
    r'\blighter\b', r'\blightest\b', r'\bbright\b', r'\bbrighter\b', r'\bbrightest\b',
    r'\bpale\b', r'\bpaler\b', r'\bpalest\b', r'\bdeep\b', r'\bdeeper\b',
    r'\bdeepest\b', r'\bscarlet\b', r'\bcrimson\b', r'\bmaroon\b', r'\bburgundy\b',
    r'\bnavy\b', r'\bcyan\b', r'\bturquoise\b', r'\bteal\b', r'\baqua\b',
    r'\bgold\b', r'\bgolden\b', r'\bsilver\b', r'\bsilvery\b', r'\bbeige\b',
    r'\btan\b', r'\bkhaki\b', r'\biron\b', r'\bmagenta\b', r'\bviolet\b',
    r'\bindigo\b', r'\blavender\b', r'\bpeach\b', r'\bcoral\b', r'\bsalmon\b',
    r'\blime\b', r'\bolive\b', r'\bjade\b', r'\bsky blue\b', r'\bneon\b', r'\bmetallic\b'
]]
VULNERABILITY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bi don\'t know\b', r'\bi don\'t understand\b', r'\bi don\'t think\b',
    r'\bi\'m not sure\b', r'\bi\'m confused\b', r'\bi can\'t\b', r'\bi couldn\'t\b',
    r'\bi don\'t have\b', r'\bi don\'t like\b', r'\bi don\'t want\b',
    r'\bwe don\'t know\b', r'\bwe\'re not sure\b', r'\bwe can\'t\b',
    r'\bsorry\b', r'\bi feel\b', r'\bi\'m scared\b', r'\bi\'m worried\b',
    r'\bi\'m nervous\b', r'\bi\'m anxious\b', r'\bi\'m sad\b', r'\bi\'m upset\b',
    r'\bi\'m stressed\b', r'\bi made a mistake\b', r'\bi was wrong\b',
    r'\bi failed\b', r'\bi\'m failing\b', r'\bhelp me\b', r'\bi need help\b',
    r'\bi\'m struggling\b', r'\bi don\'t know what to do\b', r'\bwhat should i do\b',
    r'\bi\'m not good at\b', r'\bi\'m terrible at\b', r'\bno one likes me\b',
    r'\bi\'m lonely\b', r'\bi\'m alone\b', r'\bno one cares\b', r'\bwhy does this happen to me\b',
    r'\bi\'m not\b', r'\bi\'m bad\b', r'\bi\'m stupid\b', r'\bi\'m dumb\b'
]]

LATINATE_SUFFIXES = ('tion', 'sion', 'ology', 'logy', 'graphy', 'ism', 'ment', 'ity', 'ous', 'ence',
    'ance', 'ary', 'ory', 'ive', 'ize', 'ise', 'isation', 'ization', 'al', 'tude',
    'cracy', 'ible', 'able')
FIRST_PERSON_PRONOUNS = {'i', 'me', 'my', 'mine', 'we', 'our', 'us', 'ours'}
COORDINATORS = {'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'plus'}
SUBORDINATORS = {
    'because', 'although', 'though', 'while', 'whilst', 'if', 'unless', 'since', 'before',
    'after', 'until', 'when', 'whenever', 'whereas', 'wherever', 'once', 'than', 'that',
    'whether', 'as', 'lest', 'provided', 'whereby'
}
SIMPLE_COLOR_PATTERN_MAP = {
    'color_red_count': [re.compile(r'\bred(?:der|dest|dish)?\b', re.IGNORECASE)],
    'color_orange_count': [re.compile(r'\borange(?:ish)?\b', re.IGNORECASE)],
    'color_yellow_count': [re.compile(r'\byellow(?:er|est|ish)?\b', re.IGNORECASE)],
    'color_green_count': [re.compile(r'\bgreen(?:er|est|ish)?\b', re.IGNORECASE)],
    'color_blue_count': [re.compile(r'\bblue(?:r|st|ish)?\b', re.IGNORECASE)],
    'color_purple_count': [re.compile(r'\bpurple(?:r|st|ish)?\b', re.IGNORECASE)],
    'color_pink_count': [re.compile(r'\bpink(?:er|est|ish)?\b', re.IGNORECASE)],
}
INTERROGATIVE_FORM_MAP = {
    'how_count': {'how', 'however'}, 'what_count': {'what', 'whatever'}, 'which_count': {'which', 'whichever'},
    'who_count': {'who', 'whoever'}, 'whom_count': {'whom', 'whomever'}, 'whose_count': {'whose'},
    'when_count': {'when', 'whenever'}, 'where_count': {'where', 'wherever'}, 'why_count': {'why'},
}
CONJUNCTION_FORM_MAP = {f'conjunction_{conj}_count': {conj} for conj in sorted(COORDINATORS | SUBORDINATORS)}
PRONOUN_GROUP_FORM_MAP = {
    'pronoun_i_count': {'i', 'me', 'my', 'mine', 'myself'},
    'pronoun_you_count': {'you', 'your', 'yours', 'yourself', 'yourselves'},
    'pronoun_he_count': {'he', 'him', 'his', 'himself'},
    'pronoun_she_count': {'she', 'her', 'hers', 'herself'},
    'pronoun_they_count': {'they', 'them', 'their', 'theirs', 'themself', 'themselves'},
    'pronoun_we_count': {'we', 'us', 'our', 'ours', 'ourselves'},
    'pronoun_it_count': {'it', 'its', 'itself'},
}
AUXILIARY_GROUP_FORM_MAP = {
    'auxiliary_be_count': {'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been'},
    'auxiliary_do_count': {'do', 'does', 'did', 'doing', 'done'},
    'auxiliary_have_count': {'have', 'has', 'had', 'having'},
    'auxiliary_can_count': {'can', 'could'},
    'auxiliary_will_count': {'will', 'would'},
    'auxiliary_shall_count': {'shall', 'should'},
    'auxiliary_may_count': {'may', 'might'},
    'auxiliary_must_count': {'must'},
    'auxiliary_ought_count': {'ought'},
    'auxiliary_need_count': {'need', 'needs', 'needed'},
    'auxiliary_dare_count': {'dare', 'dares', 'dared'},
}

DYNAMIC_PATTERN_METRIC_MAP: Dict[str, List[re.Pattern]] = {}

DYNAMIC_TOKEN_SET_METRIC_MAP = {}
DYNAMIC_TOKEN_SET_METRIC_MAP.update(INTERROGATIVE_FORM_MAP)
DYNAMIC_TOKEN_SET_METRIC_MAP.update(CONJUNCTION_FORM_MAP)
DYNAMIC_TOKEN_SET_METRIC_MAP.update(PRONOUN_GROUP_FORM_MAP)
DYNAMIC_TOKEN_SET_METRIC_MAP.update(AUXILIARY_GROUP_FORM_MAP)
DYNAMIC_TOKEN_SET_METRIC_MAP['conjunction_total_count'] = set().union(*CONJUNCTION_FORM_MAP.values())
DYNAMIC_TOKEN_SET_METRIC_MAP['pronoun_total_count'] = set().union(*PRONOUN_GROUP_FORM_MAP.values())
DYNAMIC_TOKEN_SET_METRIC_MAP['pronoun_third_person_count'] = (
    PRONOUN_GROUP_FORM_MAP['pronoun_he_count'] | PRONOUN_GROUP_FORM_MAP['pronoun_she_count'] | PRONOUN_GROUP_FORM_MAP['pronoun_they_count']
)
DYNAMIC_TOKEN_SET_METRIC_MAP['auxiliary_total_count'] = set().union(*AUXILIARY_GROUP_FORM_MAP.values())
DYNAMIC_COUNT_FALLBACK_METHODS: Dict[str, str] = {}

def _format_trimmed_decimal(value: float, decimals: int = 4) -> str:
    try:
        formatted = f"{float(value):.{decimals}f}".rstrip('0').rstrip('.')
        return '0' if formatted in ('-0', '-0.0', '') else formatted
    except Exception:
        return str(value)

def _is_word_count_metric_config(metric_name: str, metric_config: Dict) -> bool:
    if not metric_name.endswith('_count'):
        return False
    if metric_name.startswith('letter_'):
        return False
    compute_method = metric_config.get('compute_method', '')
    if not (isinstance(compute_method, str) and compute_method.startswith('_compute_')):
        return False
    if compute_method in ('_compute_dynamic_token_count', '_compute_letter_count'):
        return False
    return True

DYNAMIC_METRIC_CONFIGS = {
    'unique_words_count': ('Unique Words Count', 'Count of unique normalized word tokens', 'Lexical Diversity'),
    'unique_oov_count': ('Unique OOV Count', 'Count of unique out-of-vocabulary words (enchant en_US dictionary)', 'Lexical Diversity'),
    'total_oov_count': ('Total OOV Count', 'Count of total out-of-vocabulary tokens (enchant en_US dictionary)', 'Lexical Diversity'),
    'unique_non_oov_count': ('Unique Non-OOV Count', 'Count of unique in-vocabulary words (enchant en_US dictionary)', 'Lexical Diversity'),
    'total_non_oov_count': ('Total Non-OOV Count', 'Count of total in-vocabulary tokens (enchant en_US dictionary)', 'Lexical Diversity'),
    'conjunction_total_count': ('Conjunction Total Count', 'Collective count of conjunctions', 'Discourse Markers'),
    'pronoun_total_count': ('Pronoun Total Count', 'Collective count of personal pronouns', 'Discourse Markers'),
    'pronoun_third_person_count': ('3rd-Person Pronoun Count', 'Collective count of he/she/they variants (no I/you)', 'Discourse Markers'),
    'auxiliary_total_count': ('Auxiliary Total Count', 'Collective count of auxiliary verbs', 'Discourse Markers'),
}

for metric_key in SIMPLE_COLOR_PATTERN_MAP:
    color_name = metric_key.replace('color_', '').replace('_count', '').capitalize()
    DYNAMIC_METRIC_CONFIGS[metric_key] = (f'{color_name} Count', f'Count of {color_name.lower()} color terms and simple variants', 'Discourse Markers')
for metric_key in INTERROGATIVE_FORM_MAP:
    label = metric_key.replace('_count', '').capitalize()
    DYNAMIC_METRIC_CONFIGS[metric_key] = (f'{label} Count', f'Count of interrogative "{label.lower()}" and variants', 'Discourse Markers')
for metric_key in CONJUNCTION_FORM_MAP:
    conj = metric_key.replace('conjunction_', '').replace('_count', '')
    DYNAMIC_METRIC_CONFIGS[metric_key] = (f'Conjunction {conj.upper()} Count', f'Count of conjunction "{conj}"', 'Discourse Markers')
for metric_key in PRONOUN_GROUP_FORM_MAP:
    pron = metric_key.replace('pronoun_', '').replace('_count', '')
    DYNAMIC_METRIC_CONFIGS[metric_key] = (f'Pronoun {pron.capitalize()} Count', f'Count of pronoun group "{pron}" and variants', 'Discourse Markers')
for metric_key in AUXILIARY_GROUP_FORM_MAP:
    aux = metric_key.replace('auxiliary_', '').replace('_count', '')
    DYNAMIC_METRIC_CONFIGS[metric_key] = (f'Auxiliary {aux.capitalize()} Count', f'Count of auxiliary "{aux}" group and variants', 'Discourse Markers')

def _install_spacy_model(model_name: str = "en_core_web_sm"):
    try:
        import subprocess, sys
        print(f"Installing spacy and {model_name} model...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "spacy"])
        print(f"Downloading {model_name} model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name, "-q"])
        print("spaCy installation complete!")
        global spacy_nlp
        import spacy
        spacy_nlp = spacy.load(model_name)
    except Exception as e:
        print(f"Failed to install spaCy: {e}")
        print(f"You can manually install with: pip install spacy && python -m spacy download {model_name}")

METRICS_VERSION = 3
FIGHTING_WORDS_PEAK_METRIC = 'fighting_words_peak_z1'
FIGHTING_WORDS_FLOOR_METRIC = 'fighting_words_floor_z1000'
FIGHTING_WORDS_PEAK_CENTROID_METRIC = 'fighting_words_peak_z1_centroid'
FIGHTING_WORDS_FLOOR_CENTROID_METRIC = 'fighting_words_floor_z1000_centroid'
FIGHTING_WORDS_PEAK_RANK = 1
FIGHTING_WORDS_FLOOR_RANK = 1000
FIGHTING_WORDS_FLOOR_MIN_COUNT = 5
FIGHTING_WORDS_FLOOR_ALPHA = 0.01
UNIFIED_COLORSCALE = 'rdbu_r'

CENTROID_DUPLICATE_METRIC_NAMES = frozenset({
    'MTLD_centroid',
    'MATTR_centroid',
    'vader_positivity_centroid',
    'vader_volatility_centroid',
    FIGHTING_WORDS_PEAK_CENTROID_METRIC,
    FIGHTING_WORDS_FLOOR_CENTROID_METRIC,
})

CENTROID_ONLY_METRIC_NAMES = frozenset({
    'burrows_cosine_disagreement',
    'avg_words_per_video',
})

CENTROID_METRIC_NAMES = CENTROID_DUPLICATE_METRIC_NAMES | CENTROID_ONLY_METRIC_NAMES

class MetricConfig:
    METRICS = {
        'TTR': {'name': 'Type-Token Ratio', 'title_suffix': 'Type-Token Ratio (TTR)', 'compute_method': '_compute_TTR', 'category': 'Core Linguistic Metrics'},
        FIGHTING_WORDS_PEAK_METRIC: {'name': 'Fighting Words Peak (Top 1 Z)', 'title_suffix': 'Monroe log-odds z-score at rank 1', 'compute_method': '_compute_fighting_words_peak_placeholder', 'category': 'Core Linguistic Metrics'},
        FIGHTING_WORDS_FLOOR_METRIC: {'name': 'Fighting Words Floor (Top 1000 Z)', 'title_suffix': 'Monroe log-odds z-score floor at rank 1000', 'compute_method': '_compute_fighting_words_floor_placeholder', 'category': 'Core Linguistic Metrics'},
        FIGHTING_WORDS_PEAK_CENTROID_METRIC: {'name': 'Fighting Words Peak (Centroid)', 'title_suffix': 'Monroe log-odds z-score at rank 1 (centroid tokens)', 'compute_method': '_compute_fighting_words_peak_centroid_placeholder', 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Core Linguistic Metrics'},
        FIGHTING_WORDS_FLOOR_CENTROID_METRIC: {'name': 'Fighting Words Floor (Centroid)', 'title_suffix': 'Monroe log-odds z-score floor at rank 1000 (centroid tokens)', 'compute_method': '_compute_fighting_words_floor_centroid_placeholder', 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Core Linguistic Metrics'},
        'byte_count': {'name': 'Byte Count', 'title_suffix': 'Total Bytes for Text Word Set', 'compute_method': '_compute_byte_count', 'category': 'Core Linguistic Metrics'},
        'normalized_compression_ratio': {'name': 'Normalized Compression Ratio', 'title_suffix': 'LZMA Compression Ratio (normalized to 4.5 bytes/word baseline)', 'compute_method': '_compute_normalized_compression_ratio', 'category': 'Core Linguistic Metrics'},
        'window_normalized_lzma_cr': {'name': 'Window Normalized LZMA CR', 'title_suffix': 'Window LZMA Compression Ratio (5000-word chunks, normalized)', 'compute_method': '_compute_window_normalized_lzma_cr', 'category': 'Core Linguistic Metrics'},
        'compression': {'name': 'Compression Ratio', 'title_suffix': 'LZMA Compression Ratio on full word set', 'compute_method': '_compute_lzma_compression_ratio', 'category': 'Core Linguistic Metrics'},
        'moving_lzma_cr': {'name': 'Window LZMA CR', 'title_suffix': 'Window LZMA Compression Ratio (5000-word chunks)', 'compute_method': '_compute_moving_lzma_compression_ratio', 'category': 'Core Linguistic Metrics'},
        'ngram_entropy_2': {'name': '2-gram Cross-Entropy', 'title_suffix': '2-gram Cross-Entropy (Bits)', 'compute_method': '_compute_ngram_entropy_2', 'requires_ngram_entropy': True, 'category': 'Core Linguistic Metrics'},
        'ngram_entropy_3': {'name': '3-gram Cross-Entropy', 'title_suffix': '3-gram Cross-Entropy (Bits)', 'compute_method': '_compute_ngram_entropy_3', 'requires_ngram_entropy': True, 'category': 'Core Linguistic Metrics'},
        'MTLD': {'name': 'MTLD', 'title_suffix': 'MTLD (Measure of Textual Lexical Diversity)', 'compute_method': '_compute_MTLD_score', 'category': 'Core Linguistic Metrics'},
        'MTLD_centroid': {'name': 'MTLD (Centroid)', 'title_suffix': 'MTLD on centroid tokens', 'compute_method': '_compute_MTLD_score', 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Core Linguistic Metrics'},
        'MATTR': {'name': 'MATTR', 'title_suffix': 'MATTR (Moving Average Type-Token Ratio, window=25)', 'compute_method': '_compute_MATTR', 'category': 'Core Linguistic Metrics'},
        'MATTR_centroid': {'name': 'MATTR (Centroid)', 'title_suffix': 'MATTR on centroid tokens', 'compute_method': '_compute_MATTR', 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Core Linguistic Metrics'},
        'avg_words_per_video': {'name': 'Average Words per Video', 'title_suffix': 'Average words per centroid video', 'compute_method': '_compute_avg_words_per_video', 'centroid_metric': True, 'video_centroid_only': True, 'category': 'Core Linguistic Metrics'},
        'dev_specialty_count': {'name': 'Dev\'s Specialty Markers', 'title_suffix': 'count of "this is a"', 'compute_method': '_compute_dev_specialty_count', 'category': 'Core Linguistic Metrics'},
        'custom_marker_count': {'name': 'Smore markers', 'title_suffix': 'count of I think, probably, maybe, I feel like', 'compute_method': '_compute_custom_marker_count', 'category': 'Core Linguistic Metrics'},
        'smore_markers_2_count': {'name': 'Smore markers 2', 'title_suffix': 'count of also and too', 'compute_method': '_compute_smore_markers_2_count', 'category': 'Core Linguistic Metrics'},
        'smore_markers_3_count': {'name': 'Smore markers 3', 'title_suffix': 'count of words containing root "hypo"', 'compute_method': '_compute_smore_markers_3_count', 'category': 'Core Linguistic Metrics'},
        'pronoun_switching': {'name': 'Pronoun Switching', 'title_suffix': 'I/You/We Switching Frequency', 'compute_method': '_compute_pronoun_volatility', 'category': 'Core Linguistic Metrics'},

        'vader_positivity': {'name': 'VADER Positivity Ratio', 'title_suffix': 'VADER Positivity Ratio', 'compute_method': 'compute_vader_sentiment', 'result_index': 1, 'category': 'Sentiment & Tone'},
        'vader_volatility': {'name': 'VADER Sentiment Volatility', 'title_suffix': 'VADER Sentiment Volatility', 'compute_method': 'compute_vader_sentiment', 'result_index': 2, 'category': 'Sentiment & Tone'},
        'vader_positivity_centroid': {'name': 'VADER Positivity Ratio (Centroid)', 'title_suffix': 'VADER Positivity Ratio on centroid tokens', 'compute_method': 'compute_vader_sentiment', 'result_index': 1, 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Sentiment & Tone'},
        'vader_volatility_centroid': {'name': 'VADER Sentiment Volatility (Centroid)', 'title_suffix': 'VADER Sentiment Volatility on centroid tokens', 'compute_method': 'compute_vader_sentiment', 'result_index': 2, 'centroid_metric': True, 'centroid_duplicate': True, 'category': 'Sentiment & Tone'},
        'perplexity': {'name': 'Perplexity', 'title_suffix': 'GPT2 Perplexity', 'compute_method': 'calculate_perplexity', 'requires_perplexity_model': True, 'category': 'Sentiment & Tone'},

        'yules_k': {'name': "Yule's K", 'title_suffix': "Yule's K Statistic", 'compute_method': '_compute_yules_k', 'category': 'Lexical Diversity'},
        'avg_word_length': {'name': 'Average Word Length', 'title_suffix': 'Average Word Length (characters)', 'compute_method': '_compute_avg_word_length', 'category': 'Lexical Diversity'},
        'word_burstiness': {'name': 'Word Burstiness', 'title_suffix': 'Word Burstiness (Gini Coefficient)', 'compute_method': '_compute_word_burstiness', 'category': 'Lexical Diversity'},
        'semantic_disparity': {'name': 'Semantic Disparity', 'title_suffix': 'Semantic Disparity (Word Meaning Spread)', 'compute_method': '_compute_semantic_disparity', 'requires_spacy_model': True, 'category': 'Lexical Diversity'},
        'burrows_cosine_disagreement': {'name': 'Burrows/Cosine Disagreement', 'title_suffix': 'Burrows/Cosine Stylistic-Semantic Disagreement (Centroid)', 'compute_method': '_compute_burrows_cosine_disagreement', 'centroid_metric': True, 'centroid_only': True, 'category': 'Cross-Modal Similarity'},
        'unique_bigrams': {'name': 'Unique Bigrams', 'title_suffix': 'Unique Bigram Count', 'compute_method': '_compute_unique_bigrams', 'category': 'Lexical Diversity'},
        'moving_unique_bigrams': {'name': 'Window Unique Bigrams', 'title_suffix': 'Avg Unique Bigrams per 5k Chunk', 'compute_method': '_compute_moving_unique_bigrams', 'category': 'Lexical Diversity'},
        'unique_trigrams': {'name': 'Unique Trigrams', 'title_suffix': 'Unique Trigram Count', 'compute_method': '_compute_unique_trigrams', 'category': 'Lexical Diversity'},
        'moving_unique_trigrams': {'name': 'Window Unique Trigrams', 'title_suffix': 'Avg Unique Trigrams per 5k Chunk', 'compute_method': '_compute_moving_unique_trigrams', 'category': 'Lexical Diversity'},
        'lexical_density': {'name': 'Lexical Density', 'title_suffix': 'Lexical Density (Content Words)', 'compute_method': '_compute_lexical_density', 'requires_spacy_model': True, 'category': 'Lexical Diversity'},
        'hapax_ratio': {'name': 'Hapax Legomena Ratio', 'title_suffix': 'Hapax Legomena Ratio (Words Used Once)', 'compute_method': '_compute_hapax_ratio', 'category': 'Lexical Diversity'},
        'dis_ratio': {'name': 'Dis Legomena Ratio', 'title_suffix': 'Dis Legomena Ratio (Words Used Twice)', 'compute_method': '_compute_dis_ratio', 'category': 'Lexical Diversity'},

        'self_mention_count': {'name': 'Self-Mention Count', 'title_suffix': 'First-Person Pronoun Count', 'compute_method': '_compute_self_mention', 'category': 'Syntactic Patterns'},
        'hedge_count': {'name': 'Hedge Word Count', 'title_suffix': 'Hedge Words (maybe, perhaps, probably, etc.)', 'compute_method': '_compute_hedge_count', 'category': 'Syntactic Patterns'},
        'subordinate_clause_count': {'name': 'Subordinate Clause Count', 'title_suffix': 'Subordinating Conjunctions', 'compute_method': '_compute_subordinate_clause_count', 'category': 'Syntactic Patterns'},
        'coordinate_clause_count': {'name': 'Coordinate Clause Count', 'title_suffix': 'Coordinating Conjunctions', 'compute_method': '_compute_coordinate_clause_count', 'category': 'Syntactic Patterns'},
        'latinate_word_ratio': {'name': 'Latinate Word Ratio', 'title_suffix': 'Latinate/Greek Morphology Ratio', 'compute_method': '_compute_latinate_word_ratio', 'category': 'Syntactic Patterns'},

        'insult': {'name': '"Good" Insult Count', 'title_suffix': 'Insult Words (stupid, idiot, fool, punk, etc.)', 'compute_method': 'compute_insult_density', 'category': 'Profanity & Insults'},

        'family_exclamations': {'name': 'Family Exclamation Count', 'title_suffix': 'Family-Friendly Exclamation Count', 'compute_method': '_compute_family_exclamations', 'category': 'Discourse Markers'},
        'vulnerability': {'name': 'Vulnerability Count', 'title_suffix': 'Vulnerability Words/Phrases Count', 'compute_method': '_compute_vulnerability_count', 'category': 'Discourse Markers'},
        'simile_count': {'name': 'Simile Count', 'title_suffix': 'Simile and Comparative Expression Count', 'compute_method': '_compute_simile_count', 'category': 'Discourse Markers'},
        'color_count': {'name': 'Color Mention Count', 'title_suffix': 'Color Word and Shade Mention Count', 'compute_method': '_compute_color_count', 'category': 'Discourse Markers'},
        'discussion_commentary_count': {'name': 'Discussion Commentary Count', 'title_suffix': 'Discussion/Dialogue Marker Count', 'compute_method': '_compute_discussion_commentary_count', 'category': 'Discourse Markers'},
        'article_count': {'name': 'Article Count (skip what+the)', 'title_suffix': 'Article Count excluding "what the" bigrams', 'compute_method': '_compute_article_count', 'category': 'Discourse Markers'},
        'article_count_raw': {'name': 'Article Count (raw)', 'title_suffix': 'Article Count including all articles', 'compute_method': '_compute_article_count_raw', 'category': 'Discourse Markers'},
        'demonstrative_count': {'name': 'Demonstrative Count', 'title_suffix': 'Demonstrative Count (this, that, these, those, there)', 'compute_method': '_compute_demonstrative_count', 'category': 'Discourse Markers'},
        'possessive_count': {'name': 'Possessive Count', 'title_suffix': 'Possessive Determiner/Genitive Count', 'compute_method': '_compute_possessive_count', 'category': 'Discourse Markers'},
        'quantifier_count': {'name': 'Quantifier Count', 'title_suffix': 'Quantifier Count (some, many, few, several, etc.)', 'compute_method': '_compute_quantifier_count', 'category': 'Discourse Markers'},
        'gerund_count': {'name': 'Gerund Count', 'title_suffix': 'Gerund/Present Participle Count (spaCy VBG)', 'compute_method': '_compute_gerund_count', 'category': 'Discourse Markers'},
        'pronoun_article_ratio': {'name': 'Pronoun-to-Article Ratio', 'title_suffix': 'Pronoun/Article Ratio', 'compute_method': '_compute_pronoun_article_ratio', 'category': 'Discourse Markers'},
        'imperative_exclamation_density': {'name': 'Imperative/Exclamations', 'title_suffix': 'Direct Commands & Emotional Outbursts Count', 'compute_method': '_compute_imperative_exclamation_density', 'requires_spacy_model': True, 'category': 'Discourse Markers'},
        'deictic_spatial_temporal': {'name': 'Spatial/Temporal', 'title_suffix': 'Explicit Pointing Language (per 1000 words)', 'compute_method': '_compute_deictic_spatial_temporal', 'category': 'Discourse Markers'},
        'elaboration_explanation_ratio': {'name': 'Elaboration & Explanation Ratio', 'title_suffix': 'Elaboration & Explanation (Narrative Structure)', 'compute_method': '_compute_elaboration_explanation_ratio', 'category': 'Discourse Markers'},
        'narration_continuation_ratio': {'name': 'Narration & Continuation Ratio', 'title_suffix': 'Narration & Continuation (Action-Based)', 'compute_method': '_compute_narration_continuation_ratio', 'category': 'Discourse Markers'},
    }

    CENTROID_DUPLICATE_METRICS = CENTROID_DUPLICATE_METRIC_NAMES
    CENTROID_ONLY_METRICS = CENTROID_ONLY_METRIC_NAMES
    CENTROID_METRICS = CENTROID_METRIC_NAMES

    for metric_key, metric_cfg in list(METRICS.items()):
        if not _is_word_count_metric_config(metric_key, metric_cfg):
            continue
        DYNAMIC_METRIC_CONFIGS.setdefault(metric_key, (metric_cfg.get('name', metric_key), metric_cfg.get('title_suffix', metric_key), metric_cfg.get('category', 'Silly & Fun Counts')))
        DYNAMIC_COUNT_FALLBACK_METHODS.setdefault(metric_key, metric_cfg.get('compute_method', ''))

    for metric_key, (metric_name, title_suffix, category) in DYNAMIC_METRIC_CONFIGS.items():
        METRICS[metric_key] = {'name': metric_name, 'title_suffix': title_suffix, 'compute_method': '_compute_dynamic_token_count', 'category': category}

    del metric_key, metric_cfg, metric_name, title_suffix, category

    @classmethod
    def get_metric_names(cls):
        return list(cls.METRICS.keys())

    @classmethod
    def requires_perplexity_model(cls, metric_name):
        return cls.METRICS.get(metric_name, {}).get('requires_perplexity_model', False)

    @classmethod
    def requires_spacy_model(cls, metric_name):
        return cls.METRICS.get(metric_name, {}).get('requires_spacy_model', False)

    @classmethod
    def requires_ngram_entropy(cls, metric_name):
        return cls.METRICS.get(metric_name, {}).get('requires_ngram_entropy', False)

    @classmethod
    def is_centroid_metric(cls, metric_name: str) -> bool:
        return metric_name in cls.CENTROID_METRICS or bool(
            cls.METRICS.get(metric_name, {}).get('centroid_metric', False)
        )

    @classmethod
    def is_centroid_duplicate(cls, metric_name: str) -> bool:
        return metric_name in cls.CENTROID_DUPLICATE_METRICS or bool(
            cls.METRICS.get(metric_name, {}).get('centroid_duplicate', False)
        )

    @classmethod
    def is_video_centroid_only(cls, metric_name: str) -> bool:
        return bool(cls.METRICS.get(metric_name, {}).get('video_centroid_only', False))

    FAST_MODE_EXCLUDED = {'ngram_entropy_2', 'ngram_entropy_3'}

    @classmethod
    def is_metric_eligible(cls, metric_name: str, enable_spacy: bool, needs_perplexity_model: bool, enable_ngram_entropy: bool, fast_mode: bool) -> bool:
        if metric_name not in cls.METRICS:
            return False
        cfg = cls.METRICS[metric_name]
        if cfg.get('requires_spacy_model') and not enable_spacy:
            return False
        if cfg.get('requires_perplexity_model') and not needs_perplexity_model:
            return False
        if cfg.get('requires_ngram_entropy') and not enable_ngram_entropy:
            return False
        if fast_mode and metric_name in cls.FAST_MODE_EXCLUDED:
            return False
        return True

def hms_to_seconds(hms: str) -> float:
    if not hms:
        return 0
    parts = list(map(float, hms.split(':')))
    multipliers = [1, 60, 3600]
    return sum(p * multipliers[len(parts) - 1 - i] for i, p in enumerate(parts))

def extract_video_id(filename: str) -> str:
    match = re.search(r'\[([A-Za-z0-9_-]{11})\]', filename)
    return match.group(1) if match else filename.replace('.txt', '').split('.')[-1]

def load_ignore_urls(ignore_file: Optional[str]) -> set:
    if not ignore_file:
        return set()
    ignore_path = Path(ignore_file)
    if not ignore_path.exists():
        print(f"Warning: Ignore file not found: {ignore_file}")
        return set()
    ignore_ids = set()
    try:
        with open(ignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if 'youtu' in line.lower():
                    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([a-zA-Z0-9_-]{11})', line)
                    if match:
                        ignore_ids.add(match.group(1))
                else:
                    vid_id = line.replace('.txt', '').replace('.json', '')[:11]
                    if len(vid_id) >= 10:
                        ignore_ids.add(vid_id)
    except Exception as e:
        print(f"Warning: Error reading ignore file: {e}")
    if ignore_ids:
        print(f"Loaded {len(ignore_ids)} video IDs to ignore")
    return ignore_ids