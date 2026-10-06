"""atlas_gaming.py

!Purposely informal 
gaming-specific word count metrics.

These metrics are only registered into ``MetricConfig.METRICS`` when the
user runs with ``--niche-words gaming``.  Future niche files (educational,
etc.) can follow the same pattern: expose a ``register_<niche>_metrics``
function that mutates ``MetricConfig.METRICS`` and receives the classifier
class to extend.
Metrics show details about a text, 
texts are given value depending on a metric (like 42 for MTLD score).
texts are ranked against each other on a leaderboard.

Design notes
------------
* All regex patterns live at module level so that ``inspect.getsource``
  inside ``ChannelClassifier.get_word_count_metric_names`` still finds
  the token ``PATTERNS`` in the source of the wrapped mixin methods.
* The mixin only defines *compute* methods.  Shared helpers such as
  ``_compute_pattern_count``, ``_get_runtime_text`` etc. are supplied
  by ``ChannelClassifier`` via the MRO.
* ``register_gaming_metrics`` mutates the existing metric registry and adds
    compute methods to the classifier class supplied by the caller.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

UNBELIEVABLY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bunbelievably\b']]
FART_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bfart\w*\b', r'\bflatul\w*\b', r'\btoot(?:s|ed|ing)?\b']]
GOD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgod\b']]
BABY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bbaby\w*\b', r'\binfant\w*\b', r'\bnewborns?\b', r'\btoddler\w*\b',
    r'\bchild(?:ren)?\b', r'\bkid(?:s)?\b', r'\bdiaper\w*\b', r'\bcrib\w*\b',
    r'\bpacifier\w*\b', r'\bteething\b'
]]
DING_DONG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bding\s*-?\s*dong(?:s)?\b']]
NOSTRIL_HUMOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bnose\b', r'\bnoses\b', r'\bnostril\w*\b', r'\bbooger\w*\b', r'\bsnot\w*\b']]
SHUT_UP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bshut(?:ting)?\s*-?\s*up\b']]
READY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bready\b']]
YOU_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\byou\b']]
I_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bi\b']]
GO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgo\b']]
BOOGER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbooger\w*\b']]
FREAK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bfreak\w*\b']]
OOH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\boo+h+\b']]
YALL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\by'all\b", r"\byall\b"]]
YOU_GUYS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\byou guys\b"]]
YONDER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\byonder\b"]]
SMALL_WORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\blittle\w*\b', r'\bshort\w*\b', r'\bsmall\w*\b', r'\btiny\w*\b', r'\bteeny\w*\b']]
BIG_WORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbig\w*\b', r'\btall\w*\b', r'\bgiant\w*\b', r'\bginormous\w*\b', r'\bgigantic\w*\b']]
HANDSOME_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhandsome\w*\b']]
EASY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\beasy\w*\b']]
HARD_DIFFICULT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhard\w*\b', r'\bdifficult\w*\b']]
SMORES_TASTY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsmores\b', r'\btasty\b']]
ALRIGHT_THEN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\balright\s+then\b', r'\balrighty\s+then\b']]
ALSO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\balso\b']]
DANG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdang\b']]
DAMN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdamn\b']]
DARN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdarn\b']]
ANIMAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bcapybara(?:s)?\b', r'\bshark(?:s)?\b', r'\baxolotl(?:s)?\b', r'\bdog(?:s)?\b', r'\bcat(?:s)?\b',
    r'\bfrog(?:s)?\b', r'\bchicken(?:s)?\b', r'\bbird(?:s)?\b', r'\bduck(?:s)?\b', r'\bgoose(?:s)?\b',
    r'\bpig(?:s)?\b', r'\bcow(?:s)?\b', r'\bhorse(?:s)?\b', r'\bgoat(?:s)?\b', r'\bsheep\b',
    r'\brabbit(?:s)?\b', r'\bbunny(?:s)?\b', r'\bhamster(?:s)?\b', r'\bmouse\b', r'\bmice\b',
    r'\bdeer\b', r'\bfox(?:es)?\b', r'\bwolf(?:s)?\b', r'\bbear(?:s)?\b', r'\btiger(?:s)?\b',
    r'\blion(?:s)?\b', r'\belephant(?:s)?\b', r'\bgiraffe(?:s)?\b', r'\bmonkey(?:s)?\b', r'\bape(?:s)?\b',
    r'\bkoala(?:s)?\b', r'\bpanda(?:s)?\b', r'\botter(?:s)?\b', r'\bsloth(?:s)?\b', r'\bhedgehog(?:s)?\b',
    r'\blizard(?:s)?\b', r'\bsnake(?:s)?\b', r'\bturtle(?:s)?\b', r'\bwhale(?:s)?\b', r'\bdolphin(?:s)?\b',
    r'\bseal(?:s)?\b', r'\bcrab(?:s)?\b', r'\bshrimp\b', r'\boctopus(?:es)?\b', r'\bsquid\b', r'\bfish\b'
]]
MAN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bman\b']]
GOODNESS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgoodness\b']]
RAGDOLL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bragdoll\w*\b', r'\brag doll\w*\b']]
WOAH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bwoa+h?\w*\b', r'\bwhoops\b', r'\bwhoa+h?\w*\b']]
OH_BOY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\boh boy\b', r'\boh man\b']]
WEIRD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bweird\b', r'\bbizarre\b', r'\bstrange\b', r'\bwacky\b', r'\bodd\b',
    r'\bpeculiar\b', r'\bquirky\b', r'\bunusual\b', r'\bfunky\b'
]]
BRIDGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbridge\w*\b']]
ENGINEER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bengineer\w*\b']]
SUBSCRIBE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsubscribe\w*\b']]
CHILL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bchill\w*\b']]
VOLCANO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bvolcano\w*\b']]
EDITOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\beditor\w*\b']]
TEA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\btea\w*\b']]
LETS_GO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\blet'?s\s+go\b"]]
BLITZ_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bblitz\w*\b']]
ALL_RIGHT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\ball\s+right\b']]
WA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bwa\b']]
BRO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbro\b']]
PUNK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?<!cyber)(?<!cyber\s)(?<!cm)(?<!cm\s)punk(?:s)?\b(?!\s+(?:music|rock|disc|discs|record|album|band|song|genre))']]
MR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bm(?:r|rs)\b']]
GAY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgay\w*\b', r'\bgayz\w*\b']]
PASTA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bpasta\w*\b']]
PLEASE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bplease\w*\b', r'\bpls\w*\b', r'\bplz\w*\b']]
BAGEL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbagel\w*\b']]
VIOLATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bviolat\w*\b']]
GOING_TO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgoing\s+to\b']]
DOUBLE_MODAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:may|might|must)\s+(?:could|should|would)\b']]
TRIPLE_AUXILIARY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:can|could|may|might|must|shall|should|will|would)\s+(?:have|'ve)\s+(?:been|had)\b",
    r'\b(?:can|could|may|might|must|shall|should|will|would)\s+be\s+being\b',
    r'\b(?:has|have|had)\s+been\s+being\b'
]]
MODAL_PERFECT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\b(?:can|could|may|might|must|shall|should|will|would)\s+(?:have|['’]ve)\b"]]
MODAL_PROGRESSIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:can|could|may|might|must|shall|should|will|would)\s+be\s+\w+ing\b']]
MODAL_PASSIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:can|could|may|might|must|shall|should|will|would)\s+be\s+(?:\w+(?:ed|en|wn|lt|pt|ft)|done|made|seen|given|known|taken|built|bought|brought|caught|told|sold|left|felt|found)\b']]
SEMI_MODAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:have|has|had)\s+to\b', r'\bneed(?:s|ed)?\s+to\b', r'\bgot\s+to\b', r'\bought\s+to\b', r'\bgonna\b']]
MODAL_NEGATION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\b(?:can(?:not|['’]t)|won['’]t|shan['’]t|(?:could|would|should|might|may|must|shall)\s+not|(?:could|would|should|might|must|shall)n['’]t)\b"]]
CONDITIONAL_MODAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bif\b(?:\s+\w+){0,6}\s+(?:would|could|might|should|may|can|will)\b']]
BE_GONNA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:am|is|are|was|were|i['’]m|you['’]re|we['’]re|they['’]re|he['’]s|she['’]s|it['’]s)\s+gonna\b",
    r"\b(?:am|is|are|was|were|i['’]m|you['’]re|we['’]re|they['’]re|he['’]s|she['’]s|it['’]s)\s+going\s+to\b"
]]
BE_GOTTA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:have|has|had)\s+got\s+to\b', r'\bgotta\b']]
STANCE_MARKER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bi\s+think\b", r"\bi\s+guess\b", r"\bi\s+mean\b", r"\byou\s+know\b", r"\bto\s+be\s+honest\b"]]
EVIDENTIAL_MARKER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bapparently\b', r'\bit\s+seems\b', r'\bseems\s+like\b', r'\blooks\s+like\b', r'\bsounds\s+like\b', r'\bappears\s+to\b']]
INTENSIFIER_STACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:so|very|really|super|extremely|totally|absolutely)\s+(?:so|very|really|super|extremely|totally|absolutely)\b']]
HEDGE_STACK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:maybe|perhaps|probably|possibly|kinda|kind\s+of|sort\s+of|i\s+guess|i\s+think)\s+(?:maybe|perhaps|probably|possibly|kinda|kind\s+of|sort\s+of|i\s+guess|i\s+think)\b']]
TAG_QUESTION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\bright\b", r"\bisn['’]t\s+it\b", r"\bdon['’]t\s+you\b",
    r"\b(?:aren['’]t|wasn['’]t|weren['’]t|won['’]t|wouldn['’]t|couldn['’]t|shouldn['’]t)\s+(?:i|you|we|they|it|he|she)\b"
]]
REPAIR_RESTART_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bi\s+i\b", r"\bi\s+mean\b", r"\bor\s+rather\b", r"\bwell\s+no\b", r"\bno\s+i\s+mean\b"]]
DISFLUENCY_CLUSTER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\b(?:uh|um|uhm|er|erm|ah)\s+(?:uh|um|uhm|er|erm|ah)\b",
    r"\b(?:uh|um|uhm|er|erm|ah)(?:\s+(?:uh|um|uhm|er|erm|ah)){2,}\b"
]]
OVER_TO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bover\s+to\b']]
BROUGHT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbrought\b']]
TOOK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\btook\b']]
WELL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bwell\b']]
RECKON_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\breckon\w*\b']]
I_DONT_KNOW_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bi\s+don't\s+know\b"]]
YOU_KNOW_WHAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\byou\s+know\s+what\b"]]
OH_MY_GOD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\boh\s+my\s+(?:\w+\s+)*?god\b"]]
OH_MY_GOSH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\boh\s+my\s+(?:\w+\s+)*?gosh\b"]]
OH_MY_GOODNESS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\boh\s+my\s+(?:\w+\s+)*?goodness\b"]]
THERE_WE_GO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bthere\s+we\s+go\b"]]
HERE_WE_GO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bhere\s+we\s+go\b"]]
LOOK_AT_THAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\blook\s+at\s+that\b"]]
OH_NO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\boh\s+no\b"]]
IM_NOT_A_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bi'm\s+not\s+a\b", r"\byou're\s+not\s+a\b", r"\bi\s+am\s+not\s+a\b", r"\byou\s+are\s+not\s+a\b"]]
GIVE_ME_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bgive\s+me\b"]]
HATE_YOU_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bhate\s+you\b"]]
POOKIE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bpooki\w*\b', r'\bpookie\w*\b', r'\bpooky\w*\b']]
SKILL_ISSUE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bskill\s+issue\w*\b"]]
SWEAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsweat\w*\b']]
RAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\brat\w*\b']]
CRINGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcringe\w*\b', r'\bcringy\w*\b', r'\bcringey\w*\b']]
DAMAGE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdamage\w*\b']]
SECRET_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bsea\s*-?\s*top\w*\b', r'\bseatop\w*\b', r'\bctop\w*\b', r'\bc\s*-?\s*top\w*\b',
    r'\bseat\s*-?\s*top\w*\b', r'\bseattop\w*\b', r'\bsetop\w*\b', r'\bsee\s*-?\s*top\w*\b'
]]
INSULT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bstupid\w*\b', r'\bdumb\w*\b', r'\bidiot\w*\b', r'\bnerd\w*\b', r'\bdork\w*\b', r'\bdummy\w*\b',
    r'\bjerks?\b', r'\bdingle\w*\b', r'\bdoofus\w*\b', r'\bcoward\w*\b', r'\bdummies\w*\b',
    r'\bbuffoon\w*\b', r'\blame\w*\b', r'\bgoofball\w*\b', r'\bnoob\w*\b', r'\b(?:these|this|that|another|a|the)\s+fools?\b',
    r'\bnincompoop\w*\b', r'\bcretin\w*\b', r'\bmoron\w*\b', r'\b(?<!the )loser\w*\b', r'\brascal\w*\b', r'\bhooligan\w*\b',
    r"\byou(?:\s+\w+)?\s+fool\b", r"\byou(?:\s+\w+)?\s+prick\b", r'\bskank\w*\b', r'\bugly\b',
    r'\b(?<!cyber)(?<!cyber\s)(?<!cm)(?<!cm\s)punk(?:s)?\b(?!\s+(?:music|rock|disc|discs|record|album|band|song|genre))'
]]
BUDDY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbud\b', r'\bbuddy\w*\b', r'\bbuddies\b']]
HOMIE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhomie\w*\b', r'\bhomies\b']]
MURDER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bmurder\w*\b']]
NONCHALANT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bnonchalan\w*\b']]
DUDE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdude\w*\b']]
CHEEKY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcheeky\w*\b']]
SPANNER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bspanner\w*\b']]
WRENCH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bwrench\w*\b']]
RUBBISH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\brubbish\b']]
BURGER_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcheeseburger\w*\b', r'\bhamburger\w*\b', r'\bburger\w*\b']]
BYE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbye\w*\b', r'\bgoodbye\w*\b']]
EXPLOSION_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bexplod\w*\b', r'\bexplosion\w*\b', r'\bexplosive\w*\b']]
I_THINK_PATTERNS = [re.compile(r'\bi\s+think\b', re.IGNORECASE)]
PROBABLY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bprobably\w*\b']]
CRAZY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcrazy\w*\b']]
SURE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsure\w*\b']]
SNEAKY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsneaky\w*\b']]
SEE_YOU_NEXT_TIME_PATTERNS = [re.compile(r'\bsee\s+you\s+next\s+time\b', re.IGNORECASE)]
OW_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bow\b', r'\boww+\b']]
DNF_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdnf\w*\b']]
CHECKPOINT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcheckpoint\w*\b']]
MATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bmate\w*\b']]
BLOODY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbloody\w*\b']]
LOO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bloo\b', r'\bloos\b']]
FUMBLE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bfumbl\w+\b']]
OUTRAGEOUS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\boutrageous\w*\b']]
EMBARRASS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bembarrass\w*\b']]
WELL_DONE_PATTERNS = [re.compile(r'\bwell\s+done\b', re.IGNORECASE)]
HANG_ON_PATTERNS = [re.compile(r'\bhang\s+on\b', re.IGNORECASE)]
HOLD_ON_PATTERNS = [re.compile(r'\bhold\s+on\b', re.IGNORECASE)]
LOCK_IN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\block\s+in\b', r'\blocking\s+in\b', r'\blocked\s+in\b']]
TO_BE_FAIR_PATTERNS = [re.compile(r'\bto\s+be\s+fair\b', re.IGNORECASE)]
REAL_QUICK_PATTERNS = [re.compile(r'\breal\s+quick\b', re.IGNORECASE)]
FINISH_LINE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bfinish\s+line\w*\b']]
TO_BE_HONEST_PATTERNS = [re.compile(r'\bto\s+be\s+honest\b', re.IGNORECASE)]
TOO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\btoo\b']]
STOP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bstop\w*\b']]
WAIT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bwait\w*\b']]
WHAT_DO_YOU_MEAN_PATTERNS = [re.compile(r'\bwhat\s+do\s+you\s+mean\b', re.IGNORECASE)]
SON_OF_A_PATTERNS = [re.compile(r'\bson\s+of\s+a\b', re.IGNORECASE)]
SON_OF_A_NOT_BEFORE_BLOCK_PATTERNS = [re.compile(r'\bson\s+of\s+a\b(?!\s*(?:fuck\w*|f+u+c+k+\w*)\b)', re.IGNORECASE)]
ARE_YOU_KIDDING_PATTERNS = [re.compile(r'\bare\s+you\s+kidding\b', re.IGNORECASE)]
BAD_BOY_PATTERNS = [re.compile(r'\bbad\s+boy\w*\b', re.IGNORECASE)]
YOINK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\byoink\w*\b']]
HIGH_FIVE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhigh\s*-?\s*five\w*\b']]
TECHNICALLY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\btechnically\b']]
NAILED_IT_PATTERNS = [re.compile(r'\bnailed\s+it\b', re.IGNORECASE)]
JEEZ_GEEZ_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bjeez\b', r'\bgeez\b']]
TREE_WORD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bcedar\w*\b', r'\boak\w*\b', r'\bmaple\w*\b', r'\bbirch\w*\b', r'\baspen\w*\b',
    r'\bspruce\w*\b', r'\bpine\w*\b', r'\bwillow\w*\b', r'\bpoplar\w*\b', r'\belm\w*\b',
    r'\bfir\w*\b', r'\bredwood\w*\b', r'\bsequoia\w*\b', r'\bsycamore\w*\b', r'\byew\w*\b', r'\bpalm\w*\b'
]]
SO_COOL_PATTERNS = [re.compile(r'\bso\s+cool\w*\b', re.IGNORECASE)]
THANKS_FOR_WATCHING_PATTERNS = [re.compile(r'\bthanks\s+for\s+watching\b', re.IGNORECASE)]
GOO_GOOP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgoo\b', r'\bgoos\b', r'\bgoop\w*\b']]
GO_AHEAD_PATTERNS = [re.compile(r'\bgo\s+ahead\b', re.IGNORECASE)]
MAKE_SURE_PATTERNS = [re.compile(r'\bmake\s+sure\b', re.IGNORECASE)]
MAKES_ME_PATTERNS = [re.compile(r'\bmakes\s+me\b', re.IGNORECASE)]
LAME_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\blame\w*\b']]
GRANDMA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bgrandma\w*\b', r'\bgrandmother\w*\b', r'\bgrandmom\w*\b', r'\bgranny\w*\b']]
YOURE_A_LITTLE_PATTERNS = [re.compile(r"\byou(?:'re|re|\s+are)\s+(?:such\s+a\s+)?little\b", re.IGNORECASE)]
INSANE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\binsane\w*\b', r'\binsanity\w*\b']]
HYPOTHETICAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhypothetical\w*\b', r'\bhypothetically\w*\b']]
I_FEEL_PATTERNS = [re.compile(r'\bi\s+feel\b', re.IGNORECASE)]
I_SWEAR_PATTERNS = [re.compile(r'\bi\s+swear\b', re.IGNORECASE)]
GOOD_QUESTION_PATTERNS = [re.compile(r'\bgood\s+question\b', re.IGNORECASE)]
UH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\buh\b', r'\buhh+\b']]
UM_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bum\b', r'\bumm+\b', r'\buhm\b', r'\buhmm+\b']]
UPGRADE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bupgrade\w*\b']]
THATS_OKAY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bthat'?s\s+okay\b", r"\bthat\s+is\s+okay\b", r"\bthat'?s\s+fine\b", r"\bthat\s+is\s+fine\b"]]
ITS_OKAY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bit'?s\s+okay\b", r"\bit\s+is\s+okay\b", r"\bit'?s\s+ok\b", r"\bit\s+is\s+ok\b"]]
OKAY_SO_PATTERNS = [re.compile(r'\boka?y\s+so\b', re.IGNORECASE)]
HOLD_UP_PATTERNS = [re.compile(r'\bhold\s+up\b', re.IGNORECASE)]
SORRY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bsorry\w*\b']]
HEAR_ME_OUT_PATTERNS = [re.compile(r'\bhear\s+me\s+out\b', re.IGNORECASE)]
HELLO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bhello\w*\b']]
WHOLE_BUNCH_PATTERNS = [re.compile(r'\bwhole\s+bunch\b', re.IGNORECASE)]
PUT_PATTERNS = [re.compile(r'\bput\w*\b', re.IGNORECASE)]
TAKE_CARE_PATTERNS = [re.compile(r'\btake\s+care\b', re.IGNORECASE)]
I_CAN_PATTERNS = [re.compile(r'\bi\s+can\b', re.IGNORECASE)]
BOT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bbot\w*\b']]
DOG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bdog\w*\b', r'\bdoggo\w*\b', r'\bhound\w*\b', r'\bpuppy\w*\b', r'\bpuppies\b', r'\bbark\w*\b', r'\bwoof\w*\b']]
CAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcat\w*\b', r'\bkitty\w*\b', r'\bkitties\b', r'\bmeow\w*\b', r'\bpurr\w*\b']]
LESBIAN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\blesbian\w*\b']]
MONEY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bmoney\b', r'\bcash\b', r'\bdollar\w*\b', r'\bcent\w*\b', r'\bcurrency\b',
    r'\bpay\w*\b', r'\bcost\w*\b', r'\bprice\w*\b', r'\bexpensive\b', r'\bcheap\w*\b',
    r'\bbuy\w*\b', r'\bsell\w*\b', r'\bpurchase\w*\b', r'\bspend\w*\b', r'\bbudget\w*\b', r'\bfunds?\b'
]]
MOON_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bmoon\w*\b']]
FAST_FOOD_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\b(?:mcdonald\'s|burger king|kfc|taco bell|door dash|uber eat|uber eats)\w*\b']]
MAX_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bmax\w*\b']]
LOVE_YOU_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\blove\s+you\b']]
THANK_YOU_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bthank\s+you\b', r'\bthanks\b', r'\bthanky\w*\b']]
FOREIGN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bforeign\w*\b']]
HEAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bheat\w*\b']]
EXCELLENT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bexcellen\w*\b']]
CAR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bcar\b', r'\bcars\b']]
BOAT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bboat\b', r'\bboats\b']]
TRUCK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\btruck\b', r'\btrucks\b']]
PLANE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bplane\b', r'\bplanes\b']]
BLOCK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bfuck\w*\b', r'\bf+u+c+k+\w*\b']]
PATTERNS_67 = [re.compile(p, re.IGNORECASE) for p in [r'\b6[\s-]?7\b', r'\bsixty[\s-]?seven\b', r'\bsix[\s-]?seven\b']]
WAFFLESTOMP_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r"\bwaffle[\s-]?stomp\w*\b"]]
STUPID_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bstupid\w*\b', r'\bstoopid\w*\b', r'\bstoopit\w*\b']]
IDIOT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bidiot\w*\b', r'\bidoit\w*\b']]
THE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bthe\b']]
ROYGBIV_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\broy\s*g\s*b\s*i\s*v\b', r'\broy\s*g\s*b\s*i\b', r'\broy\s*g\s*b\b', r'\broy\s*g\b']]
DUCK_GOOSE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bduck(?:s)?\b', r'\bgoose(?:s)?\b', r'\bgeese\b']]
MARRY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bmarry\w*\b', r'\bmarried\b', r'\bmarriage\w*\b']]
SCARY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bscary\b', r'\bscared\b', r'\bscare\w*\b']]
YES_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\byes\b', r'\byeah\b', r'\byep\b', r'\byup\b', r'\bYEA\b']]
NO_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bno+\b', r'\bnope\b', r'\bnah\b', r'\bnuh\s*-?\s*uh\b']]
CHAT_ADDRESS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bchat\b', r'\bguys?\b', r"\by['’]?all\b"]]
YES_YEAH_PATTERNS = YES_PATTERNS
NO_NOPE_PATTERNS = NO_PATTERNS
VOMITING_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bvomit\w*\b', r'\bbarf\w*\b', r'\b(thr(?:ew|ow)(?:s|n)?)\s+up\w*\b']]

BODILY_HUMOR_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r'\bbutt\b', r'\bbutts\b', r'\bfart\w*\b', r'\bpoop\w*\b', r'\bpee\b', r'\bpeed\b',
    r'\bpeeing\b', r'\btooted?\b', r'\bburp\w*\b', r'\bbarf\w*\b', r'\bvomit\w*\b',
    r'\bpuke\w*\b', r'\bturd\w*\b', r'\bbooger\w*\b', r'\bsnot\w*\b', r'\bstink\w*\b',
    r'\bgross\b', r'\bgassy\b', r'\bbutt\s*-?\s*hole\w*\b', r'\bwee\s*-?\s*wee\w*\b'
]]
ACTUAL_PEAK_COMEDY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bpeak comedy\b']]
REALLY_ACTUAL_PEAK_COMEDY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bno fucking way mr gibbs and just joe king got skibidi\'d i am so sad sobs\b']]

UNBELIEVABLY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [r'\bunbelievably\b']]

GAMING_METRIC_CONFIGS: Dict[str, Dict[str, Any]] = {
    'unbelievably_count': {'name': 'Unbelievably', 'title_suffix': 'an unbelievably weird word', 'compute_method': '_compute_unbelievably_count', 'category': 'Silly & Fun Counts'},
    'fart_count': {'name': 'Fart', 'title_suffix': 'we need a fart count', 'compute_method': '_compute_fart_count', 'category': 'Silly & Fun Counts'},
    'dang_count': {'name': 'Dang...', 'title_suffix': 'Count of "dang"', 'compute_method': '_compute_dang_count', 'category': 'Silly & Fun Counts'},
    'damn_count': {'name': 'Damn...', 'title_suffix': 'Count of "damn"', 'compute_method': '_compute_damn_count', 'category': 'Silly & Fun Counts'},
    'darn_count': {'name': 'Darn...', 'title_suffix': 'Count of "darn"', 'compute_method': '_compute_darn_count', 'category': 'Silly & Fun Counts'},
    'man_count': {'name': 'Man...', 'title_suffix': 'Count of "man" as interjection', 'compute_method': '_compute_man_count', 'category': 'Silly & Fun Counts'},
    'shut_up_count': {'name': 'Shut Up >:(', 'title_suffix': 'Count of "shut up" phrase', 'compute_method': '_compute_shut_up_count', 'category': 'Silly & Fun Counts'},
    'ready_count': {'name': 'Ready Count', 'title_suffix': 'Count of "ready"', 'compute_method': '_compute_ready_count', 'category': 'Silly & Fun Counts'},
    'you_count': {'name': 'You Count', 'title_suffix': 'Count of "you"', 'compute_method': '_compute_you_count', 'category': 'Silly & Fun Counts'},
    'i_count': {'name': 'I Count', 'title_suffix': 'Count of "I"', 'compute_method': '_compute_i_count', 'category': 'Silly & Fun Counts'},
    'go_count': {'name': 'Go Count', 'title_suffix': 'Count of "go"', 'compute_method': '_compute_go_count', 'category': 'Silly & Fun Counts'},
    'booger_count': {'name': 'Booger Count', 'title_suffix': 'Count of "booger" and variants', 'compute_method': '_compute_booger_count', 'category': 'Silly & Fun Counts'},
    'yall_count': {'name': "Y'all Count", 'title_suffix': 'Count of "y\'all"', 'compute_method': '_compute_yall_count', 'category': 'Silly & Fun Counts'},
    'you_guys_count': {'name': "You Guys Count", 'title_suffix': 'Count of "you guys"', 'compute_method': '_compute_you_guys_count', 'category': 'Silly & Fun Counts'},
    'god_count': {'name': 'God', 'title_suffix': 'Count of "god" and variants', 'compute_method': '_compute_god_count', 'category': 'Silly & Fun Counts'},
    'baby_count': {'name': 'Baby', 'title_suffix': 'Count of "baby" and variants', 'compute_method': '_compute_baby_count', 'category': 'Silly & Fun Counts'},
    'ding_dong_count': {'name': 'Ding Dong', 'title_suffix': 'Ding Dong', 'compute_method': '_compute_ding_dong_count', 'category': 'Silly & Fun Counts'},
    'volcano_count': {'name': 'VOLCANO', 'title_suffix': 'Count of "volcano"', 'compute_method': '_compute_volcano_count', 'category': 'Silly & Fun Counts'},
    'nostril_humor_count': {'name': 'Nostril Humor Count', 'title_suffix': 'Nose/Nostril/Booger/Snot Count', 'compute_method': '_compute_nostril_humor_count', 'category': 'Silly & Fun Counts'},
    'freak_count': {'name': 'A Freaking Counter', 'title_suffix': 'Words containing "freak"', 'compute_method': '_compute_freak_count', 'category': 'Silly & Fun Counts'},
    'ooh_count': {'name': 'Ooh Count', 'title_suffix': 'Count of "ooh" exclamations', 'compute_method': '_compute_ooh_count', 'category': 'Silly & Fun Counts'},
    'small_word_count': {'name': 'Small', 'title_suffix': 'Little/Short/Small Word Count', 'compute_method': '_compute_small_word_count', 'category': 'Silly & Fun Counts'},
    'big_word_count': {'name': 'Big', 'title_suffix': 'Big/Tall/Giant Word Count', 'compute_method': '_compute_big_word_count', 'category': 'Silly & Fun Counts'},
    'handsome_count': {'name': 'Handsome', 'title_suffix': 'Count of "handsome"', 'compute_method': '_compute_handsome_count', 'category': 'Silly & Fun Counts'},
    'easy_count': {'name': 'Easy Count', 'title_suffix': 'Count of "easy" and variants', 'compute_method': '_compute_easy_count', 'category': 'Silly & Fun Counts'},
    'hard_count': {'name': 'Hard/Difficult Count', 'title_suffix': 'Count of "hard" and "difficult"', 'compute_method': '_compute_hard_count', 'category': 'Silly & Fun Counts'},
    'smores_tasty_count': {'name': 'smoresaretasty', 'title_suffix': 'Count of "smores" and "tasty"', 'compute_method': '_compute_smores_tasty_count', 'category': 'Silly & Fun Counts'},
    'alright_then_count': {'name': 'Alright Then Count', 'title_suffix': 'Count of "alright then" and "alrighty then"', 'compute_method': '_compute_alright_then_count', 'category': 'Silly & Fun Counts'},
    'goodness_count': {'name': 'goodness', 'title_suffix': 'Count of "goodness"', 'compute_method': '_compute_goodness_count', 'category': 'Silly & Fun Counts'},
    'ragdoll_count': {'name': 'Ragdoll Count', 'title_suffix': 'Count of "ragdoll"', 'compute_method': '_compute_ragdoll_count', 'category': 'Silly & Fun Counts'},
    'woah_count': {'name': 'Woah!', 'title_suffix': 'Count of "woah" exclamations', 'compute_method': '_compute_woah_count', 'category': 'Silly & Fun Counts'},
    'oh_boy_count': {'name': 'Oh Boy...', 'title_suffix': 'Count of "oh boy" and "oh man" phrase', 'compute_method': '_compute_oh_boy_count', 'category': 'Silly & Fun Counts'},
    'weird_count': {'name': "That's weird...", 'title_suffix': 'Count of "weird" and variants', 'compute_method': '_compute_weird_count', 'category': 'Silly & Fun Counts'},
    'all_right_count': {'name': 'All Right Count', 'title_suffix': 'Count of "all right" and "alright"', 'compute_method': '_compute_all_right_count', 'category': 'Silly & Fun Counts'},
    'wa_count': {'name': 'Wa Count', 'title_suffix': 'Count of "wa" exclamations', 'compute_method': '_compute_wa_count', 'category': 'Silly & Fun Counts'},
    'punk_count': {'name': 'Punk Count', 'title_suffix': 'Count of "punk" (excludes cyberpunk)', 'compute_method': '_compute_punk_count', 'category': 'Silly & Fun Counts'},
    'yonder_count': {'name': 'Yonder Count', 'title_suffix': 'Count of "yonder"', 'compute_method': '_compute_yonder_count', 'category': 'Silly & Fun Counts'},
    'please_count': {'name': 'Please Count', 'title_suffix': 'Count of "please"', 'compute_method': '_compute_please_count', 'category': 'Silly & Fun Counts'},
    'car_count': {'name': 'Car Count', 'title_suffix': 'Car mentions (car, cars)', 'compute_method': '_compute_car_count', 'category': 'Silly & Fun Counts'},
    'boat_count': {'name': 'Boat Count', 'title_suffix': 'Boat mentions (boat, boats)', 'compute_method': '_compute_boat_count', 'category': 'Silly & Fun Counts'},
    'truck_count': {'name': 'Truck Count', 'title_suffix': 'Truck mentions (truck, trucks)', 'compute_method': '_compute_truck_count', 'category': 'Silly & Fun Counts'},
    'plane_count': {'name': 'Plane Count', 'title_suffix': 'Plane mentions (plane, planes)', 'compute_method': '_compute_plane_count', 'category': 'Silly & Fun Counts'},
    'mr_count': {'name': 'Mr/Mrs Count', 'title_suffix': 'Count of "Mr" and "Mrs" titles', 'compute_method': '_compute_mr_count', 'category': 'Silly & Fun Counts'},
    'subscribe_count': {'name': 'SUBSCRIBE', 'title_suffix': 'Count of "subscribe"', 'compute_method': '_compute_subscribe_count', 'category': 'Silly & Fun Counts'},
    'block_count': {'name': 'Block Count', 'title_suffix': 'Block/Censor count [ __ ]', 'compute_method': '_compute_block_count', 'category': 'Silly & Fun Counts'},
    'vehicle_lexicon_density': {'name': 'Vehicle Vocabulary Count', 'title_suffix': 'Vehicle-Specific Vocabulary Count', 'compute_method': '_compute_vehicle_lexicon_density', 'category': 'Silly & Fun Counts'},
    'animal_count': {'name': 'Animal Count', 'title_suffix': 'Common Animal Name Count', 'compute_method': '_compute_animal_count', 'category': 'Silly & Fun Counts'},
    '67_count': {'name': '67 Count', 'title_suffix': 'Mentions of 67 (numeric and text)', 'compute_method': '_compute_67_count', 'category': 'Silly & Fun Counts'},
    'wafflestomp_count': {'name': 'Wafflestomp lol', 'title_suffix': 'Count of "wafflestomp"', 'compute_method': '_compute_wafflestomp_count', 'category': 'Silly & Fun Counts'},
    'stupid_count': {'name': 'Stupid Count', 'title_suffix': 'Count of "stupid" and variants', 'compute_method': '_compute_stupid_count', 'category': 'Silly & Fun Counts'},
    'idiot_count': {'name': 'Idiot Count', 'title_suffix': 'Count of "idiot" and variants', 'compute_method': '_compute_idiot_count', 'category': 'Silly & Fun Counts'},
    'also_count': {'name': 'Also Count', 'title_suffix': 'Count of "also"', 'compute_method': '_compute_also_count', 'category': 'Silly & Fun Counts'},
    'are_you_kidding_count': {'name': 'Are You Kidding Count', 'title_suffix': 'Count of "are you kidding" phrase', 'compute_method': '_compute_are_you_kidding_count', 'category': 'Silly & Fun Counts'},
    'bad_boy_count': {'name': 'Bad Boy Count', 'title_suffix': 'Count of "bad boy" phrase', 'compute_method': '_compute_bad_boy_count', 'category': 'Silly & Fun Counts'},
    'yoink_count': {'name': 'Yoink Count', 'title_suffix': 'Count of "yoink" and variants', 'compute_method': '_compute_yoink_count', 'category': 'Silly & Fun Counts'},
    'high_five_count': {'name': 'High Five Count', 'title_suffix': 'Count of "high five" phrase', 'compute_method': '_compute_high_five_count', 'category': 'Silly & Fun Counts'},
    'technically_count': {'name': 'Technically Count', 'title_suffix': 'Count of "technically"', 'compute_method': '_compute_technically_count', 'category': 'Silly & Fun Counts'},
    'nailed_it_count': {'name': 'Nailed It Count', 'title_suffix': 'Count of "nailed it" phrase', 'compute_method': '_compute_nailed_it_count', 'category': 'Silly & Fun Counts'},
    'jeez_geez_count': {'name': 'Jeez/Geez Count', 'title_suffix': 'Count of "jeez" and "geez"', 'compute_method': '_compute_jeez_geez_count', 'category': 'Silly & Fun Counts'},
    'tree_word_count': {'name': 'Tree Word Count', 'title_suffix': 'Count of tree words (cedar, oak, maple, birch, aspen, etc.)', 'compute_method': '_compute_tree_word_count', 'category': 'Silly & Fun Counts'},
    'blitz_count': {'name': 'Blitz Count', 'title_suffix': 'Count of "blitz"', 'compute_method': '_compute_blitz_count', 'category': 'Silly & Fun Counts'},
    'bloody_count': {'name': 'Bloody Count', 'title_suffix': 'Count of "bloody" (UK intensifier)', 'compute_method': '_compute_bloody_count', 'category': 'Silly & Fun Counts'},
    'bot_count': {'name': 'Bot Count', 'title_suffix': 'Count of "bot" and variants', 'compute_method': '_compute_bot_count', 'category': 'Silly & Fun Counts'},
    'bridge_count': {'name': 'Bridge Count', 'title_suffix': 'Count of "bridge" and variants', 'compute_method': '_compute_bridge_count', 'category': 'Silly & Fun Counts'},
    'bro_count': {'name': 'Bro Count', 'title_suffix': 'Count of "bro" and variants', 'compute_method': '_compute_bro_count', 'category': 'Silly & Fun Counts'},
    'burger_count': {'name': 'Burger Count', 'title_suffix': 'Count of burger types (cheeseburger, hamburger, burger)', 'compute_method': '_compute_burger_count', 'category': 'Silly & Fun Counts'},
    'bye_count': {'name': 'Bye Count', 'title_suffix': 'Count of "bye" and "goodbye" variants', 'compute_method': '_compute_bye_count', 'category': 'Silly & Fun Counts'},
    'checkpoint_count': {'name': 'Checkpoint Count', 'title_suffix': 'Count of "checkpoint" mentions', 'compute_method': '_compute_checkpoint_count', 'category': 'Silly & Fun Counts'},
    'cheeky_count': {'name': 'Cheeky Count', 'title_suffix': 'Count of "cheeky" and variants', 'compute_method': '_compute_cheeky_count', 'category': 'Silly & Fun Counts'},
    'chill_count': {'name': 'Chill...', 'title_suffix': 'Count of "chill" and variants', 'compute_method': '_compute_chill_count', 'category': 'Silly & Fun Counts'},
    'crazy_count': {'name': 'Crazy Count', 'title_suffix': 'Count of "crazy" and variants', 'compute_method': '_compute_crazy_count', 'category': 'Silly & Fun Counts'},
    'cringe_count': {'name': 'Cringe Count', 'title_suffix': 'Count of "cringe" and variants', 'compute_method': '_compute_cringe_count', 'category': 'Silly & Fun Counts'},
    'damage_count': {'name': 'Damage Count', 'title_suffix': 'Count of "damage" and variants', 'compute_method': '_compute_damage_count', 'category': 'Silly & Fun Counts'},
    'dnf_count': {'name': 'DNF Count', 'title_suffix': 'Count of "DNF" (did not finish) mentions', 'compute_method': '_compute_dnf_count', 'category': 'Silly & Fun Counts'},
    'dude_count': {'name': 'Dude Count', 'title_suffix': 'Count of "dude" and variants', 'compute_method': '_compute_dude_count', 'category': 'Silly & Fun Counts'},
    'editor_count': {'name': 'Editor Count', 'title_suffix': 'Count of "editor"', 'compute_method': '_compute_editor_count', 'category': 'Silly & Fun Counts'},
    'embarrass_count': {'name': 'Embarrass Count', 'title_suffix': 'Count of "embarrass"/"embarrassed"/"embarrassing"', 'compute_method': '_compute_embarrass_count', 'category': 'Silly & Fun Counts'},
    'engineer_count': {'name': 'Engineer Count', 'title_suffix': 'Count of "engineer" and variants', 'compute_method': '_compute_engineer_count', 'category': 'Silly & Fun Counts'},
    'explosion_count': {'name': 'Explosion Count', 'title_suffix': 'Count of explosion/explode and variants', 'compute_method': '_compute_explosion_count', 'category': 'Silly & Fun Counts'},
    'finish_line_count': {'name': 'Finish Line Count', 'title_suffix': 'Count of "finish line" mentions', 'compute_method': '_compute_finish_line_count', 'category': 'Silly & Fun Counts'},
    'fumble_count': {'name': 'Fumble Count', 'title_suffix': 'Count of "fumble"/"fumbling"/"fumbled"', 'compute_method': '_compute_fumble_count', 'category': 'Silly & Fun Counts'},
    'go_ahead_count': {'name': 'Go Ahead Count', 'title_suffix': 'Count of "go ahead" phrase', 'compute_method': '_compute_go_ahead_count', 'category': 'Silly & Fun Counts'},
    'goo_goop_count': {'name': 'Goo/Goop Count', 'title_suffix': 'Count of "goo", "goos", and "goop" variants', 'compute_method': '_compute_goo_goop_count', 'category': 'Silly & Fun Counts'},
    'grandma_count': {'name': 'Grandma Count', 'title_suffix': 'Count of "grandma" and variants', 'compute_method': '_compute_grandma_count', 'category': 'Silly & Fun Counts'},
    'good_question_count': {'name': 'Good Question Count', 'title_suffix': 'Count of "good question" phrase', 'compute_method': '_compute_good_question_count', 'category': 'Silly & Fun Counts'},
    'hang_on_count': {'name': 'Hang On Count', 'title_suffix': 'Count of "hang on" phrase', 'compute_method': '_compute_hang_on_count', 'category': 'Silly & Fun Counts'},
    'hear_me_out_count': {'name': 'Hear Me Out Count', 'title_suffix': 'Count of "hear me out" phrase', 'compute_method': '_compute_hear_me_out_count', 'category': 'Silly & Fun Counts'},
    'hello_count': {'name': 'Hello Count', 'title_suffix': 'Count of "hello" and variants', 'compute_method': '_compute_hello_count', 'category': 'Silly & Fun Counts'},
    'hold_on_count': {'name': 'Hold On Count', 'title_suffix': 'Count of "hold on" phrase', 'compute_method': '_compute_hold_on_count', 'category': 'Silly & Fun Counts'},
    'hold_up_count': {'name': 'Hold Up Count', 'title_suffix': 'Count of "hold up" phrase', 'compute_method': '_compute_hold_up_count', 'category': 'Silly & Fun Counts'},
    'homie_count': {'name': 'Homie Count', 'title_suffix': 'Count of "homie" and variants', 'compute_method': '_compute_homie_count', 'category': 'Silly & Fun Counts'},
    'hypothetical_count': {'name': 'Hypothetical Count', 'title_suffix': 'Count of "hypothetical" and variants', 'compute_method': '_compute_hypothetical_count', 'category': 'Silly & Fun Counts'},
    'i_can_count': {'name': 'I Can Count', 'title_suffix': 'Count of "I can" phrase', 'compute_method': '_compute_i_can_count', 'category': 'Silly & Fun Counts'},
    'i_feel_count': {'name': 'I Feel Count', 'title_suffix': 'Count of "I feel" phrase', 'compute_method': '_compute_i_feel_count', 'category': 'Silly & Fun Counts'},
    'i_swear_count': {'name': 'I Swear Count', 'title_suffix': 'Count of "I swear" phrase', 'compute_method': '_compute_i_swear_count', 'category': 'Silly & Fun Counts'},
    'i_think_count': {'name': 'I Think Count', 'title_suffix': 'Count of "I think" phrase', 'compute_method': '_compute_i_think_count', 'category': 'Silly & Fun Counts'},
    'insane_count': {'name': 'Insane Count', 'title_suffix': 'Count of "insane" and variants', 'compute_method': '_compute_insane_count', 'category': 'Silly & Fun Counts'},
    'its_okay_count': {'name': 'Its Okay Count', 'title_suffix': 'Count of "it\'s okay", "it is okay", "it\'s ok", "it is ok"', 'compute_method': '_compute_its_okay_count', 'category': 'Silly & Fun Counts'},
    'lame_count': {'name': 'Lame Count', 'title_suffix': 'Count of "lame" and variants', 'compute_method': '_compute_lame_count', 'category': 'Silly & Fun Counts'},
    'lets_go_count': {'name': 'Lets Go Count', 'title_suffix': 'Count of "let\'s go" phrase', 'compute_method': '_compute_lets_go_count', 'category': 'Silly & Fun Counts'},
    'lock_in_count': {'name': 'Lock In Count', 'title_suffix': 'Count of "lock in"/"locking in"/"locked in"', 'compute_method': '_compute_lock_in_count', 'category': 'Silly & Fun Counts'},
    'loo_count': {'name': 'Loo Count', 'title_suffix': 'Count of "loo"/"loos" (UK bathroom)', 'compute_method': '_compute_loo_count', 'category': 'Silly & Fun Counts'},
    'mate_count': {'name': 'Mate Count', 'title_suffix': 'Count of "mate" mentions', 'compute_method': '_compute_mate_count', 'category': 'Silly & Fun Counts'},
    'make_sure_count': {'name': 'Make Sure Count', 'title_suffix': 'Count of "make sure" phrase', 'compute_method': '_compute_make_sure_count', 'category': 'Silly & Fun Counts'},
    'makes_me_count': {'name': 'Makes Me Count', 'title_suffix': 'Count of "makes me" phrase', 'compute_method': '_compute_makes_me_count', 'category': 'Silly & Fun Counts'},
    'murder_count': {'name': 'Murder Count', 'title_suffix': 'Murder Count', 'compute_method': '_compute_murder_count', 'category': 'Silly & Fun Counts'},
    'nonchalant_count': {'name': 'Nonchalant Count', 'title_suffix': 'Count of "nonchalant" and variants', 'compute_method': '_compute_nonchalant_count', 'category': 'Silly & Fun Counts'},
    'outrageous_count': {'name': 'Outrageous Count', 'title_suffix': 'Count of "outrageous" and variants', 'compute_method': '_compute_outrageous_count', 'category': 'Silly & Fun Counts'},
    'ow_count': {'name': 'Ow Count', 'title_suffix': 'Count of "ow"/"oww" exclamations', 'compute_method': '_compute_ow_count', 'category': 'Silly & Fun Counts'},
    'okay_so_count': {'name': 'Okay So Count', 'title_suffix': 'Count of "okay so" and "ok so"', 'compute_method': '_compute_okay_so_count', 'category': 'Silly & Fun Counts'},
    'pookie_count': {'name': 'Pookie Count', 'title_suffix': 'Count of "pookie"/"pooki"/"pooky" and variants', 'compute_method': '_compute_pookie_count', 'category': 'Silly & Fun Counts'},
    'probably_count': {'name': 'Probably Count', 'title_suffix': 'Count of "probably" and variants', 'compute_method': '_compute_probably_count', 'category': 'Silly & Fun Counts'},
    'put_count': {'name': 'Put Count', 'title_suffix': 'Count of "put" and variants', 'compute_method': '_compute_put_count', 'category': 'Silly & Fun Counts'},
    'rat_count': {'name': 'Rat Count', 'title_suffix': 'Count of "rat" and variants', 'compute_method': '_compute_rat_count', 'category': 'Silly & Fun Counts'},
    'real_quick_count': {'name': 'Real Quick Count', 'title_suffix': 'Count of "real quick" phrase', 'compute_method': '_compute_real_quick_count', 'category': 'Silly & Fun Counts'},
    'rubbish_count': {'name': 'Rubbish Count', 'title_suffix': 'Count of "rubbish" (UK garbage/trash)', 'compute_method': '_compute_rubbish_count', 'category': 'Silly & Fun Counts'},
    'see_you_next_time_count': {'name': 'See You Next Time Count', 'title_suffix': 'Count of "see you next time" phrase', 'compute_method': '_compute_see_you_next_time_count', 'category': 'Silly & Fun Counts'},
    'skill_issue_count': {'name': 'Skill Issue Count', 'title_suffix': 'Count of "skill issue"', 'compute_method': '_compute_skill_issue_count', 'category': 'Silly & Fun Counts'},
    'sneaky_count': {'name': 'Sneaky Count', 'title_suffix': 'Count of "sneaky" and variants', 'compute_method': '_compute_sneaky_count', 'category': 'Silly & Fun Counts'},
    'so_cool_count': {'name': 'So Cool Count', 'title_suffix': 'Count of "so cool" phrase', 'compute_method': '_compute_so_cool_count', 'category': 'Silly & Fun Counts'},
    'son_of_a_count': {'name': 'Son Of A Count', 'title_suffix': 'Count of "son of a" phrase', 'compute_method': '_compute_son_of_a_count', 'category': 'Silly & Fun Counts'},
    'son_of_a_not_before_block_count': {'name': 'FF Son Of A Count', 'title_suffix': 'Count of "son of a" phrase not before a block', 'compute_method': '_compute_son_of_a_not_before_block_count', 'category': 'Silly & Fun Counts'},
    'sorry_count': {'name': 'Sorry Count', 'title_suffix': 'Count of "sorry" and variants', 'compute_method': '_compute_sorry_count', 'category': 'Silly & Fun Counts'},
    'spanner_count': {'name': 'Spanner Count', 'title_suffix': 'Count of "spanner" (UK wrench)', 'compute_method': '_compute_spanner_count', 'category': 'Silly & Fun Counts'},
    'sure_count': {'name': 'Sure Count', 'title_suffix': 'Count of "sure" and variants', 'compute_method': '_compute_sure_count', 'category': 'Silly & Fun Counts'},
    'stop_count': {'name': 'Stop Count', 'title_suffix': 'Count of "stop" and variants', 'compute_method': '_compute_stop_count', 'category': 'Silly & Fun Counts'},
    'sweat_count': {'name': 'Sweat Count', 'title_suffix': 'Count of "sweat" and variants', 'compute_method': '_compute_sweat_count', 'category': 'Silly & Fun Counts'},
    'tea_count': {'name': 'Tea Count', 'title_suffix': 'Count of "tea"', 'compute_method': '_compute_tea_count', 'category': 'Silly & Fun Counts'},
    'thanks_for_watching_count': {'name': 'Thanks For Watching Count', 'title_suffix': 'Count of "thanks for watching" phrase', 'compute_method': '_compute_thanks_for_watching_count', 'category': 'Silly & Fun Counts'},
    'thats_okay_count': {'name': 'Thats Okay Count', 'title_suffix': 'Count of "that\'s okay"/"that is fine" phrases', 'compute_method': '_compute_thats_okay_count', 'category': 'Silly & Fun Counts'},
    'take_care_count': {'name': 'Take Care Count', 'title_suffix': 'Count of "take care" phrase', 'compute_method': '_compute_take_care_count', 'category': 'Silly & Fun Counts'},
    'to_be_fair_count': {'name': 'To Be Fair Count', 'title_suffix': 'Count of "to be fair" phrase', 'compute_method': '_compute_to_be_fair_count', 'category': 'Silly & Fun Counts'},
    'to_be_honest_count': {'name': 'To Be Honest Count', 'title_suffix': 'Count of "to be honest" phrase', 'compute_method': '_compute_to_be_honest_count', 'category': 'Silly & Fun Counts'},
    'too_count': {'name': 'Too Count', 'title_suffix': 'Count of "too"', 'compute_method': '_compute_too_count', 'category': 'Silly & Fun Counts'},
    'uh_count': {'name': 'Uh Count', 'title_suffix': 'Count of "uh" filler word', 'compute_method': '_compute_uh_count', 'category': 'Silly & Fun Counts'},
    'um_count': {'name': 'Um Count', 'title_suffix': 'Count of "um"/"uhm" filler words', 'compute_method': '_compute_um_count', 'category': 'Silly & Fun Counts'},
    'upgrade_count': {'name': 'Upgrade Count', 'title_suffix': 'Count of "upgrade" and variants', 'compute_method': '_compute_upgrade_count', 'category': 'Silly & Fun Counts'},
    'violation_count': {'name': 'Violation Count', 'title_suffix': 'Count of "violation" and "violate"', 'compute_method': '_compute_violation_count', 'category': 'Silly & Fun Counts'},
    'wait_count': {'name': 'Wait Count', 'title_suffix': 'Count of "wait" and variants', 'compute_method': '_compute_wait_count', 'category': 'Silly & Fun Counts'},
    'well_done_count': {'name': 'Well Done Count', 'title_suffix': 'is this how you like your steak done?', 'compute_method': '_compute_well_done_count', 'category': 'Silly & Fun Counts'},
    'what_do_you_mean_count': {'name': 'What Do You Mean Count', 'title_suffix': 'Count of "what do you mean" phrase', 'compute_method': '_compute_what_do_you_mean_count', 'category': 'Silly & Fun Counts'},
    'whole_bunch_count': {'name': 'Whole Bunch Count', 'title_suffix': 'Count of "whole bunch" phrase', 'compute_method': '_compute_whole_bunch_count', 'category': 'Silly & Fun Counts'},
    'wrench_count': {'name': 'Wrench Count', 'title_suffix': 'Count of "wrench" and variants', 'compute_method': '_compute_wrench_count', 'category': 'Silly & Fun Counts'},
    'youre_a_little_count': {'name': 'You\'re A Little Count', 'title_suffix': 'Count of "you\'re a little" and "you\'re such a little"', 'compute_method': '_compute_youre_a_little_count', 'category': 'Silly & Fun Counts'},

    'roygbiv_count': {'name': 'Roygbiv Count', 'title_suffix': 'the boc song', 'compute_method': '_compute_roygbiv_count', 'category': 'Silly & Fun Counts'},
    'duck_goose_count': {'name': 'Duck/Goose Count', 'title_suffix': 'Count of duck/goose/geese variants', 'compute_method': '_compute_duck_goose_count', 'category': 'Silly & Fun Counts'},
    'marry_count': {'name': 'Marry Count', 'title_suffix': 'Count of "marry" and variants', 'compute_method': '_compute_marry_count', 'category': 'Silly & Fun Counts'},
    'scary_count': {'name': 'Scary Count', 'title_suffix': 'Count of "scary" and scare variants', 'compute_method': '_compute_scary_count', 'category': 'Silly & Fun Counts'},
    'yes_count': {'name': 'Yes Count', 'title_suffix': 'Count of yes/yeah/yep/yup variants', 'compute_method': '_compute_yes_count', 'category': 'Silly & Fun Counts'},
    'no_count': {'name': 'No Count', 'title_suffix': 'Count of no/nope/nah variants', 'compute_method': '_compute_no_count', 'category': 'Silly & Fun Counts'},
    'chat_address_count': {'name': 'Chat Address Count', 'title_suffix': '"chat", "guys", "yall"', 'compute_method': '_compute_chat_address_count', 'category': 'Interactional'},
    'yes_yeah_count': {'name': 'Yes/Yeah Count', 'title_suffix': 'Count of yes/yeah/yep/yup ', 'compute_method': '_compute_yes_yeah_count', 'category': 'Agreement/Disagreement'},
    'no_nope_count': {'name': 'No/Nope Count', 'title_suffix': 'Count of no/nope ', 'compute_method': '_compute_no_nope_count', 'category': 'Agreement/Disagreement'},
    'vomiting_count': {'name': 'Vomiting Count', 'title_suffix': 'Count of "vomit", "barf", and "throw up" variants', 'compute_method': '_compute_vomiting_count', 'category': 'Silly & Fun Counts'},

    'going_to_count': {'name': 'Going To Count', 'title_suffix': 'Count of "going to" phrase', 'compute_method': '_compute_going_to_count', 'category': 'Common Phrases'},
    'double_modal_count': {'name': 'Double Modal Count', 'title_suffix': 'Count of double modals (might could, may should, etc.)', 'compute_method': '_compute_double_modal_count', 'category': 'Common Phrases'},
    'triple_auxiliary_count': {'name': 'Triple Auxiliary Count', 'title_suffix': 'Count of triple auxiliary stacks (might have been, would\'ve had, etc.)', 'compute_method': '_compute_triple_auxiliary_count', 'category': 'Common Phrases'},
    'modal_perfect_count': {'name': 'Modal Perfect Count', 'title_suffix': 'Count of modal + have/\'ve constructions', 'compute_method': '_compute_modal_perfect_count', 'category': 'Common Phrases'},
    'modal_progressive_count': {'name': 'Modal Progressive Count', 'title_suffix': 'Count of modal + be + VBG constructions', 'compute_method': '_compute_modal_progressive_count', 'category': 'Common Phrases'},
    'modal_passive_count': {'name': 'Modal Passive Count', 'title_suffix': 'Count of modal + be + participle constructions', 'compute_method': '_compute_modal_passive_count', 'category': 'Common Phrases'},
    'semi_modal_count': {'name': 'Semi-Modal Count', 'title_suffix': 'Count of semi-modals (have to, need to, got to, gonna)', 'compute_method': '_compute_semi_modal_count', 'category': 'Common Phrases'},
    'modal_negation_count': {'name': 'Modal Negation Count', 'title_suffix': 'Count of modal negations (can\'t, won\'t, might not, etc.)', 'compute_method': '_compute_modal_negation_count', 'category': 'Common Phrases'},
    'conditional_modal_count': {'name': 'Conditional Modal Count', 'title_suffix': 'Count of local if + modal constructions', 'compute_method': '_compute_conditional_modal_count', 'category': 'Common Phrases'},
    'be_gonna_count': {'name': 'Be-Gonna Count', 'title_suffix': 'Count of be + gonna / be + going to future constructions', 'compute_method': '_compute_be_gonna_count', 'category': 'Common Phrases'},
    'be_gotta_count': {'name': 'Be-Gotta Count', 'title_suffix': 'Count of have got to / gotta constructions', 'compute_method': '_compute_be_gotta_count', 'category': 'Common Phrases'},
    'stance_marker_count': {'name': 'Stance Marker Count', 'title_suffix': 'Count of stance markers (I think, I guess, I mean, etc.)', 'compute_method': '_compute_stance_marker_count', 'category': 'Common Phrases'},
    'evidential_marker_count': {'name': 'Evidential Marker Count', 'title_suffix': 'Count of evidential markers (apparently, seems like, etc.)', 'compute_method': '_compute_evidential_marker_count', 'category': 'Common Phrases'},
    'intensifier_stack_count': {'name': 'Intensifier Stack Count', 'title_suffix': 'Count of stacked intensifiers (so very, really really, etc.)', 'compute_method': '_compute_intensifier_stack_count', 'category': 'Common Phrases'},
    'hedge_stack_count': {'name': 'Hedge Stack Count', 'title_suffix': 'Count of stacked hedges (maybe kinda, probably sort of, etc.)', 'compute_method': '_compute_hedge_stack_count', 'category': 'Common Phrases'},
    'tag_question_count': {'name': 'Tag Question Count', 'title_suffix': 'Count of tag-question markers (right, isn\'t it, don\'t you, etc.)', 'compute_method': '_compute_tag_question_count', 'category': 'Common Phrases'},
    'repair_restart_count': {'name': 'Repair/Restart Count', 'title_suffix': 'Count of self-repair/restart markers (I mean, or rather, etc.)', 'compute_method': '_compute_repair_restart_count', 'category': 'Common Phrases'},
    'disfluency_cluster_count': {'name': 'Disfluency Clusters Count', 'title_suffix': 'Count of clustered fillers (uh um, um uh, repeated fillers)', 'compute_method': '_compute_disfluency_cluster_count', 'category': 'Common Phrases'},
    'over_to_count': {'name': 'Over To Count', 'title_suffix': 'Count of "over to" phrase', 'compute_method': '_compute_over_to_count', 'category': 'Common Phrases'},
    'brought_count': {'name': 'Brought Count', 'title_suffix': 'Count of "brought"', 'compute_method': '_compute_brought_count', 'category': 'Common Phrases'},
    'took_count': {'name': 'Took Count', 'title_suffix': 'Count of "took"', 'compute_method': '_compute_took_count', 'category': 'Common Phrases'},
    'well_count': {'name': 'Well Count', 'title_suffix': 'Count of "well"', 'compute_method': '_compute_well_count', 'category': 'Common Phrases'},
    'reckon_count': {'name': 'Reckon Count', 'title_suffix': 'Count of "reckon" and variants', 'compute_method': '_compute_reckon_count', 'category': 'Common Phrases'},
    'i_dont_know_count': {'name': 'I Don\'t Know Count', 'title_suffix': 'Count of "I don\'t know" phrase', 'compute_method': '_compute_i_dont_know_count', 'category': 'Common Phrases'},
    'you_know_what_count': {'name': 'You Know What Count', 'title_suffix': 'Count of "you know what" phrase', 'compute_method': '_compute_you_know_what_count', 'category': 'Common Phrases'},
    'oh_my_god_count': {'name': 'Oh My God Count', 'title_suffix': 'Count of "oh my god" phrase', 'compute_method': '_compute_oh_my_god_count', 'category': 'Common Phrases'},
    'oh_my_gosh_count': {'name': 'Oh My Gosh Count', 'title_suffix': 'Count of "oh my gosh" phrase', 'compute_method': '_compute_oh_my_gosh_count', 'category': 'Common Phrases'},
    'oh_my_goodness_count': {'name': 'Oh My Goodness Count', 'title_suffix': 'Count of "oh my goodness" phrase', 'compute_method': '_compute_oh_my_goodness_count', 'category': 'Common Phrases'},
    'there_we_go_count': {'name': 'There We Go Count', 'title_suffix': 'Count of "there we go" phrase', 'compute_method': '_compute_there_we_go_count', 'category': 'Common Phrases'},
    'here_we_go_count': {'name': 'Here We Go Count', 'title_suffix': 'Count of "here we go" phrase', 'compute_method': '_compute_here_we_go_count', 'category': 'Common Phrases'},
    'look_at_that_count': {'name': 'Look At That Count', 'title_suffix': 'Count of "look at that" phrase', 'compute_method': '_compute_look_at_that_count', 'category': 'Common Phrases'},
    'oh_no_count': {'name': 'Oh No Count', 'title_suffix': 'Count of "oh no" phrase', 'compute_method': '_compute_oh_no_count', 'category': 'Common Phrases'},
    'im_not_a_count': {'name': 'I\'m Not A / You\'re Not A Count', 'title_suffix': 'Count of "I\'m not a" and "you\'re not a" phrases', 'compute_method': '_compute_im_not_a_count', 'category': 'Common Phrases'},
    'give_me_count': {'name': 'Give Me Count', 'title_suffix': 'Count of "give me" phrase', 'compute_method': '_compute_give_me_count', 'category': 'Common Phrases'},
    'secret_count': {'name': 'Secret String 1', 'title_suffix': 'Secret string 1!', 'compute_method': '_compute_secret_count', 'category': 'Common Phrases'},
    'buddy_count': {'name': 'Secret String 2', 'title_suffix': 'Secret string 2!', 'compute_method': '_compute_buddy_count', 'category': 'Common Phrases'},
    'the_count': {'name': 'The Count', 'title_suffix': 'Count of definite article "the"', 'compute_method': '_compute_the_count', 'category': 'Common Phrases'},

    'love_you_count': {'name': 'Love You Count', 'title_suffix': 'Count of "love you"', 'compute_method': '_compute_love_you_count', 'category': 'Person Name Counts'},
    'hate_you_count': {'name': 'Hate You Count', 'title_suffix': 'Count of "hate you"', 'compute_method': '_compute_hate_you_count', 'category': 'Person Name Counts'},
    'thank_you_count': {'name': 'Thank You Count', 'title_suffix': 'Count of "thank you" and variants', 'compute_method': '_compute_thank_you_count', 'category': 'Person Name Counts'},
    'foreign_count': {'name': 'Foreign Count', 'title_suffix': 'Count of "foreign"', 'compute_method': '_compute_foreign_count', 'category': 'Person Name Counts'},
    'heat_count': {'name': 'Heat Count', 'title_suffix': 'Count of "heat" and variants', 'compute_method': '_compute_heat_count', 'category': 'Person Name Counts'},
    'dog_count': {'name': 'Dog words count', 'title_suffix': 'Dog/puppy/bark mentions', 'compute_method': '_compute_dog_count', 'category': 'Person Name Counts'},
    'cat_count': {'name': 'Cat words count', 'title_suffix': 'Cat/kitty/meow mentions', 'compute_method': '_compute_cat_count', 'category': 'Person Name Counts'},
    'lesbian_count': {'name': 'Lesbian Count', 'title_suffix': 'Count of "lesbian"', 'compute_method': '_compute_lesbian_count', 'category': 'Person Name Counts'},
    'gay_count': {'name': 'Gay Count', 'title_suffix': 'Count of "gay"', 'compute_method': '_compute_gay_count', 'category': 'Person Name Counts'},
    'pasta_count': {'name': 'Pasta Count', 'title_suffix': 'Count of "pasta"', 'compute_method': '_compute_pasta_count', 'category': 'Person Name Counts'},
    'bagel_count': {'name': 'Bagel Count', 'title_suffix': 'Count of "bagel"', 'compute_method': '_compute_bagel_count', 'category': 'Person Name Counts'},
    'max_count': {'name': 'Max Count', 'title_suffix': 'Count of "max" and variants', 'compute_method': '_compute_max_count', 'category': 'Person Name Counts'},
    'money_count': {'name': 'Money Count', 'title_suffix': 'Money/cash/dollar mentions', 'compute_method': '_compute_money_count', 'category': 'Person Name Counts'},
    'moon_count': {'name': 'Moon Count', 'title_suffix': 'Count of "moon"', 'compute_method': '_compute_moon_count', 'category': 'Person Name Counts'},
    'fast_food_count': {'name': 'Fast Food Count', 'title_suffix': 'Count of "McDonald\'s", "Burger King", "KFC", and "Taco Bell"', 'compute_method': '_compute_fast_food_count', 'category': 'Person Name Counts'},
    'excellent_count': {'name': 'Excellent Count', 'title_suffix': 'Count of "excellent" and variants', 'compute_method': '_compute_excellent_count', 'category': 'Person Name Counts'},

    'minecraft_lexicon_density': {'name': 'Minecraft Vocabulary Count', 'title_suffix': 'Minecraft-Specific Vocabulary Count', 'compute_method': '_compute_minecraft_lexicon_density', 'category': 'Topic-Specific Vocabulary'},

    'bodily_humor': {'name': 'Bodily Humor Count', 'title_suffix': 'Bodily Humor Words (fart, poop, burp, etc.)', 'compute_method': '_compute_bodily_humor', 'category': 'Profanity & Insults'},
    'peak_comedy_count': {'name': 'Peak Comedy Count', 'title_suffix': 'absolute cinnamon', 'compute_method': '_compute_peak_comedy_count', 'category': 'Profanity & Insults'},
    'actual_peak_comedy_count': {'name': 'ACTUAL Peak Comedy Count', 'title_suffix': 'This is ACTUAL peak comedy no arguing', 'compute_method': '_compute_actual_peak_comedy_count', 'category': 'Profanity & Insults'},
    'really_actual_peak_comedy_count': {'name': 'TRUE ACTUAL Peak Comedy Count', 'title_suffix': 'This is TRULY ACTUAL peak comedy no arguing this time for real', 'compute_method': '_compute_really_actual_peak_comedy_count', 'category': 'Profanity & Insults'},
}

for _letter in 'abcdefghijklmnopqrstuvwxyzñ':
    GAMING_METRIC_CONFIGS[f'letter_{_letter}_count'] = {
        'name': f'Letter {_letter.upper()} Count',
        'title_suffix': f'Count of Letter {_letter.upper()}',
        'compute_method': '_compute_letter_count',
        'category': 'Letter Frequencies',
    }
del _letter


class GamingMetricsMixin:

    def _compute_letter_count(self, tokens, letter: str = None) -> float:
        if not tokens:
            return 0.0
        text_lower = self._get_runtime_text_lower(tokens)
        return float(text_lower.count(letter.lower() if letter else 'a'))

    def _compute_unbelievably_count(self, tokens): return self._compute_pattern_count(tokens, UNBELIEVABLY_PATTERNS)
    def _compute_fart_count(self, tokens): return self._compute_pattern_count(tokens, FART_PATTERNS)
    def _compute_god_count(self, tokens): return self._compute_pattern_count(tokens, GOD_PATTERNS)
    def _compute_baby_count(self, tokens): return self._compute_pattern_count(tokens, BABY_PATTERNS)
    def _compute_ding_dong_count(self, tokens): return self._compute_pattern_count(tokens, DING_DONG_PATTERNS)
    def _compute_nostril_humor_count(self, tokens): return self._compute_pattern_count(tokens, NOSTRIL_HUMOR_PATTERNS)
    def _compute_shut_up_count(self, tokens): return self._compute_pattern_count(tokens, SHUT_UP_PATTERNS)
    def _compute_ready_count(self, tokens): return self._compute_pattern_count(tokens, READY_PATTERNS)
    def _compute_you_count(self, tokens): return self._compute_pattern_count(tokens, YOU_PATTERNS)
    def _compute_i_count(self, tokens): return self._compute_pattern_count(tokens, I_PATTERNS)
    def _compute_go_count(self, tokens): return self._compute_pattern_count(tokens, GO_PATTERNS)
    def _compute_booger_count(self, tokens): return self._compute_pattern_count(tokens, BOOGER_PATTERNS)
    def _compute_freak_count(self, tokens): return self._compute_pattern_count(tokens, FREAK_PATTERNS)
    def _compute_ooh_count(self, tokens): return self._compute_pattern_count(tokens, OOH_PATTERNS)
    def _compute_yall_count(self, tokens): return self._compute_pattern_count(tokens, YALL_PATTERNS)
    def _compute_you_guys_count(self, tokens): return self._compute_pattern_count(tokens, YOU_GUYS_PATTERNS)
    def _compute_yonder_count(self, tokens): return self._compute_pattern_count(tokens, YONDER_PATTERNS)
    def _compute_small_word_count(self, tokens): return self._compute_pattern_count(tokens, SMALL_WORD_PATTERNS)
    def _compute_big_word_count(self, tokens): return self._compute_pattern_count(tokens, BIG_WORD_PATTERNS)
    def _compute_handsome_count(self, tokens): return self._compute_pattern_count(tokens, HANDSOME_PATTERNS)
    def _compute_easy_count(self, tokens): return self._compute_pattern_count(tokens, EASY_PATTERNS)
    def _compute_hard_count(self, tokens): return self._compute_pattern_count(tokens, HARD_DIFFICULT_PATTERNS)
    def _compute_smores_tasty_count(self, tokens): return self._compute_pattern_count(tokens, SMORES_TASTY_PATTERNS)
    def _compute_alright_then_count(self, tokens): return self._compute_pattern_count(tokens, ALRIGHT_THEN_PATTERNS)
    def _compute_dang_count(self, tokens): return self._compute_pattern_count(tokens, DANG_PATTERNS)
    def _compute_damn_count(self, tokens): return self._compute_pattern_count(tokens, DAMN_PATTERNS)
    def _compute_darn_count(self, tokens): return self._compute_pattern_count(tokens, DARN_PATTERNS)
    def _compute_animal_count(self, tokens): return self._compute_pattern_count(tokens, ANIMAL_PATTERNS)
    def _compute_man_count(self, tokens): return self._compute_pattern_count(tokens, MAN_PATTERNS)
    def _compute_goodness_count(self, tokens): return self._compute_pattern_count(tokens, GOODNESS_PATTERNS)
    def _compute_ragdoll_count(self, tokens): return self._compute_pattern_count(tokens, RAGDOLL_PATTERNS)
    def _compute_woah_count(self, tokens): return self._compute_pattern_count(tokens, WOAH_PATTERNS)
    def _compute_oh_boy_count(self, tokens): return self._compute_pattern_count(tokens, OH_BOY_PATTERNS)
    def _compute_weird_count(self, tokens): return self._compute_pattern_count(tokens, WEIRD_PATTERNS)
    def _compute_bridge_count(self, tokens): return self._compute_pattern_count(tokens, BRIDGE_PATTERNS)
    def _compute_engineer_count(self, tokens): return self._compute_pattern_count(tokens, ENGINEER_PATTERNS)
    def _compute_subscribe_count(self, tokens): return self._compute_pattern_count(tokens, SUBSCRIBE_PATTERNS)
    def _compute_chill_count(self, tokens): return self._compute_pattern_count(tokens, CHILL_PATTERNS)
    def _compute_volcano_count(self, tokens): return self._compute_pattern_count(tokens, VOLCANO_PATTERNS)
    def _compute_editor_count(self, tokens): return self._compute_pattern_count(tokens, EDITOR_PATTERNS)
    def _compute_tea_count(self, tokens): return self._compute_pattern_count(tokens, TEA_PATTERNS)
    def _compute_lets_go_count(self, tokens): return self._compute_pattern_count(tokens, LETS_GO_PATTERNS)
    def _compute_blitz_count(self, tokens): return self._compute_pattern_count(tokens, BLITZ_PATTERNS)
    def _compute_all_right_count(self, tokens): return self._compute_pattern_count(tokens, ALL_RIGHT_PATTERNS)
    def _compute_wa_count(self, tokens): return self._compute_pattern_count(tokens, WA_PATTERNS)
    def _compute_bro_count(self, tokens): return self._compute_pattern_count(tokens, BRO_PATTERNS)
    def _compute_bot_count(self, tokens): return self._compute_pattern_count(tokens, BOT_PATTERNS)
    def _compute_punk_count(self, tokens): return self._compute_pattern_count(tokens, PUNK_PATTERNS)
    def _compute_mr_count(self, tokens): return self._compute_pattern_count(tokens, MR_PATTERNS)
    def _compute_gay_count(self, tokens): return self._compute_pattern_count(tokens, GAY_PATTERNS)
    def _compute_pasta_count(self, tokens): return self._compute_pattern_count(tokens, PASTA_PATTERNS)
    def _compute_please_count(self, tokens): return self._compute_pattern_count(tokens, PLEASE_PATTERNS)
    def _compute_bagel_count(self, tokens): return self._compute_pattern_count(tokens, BAGEL_PATTERNS)
    def _compute_violation_count(self, tokens): return self._compute_pattern_count(tokens, VIOLATION_PATTERNS)
    def _compute_going_to_count(self, tokens): return self._compute_pattern_count(tokens, GOING_TO_PATTERNS)
    def _compute_double_modal_count(self, tokens): return self._compute_pattern_count(tokens, DOUBLE_MODAL_PATTERNS)
    def _compute_triple_auxiliary_count(self, tokens): return self._compute_pattern_count(tokens, TRIPLE_AUXILIARY_PATTERNS)
    def _compute_modal_perfect_count(self, tokens): return self._compute_pattern_count(tokens, MODAL_PERFECT_PATTERNS)
    def _compute_modal_progressive_count(self, tokens): return self._compute_pattern_count(tokens, MODAL_PROGRESSIVE_PATTERNS)
    def _compute_modal_passive_count(self, tokens): return self._compute_pattern_count(tokens, MODAL_PASSIVE_PATTERNS)
    def _compute_semi_modal_count(self, tokens): return self._compute_pattern_count(tokens, SEMI_MODAL_PATTERNS)
    def _compute_modal_negation_count(self, tokens): return self._compute_pattern_count(tokens, MODAL_NEGATION_PATTERNS)
    def _compute_conditional_modal_count(self, tokens): return self._compute_pattern_count(tokens, CONDITIONAL_MODAL_PATTERNS)
    def _compute_be_gonna_count(self, tokens): return self._compute_pattern_count(tokens, BE_GONNA_PATTERNS)
    def _compute_be_gotta_count(self, tokens): return self._compute_pattern_count(tokens, BE_GOTTA_PATTERNS)
    def _compute_stance_marker_count(self, tokens): return self._compute_pattern_count(tokens, STANCE_MARKER_PATTERNS)
    def _compute_evidential_marker_count(self, tokens): return self._compute_pattern_count(tokens, EVIDENTIAL_MARKER_PATTERNS)
    def _compute_intensifier_stack_count(self, tokens): return self._compute_pattern_count(tokens, INTENSIFIER_STACK_PATTERNS)
    def _compute_hedge_stack_count(self, tokens): return self._compute_pattern_count(tokens, HEDGE_STACK_PATTERNS)
    def _compute_tag_question_count(self, tokens): return self._compute_pattern_count(tokens, TAG_QUESTION_PATTERNS)
    def _compute_repair_restart_count(self, tokens): return self._compute_pattern_count(tokens, REPAIR_RESTART_PATTERNS)
    def _compute_disfluency_cluster_count(self, tokens): return self._compute_pattern_count(tokens, DISFLUENCY_CLUSTER_PATTERNS)
    def _compute_over_to_count(self, tokens): return self._compute_pattern_count(tokens, OVER_TO_PATTERNS)
    def _compute_brought_count(self, tokens): return self._compute_pattern_count(tokens, BROUGHT_PATTERNS)
    def _compute_took_count(self, tokens): return self._compute_pattern_count(tokens, TOOK_PATTERNS)
    def _compute_well_count(self, tokens): return self._compute_pattern_count(tokens, WELL_PATTERNS)
    def _compute_reckon_count(self, tokens): return self._compute_pattern_count(tokens, RECKON_PATTERNS)
    def _compute_i_dont_know_count(self, tokens): return self._compute_pattern_count(tokens, I_DONT_KNOW_PATTERNS)
    def _compute_you_know_what_count(self, tokens): return self._compute_pattern_count(tokens, YOU_KNOW_WHAT_PATTERNS)
    def _compute_oh_my_god_count(self, tokens): return self._compute_pattern_count(tokens, OH_MY_GOD_PATTERNS)
    def _compute_oh_my_gosh_count(self, tokens): return self._compute_pattern_count(tokens, OH_MY_GOSH_PATTERNS)
    def _compute_oh_my_goodness_count(self, tokens): return self._compute_pattern_count(tokens, OH_MY_GOODNESS_PATTERNS)
    def _compute_there_we_go_count(self, tokens): return self._compute_pattern_count(tokens, THERE_WE_GO_PATTERNS)
    def _compute_here_we_go_count(self, tokens): return self._compute_pattern_count(tokens, HERE_WE_GO_PATTERNS)
    def _compute_look_at_that_count(self, tokens): return self._compute_pattern_count(tokens, LOOK_AT_THAT_PATTERNS)
    def _compute_oh_no_count(self, tokens): return self._compute_pattern_count(tokens, OH_NO_PATTERNS)
    def _compute_im_not_a_count(self, tokens): return self._compute_pattern_count(tokens, IM_NOT_A_PATTERNS)
    def _compute_give_me_count(self, tokens): return self._compute_pattern_count(tokens, GIVE_ME_PATTERNS)
    def _compute_hate_you_count(self, tokens): return self._compute_pattern_count(tokens, HATE_YOU_PATTERNS)
    def _compute_pookie_count(self, tokens): return self._compute_pattern_count(tokens, POOKIE_PATTERNS)
    def _compute_homie_count(self, tokens): return self._compute_pattern_count(tokens, HOMIE_PATTERNS)
    def _compute_skill_issue_count(self, tokens): return self._compute_pattern_count(tokens, SKILL_ISSUE_PATTERNS)
    def _compute_sneaky_count(self, tokens): return self._compute_pattern_count(tokens, SNEAKY_PATTERNS)
    def _compute_sweat_count(self, tokens): return self._compute_pattern_count(tokens, SWEAT_PATTERNS)
    def _compute_rat_count(self, tokens): return self._compute_pattern_count(tokens, RAT_PATTERNS)
    def _compute_cringe_count(self, tokens): return self._compute_pattern_count(tokens, CRINGE_PATTERNS)
    def _compute_damage_count(self, tokens): return self._compute_pattern_count(tokens, DAMAGE_PATTERNS)
    def _compute_secret_count(self, tokens): return self._compute_pattern_count(tokens, SECRET_PATTERNS)
    def _compute_buddy_count(self, tokens): return self._compute_pattern_count(tokens, BUDDY_PATTERNS)
    def _compute_the_count(self, tokens): return self._compute_pattern_count(tokens, THE_PATTERNS)
    def _compute_murder_count(self, tokens): return self._compute_pattern_count(tokens, MURDER_PATTERNS)
    def _compute_nonchalant_count(self, tokens): return self._compute_pattern_count(tokens, NONCHALANT_PATTERNS)
    def _compute_dude_count(self, tokens): return self._compute_pattern_count(tokens, DUDE_PATTERNS)
    def _compute_cheeky_count(self, tokens): return self._compute_pattern_count(tokens, CHEEKY_PATTERNS)
    def _compute_spanner_count(self, tokens): return self._compute_pattern_count(tokens, SPANNER_PATTERNS)
    def _compute_wrench_count(self, tokens): return self._compute_pattern_count(tokens, WRENCH_PATTERNS)
    def _compute_rubbish_count(self, tokens): return self._compute_pattern_count(tokens, RUBBISH_PATTERNS)
    def _compute_burger_count(self, tokens): return self._compute_pattern_count(tokens, BURGER_PATTERNS)
    def _compute_bye_count(self, tokens): return self._compute_pattern_count(tokens, BYE_PATTERNS)
    def _compute_explosion_count(self, tokens): return self._compute_pattern_count(tokens, EXPLOSION_PATTERNS)
    def _compute_i_think_count(self, tokens): return self._compute_pattern_count(tokens, I_THINK_PATTERNS)
    def _compute_probably_count(self, tokens): return self._compute_pattern_count(tokens, PROBABLY_PATTERNS)
    def _compute_crazy_count(self, tokens): return self._compute_pattern_count(tokens, CRAZY_PATTERNS)
    def _compute_sure_count(self, tokens): return self._compute_pattern_count(tokens, SURE_PATTERNS)
    def _compute_see_you_next_time_count(self, tokens): return self._compute_pattern_count(tokens, SEE_YOU_NEXT_TIME_PATTERNS)
    def _compute_ow_count(self, tokens): return self._compute_pattern_count(tokens, OW_PATTERNS)
    def _compute_okay_so_count(self, tokens): return self._compute_pattern_count(tokens, OKAY_SO_PATTERNS)
    def _compute_dnf_count(self, tokens): return self._compute_pattern_count(tokens, DNF_PATTERNS)
    def _compute_its_okay_count(self, tokens): return self._compute_pattern_count(tokens, ITS_OKAY_PATTERNS)
    def _compute_checkpoint_count(self, tokens): return self._compute_pattern_count(tokens, CHECKPOINT_PATTERNS)
    def _compute_mate_count(self, tokens): return self._compute_pattern_count(tokens, MATE_PATTERNS)
    def _compute_bloody_count(self, tokens): return self._compute_pattern_count(tokens, BLOODY_PATTERNS)
    def _compute_loo_count(self, tokens): return self._compute_pattern_count(tokens, LOO_PATTERNS)
    def _compute_fumble_count(self, tokens): return self._compute_pattern_count(tokens, FUMBLE_PATTERNS)
    def _compute_outrageous_count(self, tokens): return self._compute_pattern_count(tokens, OUTRAGEOUS_PATTERNS)
    def _compute_embarrass_count(self, tokens): return self._compute_pattern_count(tokens, EMBARRASS_PATTERNS)
    def _compute_well_done_count(self, tokens): return self._compute_pattern_count(tokens, WELL_DONE_PATTERNS)
    def _compute_hang_on_count(self, tokens): return self._compute_pattern_count(tokens, HANG_ON_PATTERNS)
    def _compute_hold_on_count(self, tokens): return self._compute_pattern_count(tokens, HOLD_ON_PATTERNS)
    def _compute_lock_in_count(self, tokens): return self._compute_pattern_count(tokens, LOCK_IN_PATTERNS)
    def _compute_to_be_fair_count(self, tokens): return self._compute_pattern_count(tokens, TO_BE_FAIR_PATTERNS)
    def _compute_real_quick_count(self, tokens): return self._compute_pattern_count(tokens, REAL_QUICK_PATTERNS)
    def _compute_finish_line_count(self, tokens): return self._compute_pattern_count(tokens, FINISH_LINE_PATTERNS)
    def _compute_to_be_honest_count(self, tokens): return self._compute_pattern_count(tokens, TO_BE_HONEST_PATTERNS)
    def _compute_stop_count(self, tokens): return self._compute_pattern_count(tokens, STOP_PATTERNS)
    def _compute_wait_count(self, tokens): return self._compute_pattern_count(tokens, WAIT_PATTERNS)
    def _compute_what_do_you_mean_count(self, tokens): return self._compute_pattern_count(tokens, WHAT_DO_YOU_MEAN_PATTERNS)
    def _compute_son_of_a_count(self, tokens): return self._compute_pattern_count(tokens, SON_OF_A_PATTERNS)
    def _compute_son_of_a_not_before_block_count(self, tokens): return self._compute_pattern_count(tokens, SON_OF_A_NOT_BEFORE_BLOCK_PATTERNS)
    def _compute_also_count(self, tokens): return self._compute_pattern_count(tokens, ALSO_PATTERNS)
    def _compute_are_you_kidding_count(self, tokens): return self._compute_pattern_count(tokens, ARE_YOU_KIDDING_PATTERNS)
    def _compute_bad_boy_count(self, tokens): return self._compute_pattern_count(tokens, BAD_BOY_PATTERNS)
    def _compute_yoink_count(self, tokens): return self._compute_pattern_count(tokens, YOINK_PATTERNS)
    def _compute_high_five_count(self, tokens): return self._compute_pattern_count(tokens, HIGH_FIVE_PATTERNS)
    def _compute_technically_count(self, tokens): return self._compute_pattern_count(tokens, TECHNICALLY_PATTERNS)
    def _compute_nailed_it_count(self, tokens): return self._compute_pattern_count(tokens, NAILED_IT_PATTERNS)
    def _compute_jeez_geez_count(self, tokens): return self._compute_pattern_count(tokens, JEEZ_GEEZ_PATTERNS)
    def _compute_tree_word_count(self, tokens): return self._compute_pattern_count(tokens, TREE_WORD_PATTERNS)
    def _compute_so_cool_count(self, tokens): return self._compute_pattern_count(tokens, SO_COOL_PATTERNS)
    def _compute_thanks_for_watching_count(self, tokens): return self._compute_pattern_count(tokens, THANKS_FOR_WATCHING_PATTERNS)
    def _compute_go_ahead_count(self, tokens): return self._compute_pattern_count(tokens, GO_AHEAD_PATTERNS)
    def _compute_goo_goop_count(self, tokens): return self._compute_pattern_count(tokens, GOO_GOOP_PATTERNS)
    def _compute_make_sure_count(self, tokens): return self._compute_pattern_count(tokens, MAKE_SURE_PATTERNS)
    def _compute_makes_me_count(self, tokens): return self._compute_pattern_count(tokens, MAKES_ME_PATTERNS)
    def _compute_lame_count(self, tokens): return self._compute_pattern_count(tokens, LAME_PATTERNS)
    def _compute_grandma_count(self, tokens): return self._compute_pattern_count(tokens, GRANDMA_PATTERNS)
    def _compute_youre_a_little_count(self, tokens): return self._compute_pattern_count(tokens, YOURE_A_LITTLE_PATTERNS)
    def _compute_insane_count(self, tokens): return self._compute_pattern_count(tokens, INSANE_PATTERNS)
    def _compute_hypothetical_count(self, tokens): return self._compute_pattern_count(tokens, HYPOTHETICAL_PATTERNS)
    def _compute_i_feel_count(self, tokens): return self._compute_pattern_count(tokens, I_FEEL_PATTERNS)
    def _compute_i_swear_count(self, tokens): return self._compute_pattern_count(tokens, I_SWEAR_PATTERNS)
    def _compute_good_question_count(self, tokens): return self._compute_pattern_count(tokens, GOOD_QUESTION_PATTERNS)
    def _compute_uh_count(self, tokens): return self._compute_pattern_count(tokens, UH_PATTERNS)
    def _compute_um_count(self, tokens): return self._compute_pattern_count(tokens, UM_PATTERNS)
    def _compute_upgrade_count(self, tokens): return self._compute_pattern_count(tokens, UPGRADE_PATTERNS)
    def _compute_thats_okay_count(self, tokens): return self._compute_pattern_count(tokens, THATS_OKAY_PATTERNS)
    def _compute_too_count(self, tokens): return self._compute_pattern_count(tokens, TOO_PATTERNS)
    def _compute_hold_up_count(self, tokens): return self._compute_pattern_count(tokens, HOLD_UP_PATTERNS)
    def _compute_sorry_count(self, tokens): return self._compute_pattern_count(tokens, SORRY_PATTERNS)
    def _compute_hear_me_out_count(self, tokens): return self._compute_pattern_count(tokens, HEAR_ME_OUT_PATTERNS)
    def _compute_hello_count(self, tokens): return self._compute_pattern_count(tokens, HELLO_PATTERNS)
    def _compute_whole_bunch_count(self, tokens): return self._compute_pattern_count(tokens, WHOLE_BUNCH_PATTERNS)
    def _compute_put_count(self, tokens): return self._compute_pattern_count(tokens, PUT_PATTERNS)
    def _compute_take_care_count(self, tokens): return self._compute_pattern_count(tokens, TAKE_CARE_PATTERNS)
    def _compute_i_can_count(self, tokens): return self._compute_pattern_count(tokens, I_CAN_PATTERNS)
    def _compute_dog_count(self, tokens): return self._compute_pattern_count(tokens, DOG_PATTERNS)
    def _compute_cat_count(self, tokens): return self._compute_pattern_count(tokens, CAT_PATTERNS)
    def _compute_lesbian_count(self, tokens): return self._compute_pattern_count(tokens, LESBIAN_PATTERNS)
    def _compute_money_count(self, tokens): return self._compute_pattern_count(tokens, MONEY_PATTERNS)
    def _compute_moon_count(self, tokens): return self._compute_pattern_count(tokens, MOON_PATTERNS)
    def _compute_fast_food_count(self, tokens): return self._compute_pattern_count(tokens, FAST_FOOD_PATTERNS)
    def _compute_excellent_count(self, tokens): return self._compute_pattern_count(tokens, EXCELLENT_PATTERNS)
    def _compute_car_count(self, tokens): return self._compute_pattern_count(tokens, CAR_PATTERNS)
    def _compute_boat_count(self, tokens): return self._compute_pattern_count(tokens, BOAT_PATTERNS)
    def _compute_truck_count(self, tokens): return self._compute_pattern_count(tokens, TRUCK_PATTERNS)
    def _compute_plane_count(self, tokens): return self._compute_pattern_count(tokens, PLANE_PATTERNS)
    def _compute_love_you_count(self, tokens): return self._compute_pattern_count(tokens, LOVE_YOU_PATTERNS)
    def _compute_thank_you_count(self, tokens): return self._compute_pattern_count(tokens, THANK_YOU_PATTERNS)
    def _compute_foreign_count(self, tokens): return self._compute_pattern_count(tokens, FOREIGN_PATTERNS)
    def _compute_heat_count(self, tokens): return self._compute_pattern_count(tokens, HEAT_PATTERNS)
    def _compute_max_count(self, tokens): return self._compute_pattern_count(tokens, MAX_PATTERNS)
    def _compute_67_count(self, tokens): return self._compute_pattern_count(tokens, PATTERNS_67)
    def _compute_wafflestomp_count(self, tokens): return self._compute_pattern_count(tokens, WAFFLESTOMP_PATTERNS)
    def _compute_stupid_count(self, tokens): return self._compute_pattern_count(tokens, STUPID_PATTERNS)
    def _compute_idiot_count(self, tokens): return self._compute_pattern_count(tokens, IDIOT_PATTERNS)
    def _compute_block_count(self, tokens): return self._compute_pattern_count(tokens, BLOCK_PATTERNS)
    def _compute_roygbiv_count(self, tokens): return self._compute_pattern_count(tokens, ROYGBIV_PATTERNS)
    def _compute_duck_goose_count(self, tokens): return self._compute_pattern_count(tokens, DUCK_GOOSE_PATTERNS)
    def _compute_marry_count(self, tokens): return self._compute_pattern_count(tokens, MARRY_PATTERNS)
    def _compute_scary_count(self, tokens): return self._compute_pattern_count(tokens, SCARY_PATTERNS)
    def _compute_yes_count(self, tokens): return self._compute_pattern_count(tokens, YES_PATTERNS)
    def _compute_no_count(self, tokens): return self._compute_pattern_count(tokens, NO_PATTERNS)
    def _compute_chat_address_count(self, tokens): return self._compute_pattern_count(tokens, CHAT_ADDRESS_PATTERNS)
    def _compute_yes_yeah_count(self, tokens): return self._compute_pattern_count(tokens, YES_YEAH_PATTERNS)
    def _compute_no_nope_count(self, tokens): return self._compute_pattern_count(tokens, NO_NOPE_PATTERNS)
    def _compute_vomiting_count(self, tokens): return self._compute_pattern_count(tokens, VOMITING_PATTERNS)
    def _compute_bodily_humor(self, tokens): return self._compute_pattern_count(tokens, BODILY_HUMOR_PATTERNS)
    def _compute_insult_count(self, tokens): return self._compute_pattern_count(tokens, INSULT_PATTERNS)
    def _compute_actual_peak_comedy_count(self, tokens): return self._compute_pattern_count(tokens, ACTUAL_PEAK_COMEDY_PATTERNS)
    def _compute_really_actual_peak_comedy_count(self, tokens): return self._compute_pattern_count(tokens, REALLY_ACTUAL_PEAK_COMEDY_PATTERNS)

    def _compute_peak_comedy_count(self, tokens):
        if not tokens:
            return 0.0
        soft_insult_count = self._compute_insult_count(tokens)
        bodily_humor_count = self._compute_bodily_humor(tokens)
        requested30_count = self._compute_dynamic_token_count(tokens, 'requested30_count')
        return float(soft_insult_count + bodily_humor_count + requested30_count)

    def _compute_minecraft_lexicon_density(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        minecraft_terms = {
            'minecraft', 'block', 'blocks', 'iron', 'diamond', 'diamonds', 'gold', 'emerald', 'emeralds',
            'stone', 'wood', 'dirt', 'grass', 'sand', 'gravel', 'cobblestone', 'obsidian', 'bedrock',
            'netherite', 'coal', 'redstone', 'lapis', 'quartz', 'clay', 'concrete', 'glass', 'wool',
            'oak', 'spruce', 'birch', 'jungle', 'acacia', 'bamboo', 'slabs', 'stairs', 'fence',
            'pickaxe', 'sword', 'axe', 'shovel', 'hoe', 'bow', 'arrow', 'arrows', 'shield',
            'armor', 'helmet', 'chestplate', 'leggings', 'boots', 'elytra', 'trident',
            'crafting', 'mining', 'enchanting', 'enchant', 'brewing', 'smelting', 'furnace',
            'chest', 'chests', 'inventory', 'crafted', 'mending', 'sharpness', 'fortune',
            'efficiency', 'unbreaking', 'looting', 'protection', 'respawn', 'spawn',
            'creeper', 'zombie', 'zombies', 'skeleton', 'spider', 'enderman', 'ender', 'dragon',
            'wither', 'blaze', 'ghast', 'piglin', 'hoglin', 'villager', 'villagers', 'golem',
            'slime', 'phantom', 'pillager', 'ravager', 'mob', 'mobs',
            'nether', 'portal', 'overworld', 'fortress', 'stronghold', 'village', 'mansion',
            'dungeon', 'temple', 'pyramid', 'mineshaft', 'cave', 'ravine', 'canyon',
            'lava', 'water', 'bucket', 'boat', 'minecart', 'rails', 'tnt', 'explosion',
            'parkour', 'pvp', 'smp', 'server', 'realm', 'realms', 'mod', 'mods',
            'survival', 'creative', 'hardcore', 'adventure', 'spectator',
            'farm', 'farming', 'crop', 'crops', 'wheat', 'carrot', 'potato',
            'torch', 'torches', 'bread', 'steak', 'porkchop', 'apple', 'apples', 'potion', 'potions',
            'pearl', 'pearls', 'totem', 'shulker', 'hopper', 'piston', 'dispenser', 'dropper',
            'hermitcraft', 'speedrun', 'manhunt', 'uhc', 'skyblock', 'hypixel', 'bedwars',
            'skywars', 'lifesteal', 'anarchy', 'griefing', 'grief', 'raid',
            'coords', 'coordinates', 'seed', 'biome', 'chunk', 'chunks', 'render', 'fps',
            'texture', 'pack', 'shader', 'datapack', 'command', 'slash',
            'grind', 'loot', 'epic', 'clutch', 'gg', 'noob', 'pro',
            'speedbridge', 'laggy', 'lag', 'glitch'
        }
        return float(sum(1 for token in tokens if token in minecraft_terms))

    def _compute_vehicle_lexicon_density(self, tokens: List[str]) -> float:
        if not tokens:
            return 0.0
        vehicle_terms = {
            'vehicle', 'vehicles', 'car', 'cars', 'truck', 'trucks', 'bus', 'boat', 'plane',
            'helicopter', 'raft', 'ship', 'ufo', 'motorcycle', 'bike', 'scooter', 'van',
            'suv', 'sedan', 'coupe', 'convertible', 'roadster', 'supercar', 'hypercar',
            'ferrari', 'lamborghini', 'porsche', 'bugatti', 'mclaren', 'aston', 'bentley',
            'rolls', 'mercedes', 'bmw', 'audi', 'volkswagen', 'ford', 'chevrolet', 'chevy',
            'dodge', 'jeep', 'tesla', 'toyota', 'honda', 'nissan', 'mazda', 'subaru',
            'lexus', 'jaguar', 'corvette', 'mustang', 'camaro', 'challenger', 'viper',
            'engine', 'engines', 'motor', 'turbo', 'throttle', 'brakes', 'tires', 'tyre',
            'wheels', 'wheel', 'suspension', 'steering', 'fuel', 'gas', 'diesel', 'pistons',
            'piston', 'trailer', 'bumper', 'propeller', 'propellers', 'ramp', 'tail', 'wings',
            'thrusters', 'thruster', 'cannons', 'cannon', 'turret', 'bombs', 'ammo', 'hammer',
            'wedge', 'bearings', 'sensor', 'sensors', 'spud', 'kit', 'kits', 'component',
            'inventory', 'blocks', 'block', 'gearbox', 'transmission', 'clutch', 'exhaust',
            'spoiler', 'aerodynamics', 'downforce', 'diffuser', 'chassis', 'bodywork',
            'driving', 'race', 'racing', 'drift', 'drifting', 'off-road', 'highway', 'track',
            'checkpoint', 'lap', 'laps', 'qualifying', 'quali', 'podium', 'pole', 'position',
            'grid', 'pitstop', 'pit', 'pits', 'overtake', 'overtaking', 'undercut', 'overcut',
            'slipstream', 'drafting', 'apex', 'corner', 'corners', 'straight', 'straightaway',
            'chicane', 'hairpin', 'circuit', 'formula', 'f1', 'grand', 'prix', 'championship',
            'season', 'driver', 'drivers', 'team', 'teams', 'constructor', 'constructors',
            'rally', 'rallying', 'nascar', 'indycar', 'lemans', 'endurance',
            'sprint', 'timeattack', 'hotlap', 'fastest', 'record', 'dnf', 'retirement',
            'survival', 'multiplayer', 'lobby', 'scramble', 'sideways', 'jump', 'flying',
            'landing', 'spawn', 'thrust', 'backwards', 'trail', 'hover', 'underwater', 'stuck',
            'slow', 'slower', 'faster', 'speed', 'speeding', 'velocity', 'acceleration',
            'accelerate', 'decelerate', 'braking', 'challenge', 'boom', 'crash', 'crashed',
            'crashing', 'explode', 'explosion', 'exploded', 'explosive', 'broken', 'screwed',
            'suck', 'sucks', 'lag', 'hop', 'flip', 'flipped', 'spin', 'spinning', 'spinout',
            'mechanic', 'controls', 'physics', 'logic', 'creation', 'creations', 'mode',
            'workshop', 'weld', 'delete', 'attach', 'reset', 'direction',
            'turning', 'lift', 'bot', 'bots', 'makers', 'elevator', 'tuning', 'tune',
            'setup', 'settings', 'simulation'
        }
        return float(sum(1 for token in tokens if token in vehicle_terms))


def register_gaming_metrics(metric_config: Any, classifier_cls: Any) -> None:
    metric_config.METRICS.update(GAMING_METRIC_CONFIGS)

    for method_name, method in GamingMetricsMixin.__dict__.items():
        if method_name.startswith('_compute_'):
            setattr(classifier_cls, method_name, method)