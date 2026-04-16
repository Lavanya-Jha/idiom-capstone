"""
streamlit_app.py — Figurative Language Understanding (Streamlit Version)
========================================================================
Upload 5 unlabelled images + a sentence.
The model automatically:
  1. Detects if the sentence contains figurative/idiomatic language
  2. Identifies and explains the idiom meaning
  3. Detects precise objects in each image (improved CLIP zero-shot)
  4. Chains detected objects → scene → commonsense → figurative meaning
  5. Categorises each image as: Figurative / Literal / Partial Literal / Random / Distractor
  6. Ranks all images by figurative relevance

HOW TO RUN:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import os, io, pickle, re
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Figurative Language Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (cached so they load only once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, preprocess, device

@st.cache_resource
def load_phase4(device):
    """
    Load Phase 4 models. Tries gridsearch checkpoint first (most reliable),
    then falls back to MLP checkpoint. Both now use 4 signals.
    """
    model_a, model_b = None, None
    try:
        from phase4_model import Phase4CaptionFusedRanker
        models_dir = os.path.join(os.path.dirname(__file__), "models")

        def _load(ckpt_path):
            if not os.path.exists(ckpt_path):
                return None
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            head = ckpt.get("head", "mlp")
            # gridsearch checkpoints use linear head with 4 signals
            n_sig = 4 if head in ("gridsearch", "linear") else 4
            m = Phase4CaptionFusedRanker(n_signals=n_sig, head='linear' if head == 'gridsearch' else head).to(device)
            m.load_state_dict(ckpt["model_state"])
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            return m

        # Prefer gridsearch > mlp for task A
        model_a = (_load(os.path.join(models_dir, "phase4_task_a_gridsearch.pt")) or
                   _load(os.path.join(models_dir, "phase4_task_a_mlp.pt")))
        model_b = (_load(os.path.join(models_dir, "phase4_task_b_gridsearch.pt")) or
                   _load(os.path.join(models_dir, "phase4_task_b_mlp.pt")))
    except Exception:
        pass
    return model_a, model_b

@st.cache_resource
def load_conceptnet():
    pkl = os.path.join(os.path.dirname(__file__), "conceptnet", "numberbatch_en.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return None

CLIP_MODEL, PREPROCESS, DEVICE = load_clip()
PHASE4_MODEL, PHASE4_MODEL_B = load_phase4(DEVICE)
CN = load_conceptnet()

STOPWORDS = {"a","an","the","of","in","on","at","to","for","with","and",
             "or","but","is","are","was","were","i","he","she","it","we",
             "you","they","this","that","my","his","her","our","their"}

# ──────────────────────────────────────────────────────────────────────────────
# CURATED INFERENCE VOCABULARY (for ConceptNet commonsense reasoning)
# ──────────────────────────────────────────────────────────────────────────────
INFERENCE_VOCAB = [
    "anger","joy","fear","sadness","disgust","trust","surprise","anxiety",
    "stress","happiness","grief","shame","pride","guilt","love","hate",
    "envy","jealousy","hope","despair","confidence","excitement","frustration",
    "relief","disappointment","loneliness","regret","satisfaction","admiration",
    "corrupt","honest","dishonest","lazy","hardworking","clever","foolish",
    "stubborn","generous","greedy","brave","cowardly","loyal","disloyal",
    "deceptive","trustworthy","selfish","kind","cruel","innocent","naive",
    "wise","reckless","cautious","ambitious","humble","arrogant","patient",
    "impulsive","responsible","irresponsible","reliable","unreliable",
    "conflict","cooperation","betrayal","loyalty","rivalry","friendship",
    "leadership","authority","power","weakness","competition","collaboration",
    "manipulation","control","exclusion","inclusion","rejection","acceptance",
    "isolation","influence","dominance","submission","alliance","opposition",
    "struggle","compete","fight","escape","hide","reveal","succeed","fail",
    "help","harm","deceive","protect","attack","reward","punish","sacrifice",
    "risk","invest","waste","gain","lose","exploit","abuse","support","resist",
    "endure","overcome","abandon","pursue","achieve","avoid","confront",
    "success","failure","problem","solution","opportunity","threat","change",
    "growth","decay","progress","obstacle","consequence","responsibility",
    "freedom","constraint","advantage","disadvantage","burden","challenge",
    "pressure","crisis","danger","safety","loss","damage","benefit","reward",
    "justice","injustice","fairness","corruption","honesty","deception",
    "harm","good","evil","moral","unethical","punishment","integrity","sin",
    "team","group","individual","leader","follower","outsider","community",
    "society","family","relationship","partner","enemy","ally","stranger",
    "colleague","superior","subordinate","rival","victim","aggressor",
    "symbol","implication","meaning","expression","hidden","literal","irony",
    "metaphor","representation","indication","suggestion","inference",
]

_INFERENCE_VECS = {}
if CN is not None:
    for w in INFERENCE_VOCAB:
        if w in CN:
            _INFERENCE_VECS[w] = CN[w]

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPTNET RELATION CATEGORIES  (used to assign relation-type labels)
# ──────────────────────────────────────────────────────────────────────────────
_CN_EMOTION_WORDS = {
    "anger","joy","fear","sadness","disgust","trust","surprise","anxiety",
    "stress","happiness","grief","shame","pride","guilt","love","hate",
    "envy","jealousy","hope","despair","confidence","excitement","frustration",
    "relief","disappointment","loneliness","regret","satisfaction","admiration",
}
_CN_TRAIT_WORDS = {
    "corrupt","honest","dishonest","lazy","hardworking","clever","foolish",
    "stubborn","generous","greedy","brave","cowardly","loyal","disloyal",
    "deceptive","trustworthy","selfish","kind","cruel","innocent","naive",
    "wise","reckless","cautious","ambitious","humble","arrogant","patient",
    "impulsive","responsible","irresponsible","reliable","unreliable",
}
_CN_ACTION_WORDS = {
    "struggle","compete","fight","escape","hide","reveal","succeed","fail",
    "help","harm","deceive","protect","attack","reward","punish","sacrifice",
    "risk","invest","waste","gain","lose","exploit","abuse","support","resist",
    "endure","overcome","abandon","pursue","achieve","avoid","confront",
}
_CN_ABSTRACT_WORDS = {
    "success","failure","problem","solution","opportunity","threat","change",
    "growth","decay","progress","obstacle","consequence","responsibility",
    "freedom","constraint","advantage","disadvantage","burden","challenge",
    "pressure","crisis","danger","safety","loss","damage","benefit","reward",
    "justice","injustice","fairness","corruption","honesty","deception",
    "good","evil","moral","unethical","punishment","integrity","sin",
    "symbol","implication","meaning","expression","hidden","literal","irony",
    "metaphor","representation","indication","suggestion","inference",
}


def _cn_relation_label(concept):
    """Return a ConceptNet-style relation label for a vocabulary concept."""
    if concept in _CN_EMOTION_WORDS:
        return "EvokesEmotion"
    if concept in _CN_TRAIT_WORDS:
        return "HasProperty"
    if concept in _CN_ACTION_WORDS:
        return "Causes"
    if concept in _CN_ABSTRACT_WORDS:
        return "SymbolOf"
    return "RelatedTo"

# ──────────────────────────────────────────────────────────────────────────────
# IDIOM DICTIONARY
# ──────────────────────────────────────────────────────────────────────────────
IDIOM_DB = {
    "couch potato":         ("a lazy person who spends all day sitting watching TV",
                             "a lazy person lying on a sofa watching television all day",
                             "a potato vegetable sitting on a couch or sofa",
                             "coined in 1976 by Tom Iacino; combines couch-sitting and potato passivity"),
    "kick the bucket":      ("to die",
                             "a person dying or a funeral or gravestone",
                             "a person kicking a metal bucket with their foot",
                             "originates from a method of hanging; 'bucket' referred to a beam"),
    "elbow grease":         ("hard physical effort or work",
                             "a person scrubbing or cleaning very hard with great effort",
                             "a person's elbow with grease or oil on it",
                             "metaphor: elbow movement = manual scrubbing effort"),
    "night owl":            ("a person who stays up very late at night",
                             "a person awake and active late at night under the moon",
                             "an owl bird sitting on a branch at night",
                             "owls are nocturnal; associated with alertness at night"),
    "bite the bullet":      ("to endure pain or difficulty bravely",
                             "a person bravely enduring pain or hardship with determination",
                             "a person biting a bullet or cartridge with their teeth",
                             "soldiers bit bullets during surgery before anaesthesia"),
    "black sheep":          ("the odd one out; a disgrace to the family",
                             "a person standing alone and rejected from a group",
                             "a black sheep standing among white sheep in a flock",
                             "a black sheep in a flock was considered less valuable"),
    "break a leg":          ("good luck, especially before a performance",
                             "a performer on stage receiving applause and success",
                             "a person breaking their leg or a broken leg",
                             "theatre superstition: wishing bad luck brings good luck"),
    "cold turkey":          ("sudden and complete withdrawal from an addiction",
                             "a person suffering withdrawal symptoms looking pale and shaking",
                             "a cold uncooked turkey on a plate",
                             "resembles pale, cold skin of someone withdrawing"),
    "raining cats and dogs":("raining extremely heavily",
                             "a heavy downpour of rain with dark storm clouds",
                             "cats and dogs falling from the sky like rain",
                             "old Norse: cats symbolised rain, dogs symbolised wind"),
    "over the moon":        ("extremely happy and delighted",
                             "a very happy joyful person celebrating with arms raised",
                             "a person flying over or jumping over the moon in space",
                             "happiness so extreme one could float to the moon"),
    "piece of cake":        ("something very easy to do",
                             "a task being done effortlessly with ease and no difficulty",
                             "a slice of cake on a plate",
                             "eating cake requires little effort"),
    "under the weather":    ("feeling unwell or ill",
                             "a sick person lying in bed with tissues looking unwell",
                             "a person standing under dark stormy weather clouds",
                             "sailors went below deck when feeling sick in bad weather"),
    "spill the beans":      ("to accidentally reveal secret information",
                             "a person whispering or revealing a secret to someone",
                             "a bag or bowl of beans being spilled on the floor",
                             "beans used in secret voting — spilling revealed the count"),
    "on thin ice":          ("in a risky or dangerous situation",
                             "a person in a dangerous or precarious risky situation",
                             "a person walking carefully on thin cracked ice over water",
                             "thin ice can break — danger beneath"),
    "hit the nail on the head":("to be exactly right about something",
                             "a person pointing confidently having discovered the exact truth",
                             "a hammer hitting a nail precisely on its head",
                             "precise hammer strike = accurate observation"),
    "let the cat out of the bag":("to accidentally reveal a secret",
                             "a surprised person discovering a hidden secret revealed",
                             "a cat jumping out of a bag or sack",
                             "medieval fraud: cat hidden in a bag sold as a pig"),
    "burn the midnight oil": ("to work very late into the night",
                             "a person working at a desk late at night under a lamp",
                             "an oil lamp burning at midnight",
                             "before electricity, burning oil lamps meant late work"),
    "red herring":          ("a misleading clue or distraction",
                             "a person being misled or distracted from the real issue",
                             "a red fish herring being dragged across a path",
                             "strong-smelling fish used to distract hunting dogs"),
    "beat around the bush": ("to avoid the main topic or point",
                             "a person nervously talking and avoiding saying something directly",
                             "a person walking around and beating bushes",
                             "beaters drove birds from bushes — circling before hunting"),
    "break the ice":        ("to reduce awkwardness and start conversation",
                             "people meeting and starting to talk comfortably at a social event",
                             "a ship or person breaking through ice",
                             "icebreaker ships clear paths — metaphor for social ease"),
    "wolf in sheep's clothing":("a dangerous person pretending to be harmless",
                             "a dangerous deceptive person hiding their true harmful intentions",
                             "a wolf wearing a sheep fleece costume",
                             "biblical metaphor for deception"),
    "cost an arm and a leg":("extremely expensive",
                             "a very expensive luxury item with a high price tag",
                             "a person giving away their arm and leg as payment",
                             "limbs are invaluable — losing one has a great cost"),
    "sour grapes":          ("pretending not to want what you cannot have",
                             "a jealous or bitter person pretending not to care about something they want",
                             "a person eating sour unripe grapes with a grimace",
                             "Aesop's fable: fox called grapes sour after failing to reach them"),
    "storm in a teacup":    ("a lot of fuss over something very minor",
                             "people overreacting dramatically to something very small and trivial",
                             "a storm with lightning and waves inside a tiny teacup",
                             "major agitation in a tiny vessel = overreaction"),
    "barking up the wrong tree":("to pursue the wrong course of action",
                             "a person searching in completely the wrong place for an answer",
                             "a dog barking at a tree while an animal watches from another tree",
                             "hunting dog barks at wrong tree where prey is not hiding"),
    "hit the sack":         ("to go to bed and sleep",
                             "a tired person getting into bed to sleep",
                             "a person hitting or punching a large sack or bag",
                             "old mattresses were sacks filled with material"),
    "burn bridges":         ("to permanently destroy a relationship",
                             "a person leaving angrily with no intention of returning",
                             "a wooden bridge on fire burning down over water",
                             "burning a bridge prevents retreat or return"),
    "cut corners":          ("to do something poorly to save time or money",
                             "a person doing sloppy rushed work to save effort",
                             "scissors or a tool cutting the corners off a shape",
                             "taking a shorter path skips required steps"),
    "in hot water":         ("to be in trouble or a difficult situation",
                             "a person in trouble or facing a serious problem",
                             "a person sitting uncomfortably in hot boiling water",
                             "hot water causes discomfort — metaphor for difficulty"),
    "bite off more than you can chew":("to take on more than you can handle",
                             "a person overwhelmed with too many tasks and responsibilities",
                             "a person trying to fit an enormous piece of food in their mouth",
                             "chewing too large a piece leads to choking"),
    "throw in the towel":   ("to admit defeat and give up",
                             "a person giving up and accepting defeat in a competition",
                             "a white towel being thrown into a boxing ring",
                             "boxing: a trainer throws a towel to stop the fight"),
    "once in a blue moon":  ("very rarely; almost never",
                             "a very rare unusual event happening almost never",
                             "a glowing blue moon in a dark night sky",
                             "a blue moon is rare — once per 2.7 years"),
    "straight from the horse's mouth":("information from the original reliable source",
                             "a person hearing important news directly from the source",
                             "a person looking inside a horse's open mouth",
                             "checking a horse's teeth reveals its true age"),
    "two peas in a pod":    ("two people who are very similar",
                             "two identical or very similar people standing together",
                             "two green peas sitting together inside an open pod",
                             "peas in the same pod are identical"),
    "see eye to eye":       ("to agree with someone completely",
                             "two people nodding and agreeing with each other happily",
                             "two pairs of eyes looking directly at each other at the same level",
                             "looking at eye level = equality and agreement"),
    "apples and oranges":   ("two things that are too different to be compared fairly",
                             "two people trying to compare completely different and incomparable things",
                             "a red apple and an orange placed side by side",
                             "comparing fundamentally different things is as absurd as comparing fruits"),
    "bad apple":            ("one corrupt person who spoils the entire group",
                             "a troublemaker being expelled from a team while upset teammates watch",
                             "one rotten brown apple among fresh red apples",
                             "one rotten apple rots the whole barrel"),
    "baby blues":           ("feeling of sadness or depression after giving birth",
                             "a new mother feeling sad overwhelmed and tearful after having a baby",
                             "a baby wearing or surrounded by blue coloured things",
                             "postpartum sadness; blue = low mood; common in new mothers"),
    "big cheese":           ("an important and powerful person in an organisation",
                             "a self-important powerful executive commanding authority over everyone",
                             "an enormous impressive wheel of cheese on display",
                             "Urdu 'chiz' meaning thing; misinterpreted as cheese in English"),
    "big fish":             ("an important influential person especially in a small group",
                             "a dominant influential person standing out and commanding respect in a small group",
                             "a large fish swimming among much smaller fish in a pond",
                             "a big fish in a small pond has disproportionate influence"),
    "brass ring":           ("a highly desirable prize goal or opportunity",
                             "a person reaching and striving ambitiously for a coveted prize or goal",
                             "a shiny brass metal ring held up as a prize or reward",
                             "carousel riders grabbed brass rings for a free ride; became a success metaphor"),
    "bread and butter":     ("the main reliable source of income or basic necessities of life",
                             "a person doing their core everyday essential work that pays the bills",
                             "a slice of bread being spread with butter",
                             "bread and butter are the most basic staple foods — metaphor for basics"),
    "busy bee":             ("a very industrious and hardworking person always doing things",
                             "an energetic person rushing between tasks working tirelessly all day",
                             "a bee flying busily between flowers collecting pollen",
                             "bees are famously industrious creatures always working in the hive"),
    "cold feet":            ("sudden nervousness or loss of confidence before doing something",
                             "a nervous anxious person hesitating and backing away from a commitment",
                             "a person's bare feet standing in cold snow or icy water",
                             "soldiers got cold feet literally in trenches; adopted as metaphor for cowardice"),
    "copy cat":             ("a person who imitates or copies others without originality",
                             "a person shamelessly copying everything another person does or creates",
                             "a cat perfectly mimicking and copying another cat's exact actions",
                             "cats were thought to imitate; 'copy' was added to emphasise imitation"),
    "dead wood":            ("useless unproductive people or elements in an organization",
                             "unproductive lazy employees being removed from a company",
                             "dead dry wood branches on a dead tree",
                             "dead wood in a tree contributes nothing"),
    "green light":          ("official permission or approval to proceed",
                             "a person receiving approval and permission to start a project",
                             "a green traffic light signal on a road",
                             "traffic lights: green means go — adapted to mean approval to proceed"),
    "heart of gold":        ("an extremely kind generous and caring person",
                             "a genuinely warm kind-hearted person helping others selflessly and lovingly",
                             "a golden heart shape made of shining gold",
                             "gold is precious and pure; a golden heart represents pure kindness"),
    "heart of stone":       ("a cold unfeeling person with no compassion or empathy",
                             "a cold heartless person showing no emotion or compassion whatsoever",
                             "a literal stone carved into the shape of a heart",
                             "stone is hard and cold — metaphor for someone without warm feelings"),
    "hot air":              ("empty meaningless talk with no substance or truth",
                             "a politician or person talking loudly while saying nothing of substance",
                             "hot steam or air rising from a vent or hot air balloon",
                             "inflated speech like a hot air balloon — full of air but no substance"),
    "hot potato":           ("a controversial sensitive issue that nobody wants to deal with",
                             "politicians and officials desperately passing a controversial problem to each other",
                             "a person struggling to hold a steaming hot potato that burns their hands",
                             "a hot potato burns if held too long — no one wants to handle it"),
    "ivory tower":          ("a state of privileged isolation from practical reality",
                             "an academic or intellectual isolated from real world problems in luxury",
                             "a tall elegant tower built from white ivory material",
                             "from the Bible Song of Solomon; adopted for academic detachment from reality"),
    "low-hanging fruit":    ("an easy task or goal that can be achieved with minimal effort",
                             "a person easily picking the simplest most accessible tasks first",
                             "fruit hanging low on a tree branch easy to reach and pick",
                             "fruit at the bottom is easiest to harvest; metaphor for easy wins"),
    "monkey business":      ("silly mischievous behaviour or dishonest suspicious activity",
                             "people engaged in suspicious sneaky or mischievous dishonest activity",
                             "monkeys causing chaos and mischief playing pranks in a comical scene",
                             "monkeys are associated with mischief and unpredictable playful behaviour"),
    "nest egg":             ("money saved for the future especially for retirement",
                             "a person carefully saving money for a secure comfortable future",
                             "a real egg sitting safely in a bird's nest",
                             "farmers put fake eggs in nests to encourage hens to lay more; savings metaphor"),
    "old flame":            ("a former romantic partner from the past",
                             "two former lovers unexpectedly meeting again after many years apart",
                             "an old burning candle flame or an aged fire burning low",
                             "flame = passion and romance; old flame = a past romantic relationship"),
    "open book":            ("a person who is completely honest and easy to understand",
                             "a completely transparent honest person with nothing to hide from anyone",
                             "a book lying open with all its pages and words fully visible",
                             "an open book reveals everything; no secrets are hidden"),
    "party animal":         ("a person who loves socialising and partying enthusiastically",
                             "an energetic enthusiastic person dancing and celebrating at a wild party",
                             "a wild animal dressed up or attending a party celebration",
                             "combines the social joy of a party with the wildness of an animal"),
    "pipe dream":           ("a hope or plan that is completely impossible to achieve",
                             "a person daydreaming about a completely unrealistic impossible fantasy",
                             "a person smoking a pipe while lost in an elaborate daydream",
                             "opium pipe hallucinations were vivid but unreal — metaphor for impossible fantasies"),
    "rat race":             ("the exhausting competitive struggle of modern working life",
                             "exhausted office workers competing desperately in a stressful never-ending routine",
                             "rats running frantically and competitively around a maze or race track",
                             "rats in laboratory mazes run endlessly without reward — metaphor for futile competition"),
    "red flag":             ("a warning sign that something is wrong or dangerous",
                             "a person noticing a serious warning sign that something is very wrong",
                             "a bright red flag being waved or raised as a signal",
                             "red flags historically warned of danger; now used for warning signs in relationships"),
    "silver bullet":        ("a simple magical solution to a complex problem",
                             "a single perfect solution that easily solves a difficult problem",
                             "a silver-coloured bullet or ammunition",
                             "werewolves killed by silver bullets; metaphor for perfect fix"),
    "snake in the grass":   ("a treacherous person who hides their true dangerous intentions",
                             "a smiling friendly person secretly plotting betrayal against their trusting friend",
                             "a dangerous snake hidden and camouflaged in tall grass",
                             "Virgil's Aeneid: a snake lurking in grass strikes unexpectedly"),
    "swan song":            ("a final performance, effort or work before ending",
                             "a performer giving their final farewell performance on stage",
                             "a white swan opening its beak and singing",
                             "swans believed to sing beautifully just before dying"),
    "top dog":              ("the most powerful or dominant person in a group",
                             "a confident dominant leader commanding authority and respect from everyone",
                             "a dog standing proudly above other dogs asserting dominance",
                             "in dogfighting the winning dog was on top; adapted to mean leader"),
    "trojan horse":         ("a deceptive strategy that appears helpful but causes harm",
                             "something appearing friendly or useful that is actually a trap",
                             "a large wooden horse sculpture outside ancient Troy city walls",
                             "Greeks hid inside a wooden horse to infiltrate Troy"),
    "two-way street":       ("a situation that requires equal effort or contribution from both sides",
                             "two people in a relationship both giving and taking equally",
                             "a road with traffic flowing in both directions simultaneously",
                             "a two-way street allows movement in both directions — both parties must contribute"),
    "wet blanket":          ("a person who dampens others enthusiasm and ruins their fun",
                             "a gloomy complaining person ruining the enjoyment and mood of a happy group",
                             "a person wrapped in a soaking wet heavy blanket looking miserable",
                             "a wet blanket smothers fire; a wet blanket person smothers enthusiasm"),
    "white elephant":       ("a possession that is useless costly and difficult to get rid of",
                             "a person stuck with a large useless expensive possession they cannot get rid of",
                             "a rare albino white elephant being presented as a gift",
                             "Siamese kings gifted sacred white elephants to troublesome nobles; upkeep ruined them"),
    "zebra crossing":       ("a pedestrian crossing marked with black and white stripes",
                             "pedestrians safely crossing a road at a striped pedestrian crossing",
                             "a zebra animal standing on a road crossing",
                             "the black and white stripes resemble a zebra's markings"),
    "peas in a pod":        ("two people who are very similar or always together",
                             "two people who are identical in personality and always inseparable",
                             "two green peas sitting together inside an open pea pod",
                             "peas in the same pod are indistinguishable from each other"),
    "snail mail":           ("traditional slow postal mail as opposed to email",
                             "a slow postal worker delivering letters very slowly compared to email",
                             "a snail carrying a letter or envelope on its shell",
                             "contrasts the slowness of a snail with the speed of email"),
    "watering hole":        ("a bar pub or place where people regularly gather to drink",
                             "friends gathering and drinking together at a busy local pub or bar",
                             "wild animals gathering at a waterhole in the savanna to drink",
                             "animals gather at watering holes to drink — metaphor for social drinking spots"),
    "guinea pig":           ("a person used as a test subject for an experiment",
                             "a person being tested or experimented on in a laboratory",
                             "a guinea pig animal in a laboratory or cage",
                             "guinea pigs were used in lab experiments"),
    "ghost town":           ("a once-busy place that is now completely empty and abandoned",
                             "an empty abandoned town with deserted streets and boarded-up buildings",
                             "a town full of ghosts or a spooky haunted abandoned settlement",
                             "towns deserted after mining booms ended; became eerie ghost-like places"),
    "white whale":          ("an obsessive unattainable goal that consumes someone",
                             "a person obsessively and desperately chasing an impossible dream",
                             "a large white sperm whale swimming in the ocean",
                             "from Moby Dick: Captain Ahab's obsessive pursuit"),
    "act of god":           ("an uncontrollable natural disaster or event",
                             "a massive storm flood earthquake or natural disaster destroying everything",
                             "a deity or divine figure performing a miraculous act from the sky",
                             "legal term for catastrophic events beyond human control"),
    "smoking gun":          ("clear and undeniable proof of guilt or wrongdoing",
                             "a detective holding up undeniable evidence that proves someone's guilt",
                             "a gun with smoke still rising from its barrel after being fired",
                             "a recently fired gun proves someone just shot it — irrefutable evidence"),
}

def lookup_idiom(sentence):
    s = sentence.lower()
    best, best_len = None, 0
    for idiom in IDIOM_DB:
        if idiom in s and len(idiom) > best_len:
            best, best_len = idiom, len(idiom)
    if best:
        entry = IDIOM_DB[best]
        meaning, vis_fig, vis_lit, origin = entry
        return best, meaning, origin, vis_fig, vis_lit
    return None, None, None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# OBJECT DETECTION: CLIP-based zero-shot (fast, no extra model download)
# ──────────────────────────────────────────────────────────────────────────────
# Uses the already-loaded CLIP model to compare image embeddings against
# candidate object labels — very fast since CLIP is already in memory.

# Comprehensive candidate labels — OWL-ViT checks the image for each of these.
OWL_CANDIDATES = [
    # People
    "person", "child", "boy", "girl", "man", "woman", "baby",
    # Animals
    "dog", "cat", "bird", "horse", "owl", "sheep", "fish", "wolf",
    "snake", "bear", "cow", "duck", "swan", "monkey", "bee", "rat",
    "donkey", "chicken", "pig", "frog", "butterfly",
    # Tools & implements
    "axe", "hatchet", "pickaxe", "shovel", "hammer", "saw", "wrench",
    "screwdriver", "scissors", "knife", "broom", "brush",
    "ladder", "rope", "chain", "bucket", "basket",
    # Furniture & household
    "stool", "chair", "table", "bed", "sofa", "lamp", "candle",
    "mirror", "clock", "book", "cup", "plate", "bowl", "bottle",
    "glass", "vase",
    # Technology
    "light bulb", "computer", "phone", "television", "camera",
    "microphone", "headphones",
    # Food & drink
    "apple", "orange", "banana", "cake", "bread", "egg",
    "soup", "coffee", "ice cream", "candy", "fruit", "vegetable",
    "cheese", "strawberry", "dessert",
    # Nature
    "tree", "flower", "grass", "fire", "ice", "snow", "rock",
    "mountain", "cloud", "sun", "moon",
    # Clothing & accessories
    "hat", "glasses", "shoes", "bag", "umbrella", "crown", "mask",
    # Symbolic
    "medal", "trophy", "money", "coins", "flag", "gift box", "balloon", "heart",
    # Vehicles
    "car", "bicycle", "boat", "airplane", "train",
    # Buildings
    "house", "building", "bridge", "fence", "tower", "wall",
    # Weapons
    "sword", "gun", "shield", "arrow",
    # Musical instruments
    "guitar", "piano", "drum", "violin",
    # Sports
    "ball",
]

# Scene vocab — CLIP-based high-level scene/action detection
SCENE_VOCAB = [
    "a person working very hard",
    "someone helping another person",
    "a group of people cooperating",
    "a person feeling exhausted or overwhelmed",
    "someone feeling happy and joyful",
    "a person in a difficult or dangerous situation",
    "a competitive or rivalry situation",
    "a peaceful and relaxing scene",
    "someone being deceptive or sneaky",
    "a person being ignored or excluded",
    "someone achieving success or victory",
    "a chaotic or disorganised scene",
    "a person being lazy or inactive",
    "a risky or threatening situation",
    "a person feeling anxious or worried",
    "someone being generous or kind",
    "a person feeling isolated or alone",
    "people arguing or in conflict",
    "a celebratory or festive scene",
    "someone comparing two different things",
    "children playing together",
    "a sad or emotional scene",
    "a person standing on a stool or elevated surface",
    "two people doing different tasks",
]

@st.cache_resource
def encode_scene_vocab():
    with torch.no_grad():
        scene_tokens = clip.tokenize(SCENE_VOCAB, truncate=True).to(DEVICE)
        scene_embs = CLIP_MODEL.encode_text(scene_tokens)
        scene_embs = scene_embs / scene_embs.norm(dim=-1, keepdim=True)
    return scene_embs

SCENE_EMBEDDINGS = encode_scene_vocab()


@st.cache_resource
def encode_object_vocab():
    """Pre-encode all candidate object labels with CLIP (runs once at startup)."""
    prompts = [f"a photo of a {lbl}" for lbl in OWL_CANDIDATES]
    # CLIP tokenizer accepts max 77 tokens; truncate to be safe
    with torch.no_grad():
        tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
        embs = CLIP_MODEL.encode_text(tokens)
        embs = embs / embs.norm(dim=-1, keepdim=True)
    return embs  # shape: (N_labels, D)

OBJECT_EMBEDDINGS = encode_object_vocab()


def detect_objects(pil_image, top_k=6, threshold=0.24):
    """
    Fast CLIP-based object detection.
    Encodes the image and compares against pre-encoded object label embeddings.
    threshold=0.24 is calibrated for CLIP cosine similarity scores.
    """
    img_tensor = PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_emb = CLIP_MODEL.encode_image(img_tensor)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    # Cosine similarities: shape (N_labels,)
    sims = (OBJECT_EMBEDDINGS @ img_emb.T).squeeze(-1)
    top_indices = sims.argsort(descending=True)
    results = []
    for idx in top_indices:
        score = float(sims[idx])
        if score >= threshold:
            results.append((OWL_CANDIDATES[idx], round(score, 4)))
        if len(results) >= top_k:
            break
    return results


def detect_scene(img_emb, top_k=3, threshold=0.22):
    """Return top scene/action descriptors using CLIP (good for high-level scenes)."""
    sims = (SCENE_EMBEDDINGS @ img_emb.unsqueeze(-1)).squeeze(-1)
    topk = sims.topk(min(top_k, len(SCENE_VOCAB)))
    results = []
    for j, i in enumerate(topk.indices.tolist()):
        score = float(topk.values[j])
        if score >= threshold:
            results.append((SCENE_VOCAB[i], round(score, 4)))
    return results



# ──────────────────────────────────────────────────────────────────────────────
# CLIP ENCODE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def encode_text(text):
    tok = clip.tokenize([text], truncate=True).to(DEVICE)
    with torch.no_grad():
        f = CLIP_MODEL.encode_text(tok)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0)

def encode_image(pil_img):
    t = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = CLIP_MODEL.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# IAPD PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
def iapd_prompts(sentence, idiom=None, vis_fig=None, vis_lit=None, idiom_meaning=None):
    phrase = idiom if idiom else sentence
    words = [w.strip(".,!?\"'") for w in phrase.lower().split() if w.lower() not in STOPWORDS]

    if vis_lit:
        literal = vis_lit
    elif len(words) >= 2:
        literal = f"a photo showing {' and '.join(words[:3])}"
    else:
        literal = f"a photo of {phrase}"

    if vis_fig:
        figurative = vis_fig
    else:
        figurative = f"an image representing the figurative meaning of '{phrase}'"

    if idiom and idiom_meaning and idiom in sentence.lower():
        contextual = sentence.lower().replace(idiom, idiom_meaning)
    else:
        contextual = sentence

    return literal, figurative, contextual


# ──────────────────────────────────────────────────────────────────────────────
# CONCEPTNET REASONING
# ──────────────────────────────────────────────────────────────────────────────
def cn_word_associations(words, top_n=5):
    if CN is None or not _INFERENCE_VECS:
        return {}
    results = {}
    for word in words:
        w = word.lower().strip(".,!?\"'s")
        if w not in CN:
            continue
        wvec = CN[w]
        scores = {}
        for k, kvec in _INFERENCE_VECS.items():
            if k == w:
                continue
            s = float(np.dot(wvec, kvec) /
                      (np.linalg.norm(wvec) * np.linalg.norm(kvec) + 1e-8))
            scores[k] = s
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        if top:
            results[word] = [k for k, _ in top]
    return results


def build_cn_relation_chains(idiom_phrase, top_n=5):
    """
    For each content word in the idiom phrase, find top-N semantically related
    INFERENCE_VOCAB concepts using NumberBatch cosine similarity, and assign a
    ConceptNet-style relation label to each concept.

    Returns:
        dict: {word: [(concept, relation_label), ...]}

    Example output for "kick the bucket":
        {
          "kick":   [("struggle", "Causes"), ("fight", "Causes"), ("confront", "Causes"),
                     ("harm", "SymbolOf"), ("resistance", "RelatedTo")],
          "bucket": [("loss", "SymbolOf"), ("decay", "SymbolOf"), ("harm", "SymbolOf"),
                     ("death", "SymbolOf"), ("danger", "SymbolOf")],
        }
    """
    if CN is None or not _INFERENCE_VECS:
        return {}

    words = [w.lower().strip(".,!?\"'s") for w in idiom_phrase.split()]
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    chains = {}
    for word in content_words:
        if word not in CN:
            continue
        wvec = CN[word]
        scores = {}
        for k, kvec in _INFERENCE_VECS.items():
            if k == word:
                continue
            s = float(np.dot(wvec, kvec) /
                      (np.linalg.norm(wvec) * np.linalg.norm(kvec) + 1e-8))
            scores[k] = s
        top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        if top:
            chains[word] = [(_concept, _cn_relation_label(_concept)) for _concept, _ in top]

    return chains


def build_cn_assembled_sentence(idiom_phrase, chains, idiom_meaning=None):
    """
    Assemble a human-readable interpretation sentence from the relation chains.

    Example:
        "kick the bucket" → kick Causes struggle & fight;
        bucket SymbolOf loss & decay → meaning: to die
    """
    if not chains:
        return None

    parts = []
    for word, relations in chains.items():
        # Group by relation type
        by_type = {}
        for concept, rel in relations:
            by_type.setdefault(rel, []).append(concept)
        for rel, concepts in by_type.items():
            concs = " & ".join(concepts[:2])
            parts.append(f"{word} {rel} {concs}")

    sentence = "; ".join(parts)
    if idiom_meaning:
        sentence += f" → meaning: \"{idiom_meaning}\""
    return sentence


def find_idiom_candidates(sentence, exclude=None, top_k=3):
    s_emb = encode_text(sentence)
    scored = []
    for phrase, entry in IDIOM_DB.items():
        if exclude and phrase == exclude:
            continue
        meaning = entry[0]
        m_emb = encode_text(meaning)
        score = float((s_emb * m_emb).sum())
        scored.append((phrase, meaning, score))
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# IMPROVED COMMONSENSE REASONING — Object Chain → Figurative Meaning
# ──────────────────────────────────────────────────────────────────────────────
def build_object_chain_reasoning(objects, scenes, idiom, idiom_meaning, idiom_origin,
                                  sentence, best_result, all_results, perspectives,
                                  gap_weight, scoring_method, known_idiom=False):
    """
    Build commonsense reasoning by chaining detected objects into a narrative
    that connects to the figurative meaning.

    Example flow:
        Detected: [child, stool, light bulb, pickaxe]
        Scene: "children playing together"
        Chain: child + stool + light bulb → a child standing on a stool reaching
               for something, with another child holding a tool →
               two children engaged in different activities side by side →
               this represents comparing different approaches/things →
               idiom "apples and oranges" = comparing incomparable things
    """
    obj_labels = [o for o, _ in objects] if objects else []
    scene_labels = [(s, sc) for s, sc in scenes] if scenes else []

    reasoning = {}

    # ── Section 1: Visual Evidence ──
    reasoning["visual_evidence"] = {
        "objects": [(o, s) for o, s in objects] if objects else [],
        "scenes": scene_labels,
    }

    # ── Section 2: Scene Description ──
    if obj_labels:
        obj_str = ", ".join(obj_labels[:-1]) + (" and " + obj_labels[-1] if len(obj_labels) > 1 else obj_labels[0])
        scene_desc = f"The image contains: {obj_str}."
        if scene_labels:
            scene_desc += f" The overall scene suggests: {scene_labels[0][0]}."
    else:
        # Fix 2: show scene when no objects detected instead of "No objects above threshold"
        if scene_labels:
            scene_desc = f"Scene detected: {scene_labels[0][0]}."
            if len(scene_labels) > 1:
                scene_desc += f" Also: {scene_labels[1][0]}."
        else:
            scene_desc = "No specific objects or scene detected for this image."
    reasoning["scene_description"] = scene_desc

    # ── Section 3: Idiom-driven Object Chain ──
    # Fix 3: Build the chain from the IDIOM meaning outward through visual evidence,
    # rather than from objects inward. This grounds the reasoning in the actual
    # figurative interpretation rather than just listing what CLIP detected.
    chain_steps = []

    # Determine primary visual evidence (objects or scene)
    visual_evidence = obj_labels[:4] if obj_labels else []
    primary_scene   = scene_labels[0][0] if scene_labels else None

    if idiom and idiom_meaning:
        # Step 1: State the identified idiom and its meaning
        chain_steps.append(
            f"**Identified idiom:** \"{idiom}\" — meaning: \"{idiom_meaning}\""
        )

        # Step 2: State what we see in the image
        if visual_evidence:
            chain_steps.append(
                f"**Visual elements detected:** {', '.join(visual_evidence)}"
            )
        elif primary_scene:
            chain_steps.append(
                f"**Scene detected:** {primary_scene}"
            )

        # Step 3: ConceptNet associations linking objects → idiom meaning words
        if visual_evidence:
            obj_assocs = cn_word_associations(visual_evidence[:3])
            meaning_words = [w.strip(".,;") for w in idiom_meaning.lower().split()
                             if w not in STOPWORDS and len(w) > 3]
            for obj, neighbours in obj_assocs.items():
                overlap = [n for n in neighbours[:6]
                           if any(m in n or n in m for m in meaning_words)]
                if overlap:
                    chain_steps.append(
                        f"**{obj.capitalize()}** → associated with: {', '.join(neighbours[:4])} "
                        f"→ connects to \"{idiom_meaning}\" via: {', '.join(overlap[:2])}"
                    )
                else:
                    chain_steps.append(
                        f"**{obj.capitalize()}** → associated with: {', '.join(neighbours[:4])}"
                    )

        # Step 4: Scene → meaning bridge
        if primary_scene:
            chain_steps.append(
                f"**Scene → meaning:** \"{primary_scene}\" "
                f"→ this visually represents someone who is \"{idiom_meaning}\""
            )

        # Step 5: Final reasoning chain — idiom-grounded narrative
        # Build the chain: visual elements → scene concept → idiom meaning
        chain_parts = []
        if visual_evidence:
            chain_parts.extend(visual_evidence[:2])
        if primary_scene:
            # Condense scene to a short concept
            scene_concept = primary_scene.replace("a person ", "").replace("someone ", "").strip()
            scene_concept = scene_concept[:50] + "…" if len(scene_concept) > 50 else scene_concept
            chain_parts.append(scene_concept)
        chain_parts.append(idiom_meaning)
        chain_parts.append(f'"{idiom}"')

        if chain_parts:
            chain_steps.append(
                f"**Reasoning chain:** " + " → ".join(chain_parts)
            )

    else:
        # No idiom — general figurative analysis from scene/objects
        if visual_evidence:
            chain_steps.append(f"**Observed elements:** {', '.join(visual_evidence)}")
            obj_assocs = cn_word_associations(visual_evidence[:3])
            for obj, neighbours in obj_assocs.items():
                chain_steps.append(f"**{obj.capitalize()}** is associated with: {', '.join(neighbours[:4])}")
        if primary_scene:
            chain_steps.append(f"**Scene interpretation:** {primary_scene}")
        if visual_evidence or primary_scene:
            evidence = " → ".join(visual_evidence[:2]) if visual_evidence else primary_scene
            chain_steps.append(
                f"**Reasoning chain:** {evidence} → these elements suggest "
                f"the figurative meaning of the sentence."
            )

    reasoning["object_chain"] = chain_steps

    # ── Section 4: Idiom Match ──
    reasoning["idiom_match"] = {
        "idiom": idiom,
        "meaning": idiom_meaning,
        "origin": idiom_origin,
        "known": known_idiom,
        "figurative_probe": perspectives[1] if perspectives else None,
    }

    # ── Section 5: Score Breakdown ──
    fig_sc = best_result.get("fig_score", 0)
    lit_sc = best_result.get("lit_score", 0)
    ctx_sc = best_result.get("ctx_score", 0)
    gap = best_result.get("gap", 0)
    flgs = best_result.get("flgs", 0)
    category = best_result.get("category", "Unknown")

    reasoning["score_breakdown"] = {
        "category": category,
        "fig_score": fig_sc,
        "lit_score": lit_sc,
        "ctx_score": ctx_sc,
        "gap": gap,
        "flgs": flgs,
        "scoring_method": scoring_method,
        "gap_interpretation": "supports figurative" if gap > 0 else "supports literal",
    }

    # ── Section 6: Rejected Candidates ──
    candidates = find_idiom_candidates(sentence, exclude=idiom, top_k=3)
    reasoning["rejected_candidates"] = [
        {"idiom": phrase, "meaning": meaning, "score": score}
        for phrase, meaning, score in candidates
    ]

    # ── Section 7: Final Conclusion ──
    if idiom and idiom_meaning:
        reasoning["conclusion"] = (
            f"The sentence uses the idiom \"{idiom}\" to express: "
            f"**{idiom_meaning}**. "
            f"The top-ranked image (classified as {category}) visually grounds "
            f"this figurative meaning with a confidence of "
            f"{best_result.get('confidence_pct', 0):.1f}%."
        )
    else:
        reasoning["conclusion"] = (
            f"The sentence is treated as a general figurative expression. "
            f"The top-ranked image ({category}) shows the strongest visual alignment."
        )

    return reasoning


# ──────────────────────────────────────────────────────────────────────────────
# AUTO-CATEGORISATION
# ──────────────────────────────────────────────────────────────────────────────
def auto_categorise(results):
    """
    Categorise each image as Figurative / Literal / Partial Literal / Random.

    Logic:
      1. Figurative  — highest FLGS score AND positive gap (fig > lit)
      2. Literal     — highest lit_score AND negative gap (lit > fig)
      3. Partial Literal — among remaining, those whose lit_score is ABOVE
                          the mean lit_score of all images (genuinely related
                          to the literal scene, just not the best match)
      4. Random      — images with low lit_score relative to the group
                       (semantically unrelated to both prompts)

    Key fix: previously "Partial Literal" was assigned by smallest |gap|,
    which caused a bird/nest image (low lit_score) to be called "Partial
    Literal" while all-white-sheep (high lit_score) was called "Random".
    Now we use absolute lit_score magnitude to distinguish genuine partial
    matches from unrelated distractors.
    """
    n = len(results)
    by_flgs = sorted(range(n), key=lambda i: results[i]["flgs"], reverse=True)
    by_lit  = sorted(range(n), key=lambda i: results[i]["lit_score"], reverse=True)

    assigned = {}
    used_slots = set()

    def assign(idx, cat):
        assigned[idx] = cat
        used_slots.add(idx)

    # 1. Figurative = highest FLGS with positive gap
    pos_gap = [i for i in by_flgs if results[i]["gap"] > 0]
    fig_candidates = pos_gap if pos_gap else by_flgs
    for i in fig_candidates:
        if i not in used_slots:
            assign(i, "Figurative")
            break

    # 2. Literal = highest lit_score with negative gap (lit_score > fig_score)
    neg_gap = [i for i in by_lit if results[i]["gap"] < 0]
    lit_candidates = neg_gap if neg_gap else by_lit
    for i in lit_candidates:
        if i not in used_slots:
            assign(i, "Literal")
            break

    # 3 & 4. Partial Literal vs Random — use lit_score as the discriminator.
    # Images whose lit_score is above the global mean are genuinely related to
    # the literal scene (even if imperfect). Below-mean images are distractors.
    mean_lit = sum(r["lit_score"] for r in results) / max(n, 1)

    remaining = [i for i in range(n) if i not in used_slots]
    # Sort remaining by lit_score descending so we assign Partial Literal
    # to the highest-lit-score ones first
    remaining_by_lit = sorted(remaining, key=lambda i: results[i]["lit_score"], reverse=True)

    for i in remaining_by_lit:
        if results[i]["lit_score"] >= mean_lit:
            assign(i, "Partial Literal")
        else:
            assign(i, "Random")

    # Ensure no image is left unassigned
    for i in range(n):
        if i not in used_slots:
            assign(i, "Random")

    for i, r in enumerate(results):
        r["category"] = assigned.get(i, "Unknown")
    return results


CATEGORY_COLORS = {
    "Figurative":      "#2e7d32",
    "Literal":         "#e65100",
    "Partial Literal": "#1565c0",
    "Random":          "#6d4c41",
}


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def run_analysis(sentence, images):
    """Main analysis pipeline. Takes sentence + list of 5 PIL images."""

    # Idiom lookup
    idiom, idiom_meaning, idiom_origin, vis_fig, vis_lit = lookup_idiom(sentence)

    # Figurative detection
    fig_probes = [
        "This sentence uses figurative, idiomatic or metaphorical language.",
        "This is an idiomatic expression with a non-literal meaning.",
        "This sentence contains a figure of speech or idiom.",
    ]
    lit_probes = [
        "This sentence is literal and straightforward with no idioms.",
        "This sentence means exactly what it says with no figurative language.",
        "This is a plain factual statement with no metaphors.",
    ]
    s_emb = encode_text(sentence)
    fs = float(sum((s_emb * encode_text(p)).sum().item() for p in fig_probes) / len(fig_probes))
    ls = float(sum((s_emb * encode_text(p)).sum().item() for p in lit_probes) / len(lit_probes))
    is_fig = fs > ls
    diff = fs - ls
    conf = float(min(abs(diff) * 20, 1.0))

    # IAPD perspectives
    perspectives = iapd_prompts(sentence, idiom, vis_fig, vis_lit, idiom_meaning)
    lit_emb = encode_text(perspectives[0])
    fig_emb = encode_text(perspectives[1])
    ctx_emb = encode_text(perspectives[2])

    gap_weight = 0.40 if (vis_fig is not None) else 0.15

    # Per-image scoring
    img_embs = []
    results = []
    for i, pil_img in enumerate(images):
        img_emb = encode_image(pil_img)
        img_embs.append(img_emb)

        ctx_sc = float((ctx_emb * img_emb).sum())
        fig_sc = float((fig_emb * img_emb).sum())
        lit_sc = float((lit_emb * img_emb).sum())
        gap = fig_sc - lit_sc
        flgs = ctx_sc + gap_weight * gap

        objects = detect_objects(pil_img, top_k=6)   # OWL-ViT needs PIL image
        scenes = detect_scene(img_emb, top_k=3)       # CLIP for scene-level

        results.append({
            "index": i,
            "flgs": flgs,
            "ctx_score": ctx_sc,
            "fig_score": fig_sc,
            "lit_score": lit_sc,
            "gap": gap,
            "objects": objects,
            "scenes": scenes,
            "category": "",
        })

    # Phase 4 override — uses 4 discriminative signals (ctx→img, fig→img, lit→img, gap)
    scoring_method = "FLGS zero-shot"
    if PHASE4_MODEL is not None and len(img_embs) == 5:
        try:
            imgs_t   = torch.stack(img_embs).unsqueeze(0)          # (1, 5, 512)
            ctx_exp  = ctx_emb.unsqueeze(0).unsqueeze(0)            # (1, 1, 512)
            fig_exp  = fig_emb.unsqueeze(0).unsqueeze(0)            # (1, 1, 512)
            lit_exp  = lit_emb.unsqueeze(0).unsqueeze(0)            # (1, 1, 512)

            s_ctx_img = (ctx_exp * imgs_t).sum(dim=-1)              # (1, 5)
            s_fig_img = (fig_exp * imgs_t).sum(dim=-1)
            s_lit_img = (lit_exp * imgs_t).sum(dim=-1)
            gap_t     = s_fig_img - s_lit_img

            signals = torch.stack([
                s_ctx_img, s_fig_img, s_lit_img, gap_t
            ], dim=-1)                                               # (1, 5, 4)

            with torch.no_grad():
                p4_scores = PHASE4_MODEL(signals).squeeze(0)

            flgs_raw = torch.tensor([r["flgs"] for r in results])
            flgs_norm = F.normalize(flgs_raw.unsqueeze(0), dim=-1).squeeze(0)
            p4_norm = F.normalize(p4_scores.unsqueeze(0), dim=-1).squeeze(0)
            blended = 0.6 * p4_norm + 0.4 * flgs_norm

            for i, r in enumerate(results):
                r["flgs"] = float(blended[i])
            scoring_method = "Phase 4 CaptionFused + FLGS blend"
        except Exception:
            pass

    # Auto-categorise
    results = auto_categorise(results)

    # Rank
    results.sort(key=lambda x: x["flgs"], reverse=True)
    raw = torch.tensor([r["flgs"] for r in results])
    probs = F.softmax(raw * 10, dim=0).tolist()
    for rank, (r, p) in enumerate(zip(results, probs)):
        r["rank"] = rank + 1
        r["confidence_pct"] = round(p * 100, 1)

    # Build reasoning
    best_result = results[0] if results else {}
    reasoning = build_object_chain_reasoning(
        objects=best_result.get("objects", []),
        scenes=best_result.get("scenes", []),
        idiom=idiom,
        idiom_meaning=idiom_meaning,
        idiom_origin=idiom_origin,
        sentence=sentence,
        best_result=best_result,
        all_results=results,
        perspectives=perspectives,
        gap_weight=gap_weight,
        scoring_method=scoring_method,
        known_idiom=(vis_fig is not None),
    )

    return {
        "is_figurative": is_fig,
        "confidence": conf,
        "idiom": idiom,
        "idiom_meaning": idiom_meaning,
        "idiom_origin": idiom_origin,
        "perspectives": perspectives,
        "results": results,
        "reasoning": reasoning,
        "scoring_method": scoring_method,
    }




# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI — plain, no custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.title("Figurative Language Understanding")
st.caption("Visual Grounding and Commonsense Reasoning for Multimodal Figurative Language")

# Model status
cn_status   = "✅ loaded" if CN is not None else "❌ not found"
p4a_status  = "✅ loaded (73.3% val)" if PHASE4_MODEL  is not None else "⬜ not found — using FLGS zero-shot"
p4b_status  = "✅ loaded (80.0% val)" if PHASE4_MODEL_B is not None else "⬜ not found"
with st.expander("Model Status"):
    st.write(f"**CLIP ViT-B/32:** ✅ loaded")
    st.write(f"**ConceptNet:** {cn_status}")
    st.write(f"**Phase 4 Task A:** {p4a_status}")
    st.write(f"**Phase 4 Task B:** {p4b_status}")

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────
st.subheader("Input")

sentence = st.text_input(
    "Sentence (must contain an idiom):",
    value="Comparing the two job offers is like comparing apples and oranges; they have different salary structures and benefits.",
)

uploaded_files = st.file_uploader(
    "Select all 5 images at once (hold Ctrl/Cmd to select multiple):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

pil_images = []
if uploaded_files:
    if len(uploaded_files) != 5:
        st.warning(f"Please select exactly 5 images. You selected {len(uploaded_files)}.")
    else:
        pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]
        # Show thumbnails in a row
        cols = st.columns(5)
        for i, (col, img, f) in enumerate(zip(cols, pil_images, uploaded_files)):
            col.image(img, caption=f.name, use_container_width=True)

st.divider()

# ── Analyse button ──────────────────────────────────────────────────────────
ready = len(pil_images) == 5 and sentence.strip()
if not ready and uploaded_files:
    pass  # warning already shown above

if st.button("Analyse", type="primary", disabled=not ready):
    with st.spinner("Running analysis — CLIP encoding, IAPD prompts, scoring, object detection..."):
        data = run_analysis(sentence.strip(), pil_images)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1: SENTENCE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("1. Sentence Analysis")

    if data["is_figurative"]:
        st.success(f"Figurative / Idiomatic language detected — Confidence: {data['confidence']*100:.1f}%")
    else:
        st.warning(f"Likely literal language — Confidence: {data['confidence']*100:.1f}%")

    if data["idiom"]:
        st.write(f"**Idiom detected:** \"{data['idiom']}\"")
        st.write(f"**Meaning:** {data['idiom_meaning']}")
        st.write(f"**Origin:** {data['idiom_origin']}")
    else:
        st.write("No known idiom matched — analysing as general figurative expression.")

    st.write("**IAPD Prompts used for scoring:**")
    pcols = st.columns(3)
    pcols[0].info(f"**Literal prompt:**\n\n{data['perspectives'][0]}")
    pcols[1].success(f"**Figurative prompt:**\n\n{data['perspectives'][1]}")
    pcols[2].warning(f"**Contextual prompt:**\n\n{data['perspectives'][2]}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2: IMAGE RANKING
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("2. Image Ranking & Categorisation")
    st.write("Images ranked by figurative relevance. Objects detected using improved CLIP zero-shot (threshold ≥ 0.24).")

    sorted_results = sorted(data["results"], key=lambda r: r["rank"])
    rcols = st.columns(5)

    CATEGORY_COLORS = {
        "Figurative":      "🟢",
        "Literal":         "🟠",
        "Partial Literal": "🔵",
        "Random":          "🟤",
    }

    for r, col in zip(sorted_results, rcols):
        img = pil_images[r["index"]]
        icon = CATEGORY_COLORS.get(r["category"], "⚪")
        best = " ✓ Best" if r["rank"] == 1 else ""
        col.image(img, use_container_width=True)
        col.write(f"**#{r['rank']}{best}**")
        col.write(f"{icon} **{r['category']}**")
        col.write(f"Match: {r['confidence_pct']}%")
        col.write(f"Fig: {r['fig_score']:.3f} | Lit: {r['lit_score']:.3f}")
        gap_str = f"+{r['gap']:.3f}" if r['gap'] >= 0 else f"{r['gap']:.3f}"
        col.write(f"Gap: {gap_str}")
        if r["objects"]:
            col.write("**Objects:** " + ", ".join(o for o, _ in r["objects"]))
        elif r.get("scenes"):
            # Fix 2: show scene when no objects detected
            col.write("**Scene:** " + r["scenes"][0][0][:60])
        else:
            col.write("*No visual features detected*")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3: COMMONSENSE REASONING CHAIN
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("3. Visual Commonsense Reasoning Chain")
    st.write("Step-by-step reasoning from detected objects through to figurative meaning.")

    reasoning = data["reasoning"]

    # 1. Visual Evidence
    st.write("#### 👁️ Visual Evidence")
    ve = reasoning["visual_evidence"]
    if ve["objects"]:
        st.write("**Objects detected in top-ranked image:**")
        for obj_name, obj_score in ve["objects"]:
            st.write(f"- {obj_name} (similarity: {obj_score:.4f})")
    else:
        st.write("No high-confidence objects detected above threshold.")
    if ve["scenes"]:
        st.write("**Scene descriptions:**")
        for scene_text, score in ve["scenes"]:
            st.write(f"- \"{scene_text}\" ({score:.4f})")

    st.write("---")

    # 2. Scene Description
    st.write("#### 📷 Scene Description")
    st.write(reasoning["scene_description"])

    st.write("---")

    # 3. Object Chain → Commonsense Reasoning
    st.write("#### 🧠 Object Chain → Commonsense Reasoning")
    if reasoning["object_chain"]:
        for step in reasoning["object_chain"]:
            # strip markdown bold for plain display
            clean = step.replace("**", "")
            st.write(clean)
    else:
        st.write("No object chain could be built (no objects detected).")

    st.write("---")

    # 3.5  ConceptNet Knowledge Graph
    st.write("#### 🔗 ConceptNet Knowledge Graph")
    st.write("Semantic relations from idiom words to abstract concepts (via NumberBatch embeddings).")

    _cn_idiom = data.get("idiom") or ""
    _cn_meaning = data.get("idiom_meaning") or ""

    if _cn_idiom and CN is not None:
        _chains = build_cn_relation_chains(_cn_idiom, top_n=5)

        if _chains:
            st.write(f"**Idiom:** \"{_cn_idiom}\"")
            st.write("")

            # Per-word relation chains
            for _word, _relations in _chains.items():
                # Group concepts by relation type for a cleaner display
                _by_rel = {}
                for _concept, _rel in _relations:
                    _by_rel.setdefault(_rel, []).append(_concept)

                # Build the arrow chain string: word → RelType → c1 · c2 · c3
                _chain_parts = []
                for _rel, _concepts in _by_rel.items():
                    _chain_parts.append(f"**{_rel}** → {' · '.join(_concepts)}")

                _chain_str = f"**{_word}** → " + "  |  ".join(_chain_parts)
                st.markdown(_chain_str)

            st.write("")

            # Assembled interpretation sentence
            _assembled = build_cn_assembled_sentence(_cn_idiom, _chains, _cn_meaning)
            if _assembled:
                st.info(f"**Assembled interpretation:** {_assembled}")
        else:
            st.write(
                f"No NumberBatch entries found for the content words of \"{_cn_idiom}\". "
                "The word may not be in the ConceptNet vocabulary."
            )
    elif CN is None:
        st.write("ConceptNet NumberBatch not loaded — place `conceptnet/numberbatch_en.pkl` in the project folder.")
    else:
        st.write("No idiom detected — ConceptNet graph not available for general sentences.")

    st.write("---")

    # 4. Idiom Match
    st.write("#### 💡 Idiom Match")
    im = reasoning["idiom_match"]
    if im["idiom"]:
        st.write(f"**Matched idiom:** \"{im['idiom']}\"")
        st.write(f"**Meaning:** {im['meaning']}")
        st.write(f"**Origin:** {im['origin']}")
        st.write(f"**Figurative probe used:** \"{im['figurative_probe']}\"")
        if im["known"]:
            st.write("*Note: De-idiomised contextual prompt was used — idiom phrase replaced with its meaning to prevent CLIP anchoring to literal words.*")
    else:
        st.write("No idiom matched in dictionary — analysed as general figurative expression.")

    st.write("---")

    # 5. Score Breakdown
    st.write("#### ✅ Score Breakdown (Top-Ranked Image)")
    sb = reasoning["score_breakdown"]
    st.write(f"**Category:** {sb['category']}")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Figurative score", f"{sb['fig_score']:.4f}")
    sc2.metric("Literal score", f"{sb['lit_score']:.4f}")
    sc3.metric("Contextual score", f"{sb['ctx_score']:.4f}")
    sc4.metric("Gap (fig − lit)", f"{sb['gap']:+.4f}")
    st.write(f"**FLGS score:** {sb['flgs']:.4f}")
    st.write(f"**Scoring engine:** {sb['scoring_method']}")
    st.write(f"**Gap interpretation:** {sb['gap_interpretation']}")

    st.write("---")

    # 6. Final Conclusion
    st.write("#### 🎯 Final Figurative Meaning")
    clean_conclusion = reasoning["conclusion"].replace("**", "")
    st.write(clean_conclusion)
