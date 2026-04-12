"""
app.py — Figurative Language Understanding Demo (Full Version)
==============================================================
Upload 5 unlabelled images + a sentence.
The model automatically:
  1. Detects if the sentence contains figurative/idiomatic language
  2. Identifies and explains the idiom meaning
  3. Categorises each image as: Figurative / Literal / Partial Literal / Random / Distractor
  4. Ranks all images by figurative relevance
  5. Detects key objects in each image (CLIP zero-shot)
  6. Explains commonsense reasoning used (ConceptNet + IAPD)

HOW TO RUN:
    pip install flask
    python app.py
    Open: http://127.0.0.1:5000
"""

import os, io, base64, pickle, re
import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# STARTUP: Load models
# ──────────────────────────────────────────────────────────────────────────────
print("Loading CLIP ViT-B/32 …")
DEVICE = "cpu"
CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()
for p in CLIP_MODEL.parameters():
    p.requires_grad = False
print("✓ CLIP loaded.")

# ── Phase 4 Caption-Fused Ranker (trained models) ─────────────────────────────
# Task A model: 73.3% val acc — MLP combines 8 signals nonlinearly
# Task B model: 80.0% val acc — ctx→caption dominates for Task B
# Falls back to FLGS zero-shot if checkpoints are not found.
PHASE4_MODEL    = None   # Task A
PHASE4_MODEL_B  = None   # Task B (ctx→caption dominant)
try:
    from phase4_model import Phase4CaptionFusedRanker
    _models_dir = os.path.join(os.path.dirname(__file__), "models")

    _ckpt_a = os.path.join(_models_dir, "phase4_task_a_mlp.pt")
    if os.path.exists(_ckpt_a):
        _ckpt = torch.load(_ckpt_a, map_location=DEVICE)
        PHASE4_MODEL = Phase4CaptionFusedRanker(n_signals=8, head='mlp').to(DEVICE)
        PHASE4_MODEL.load_state_dict(_ckpt["model_state"])
        PHASE4_MODEL.eval()
        for p in PHASE4_MODEL.parameters():
            p.requires_grad = False
        print(f"✓ Phase 4 Task A loaded (val acc: {_ckpt.get('val_acc',0)*100:.1f}%).")
    else:
        print("  Phase 4 Task A checkpoint not found — using FLGS zero-shot scoring.")

    _ckpt_b = os.path.join(_models_dir, "phase4_task_b_mlp.pt")
    if os.path.exists(_ckpt_b):
        _ckpt = torch.load(_ckpt_b, map_location=DEVICE)
        PHASE4_MODEL_B = Phase4CaptionFusedRanker(n_signals=8, head='mlp').to(DEVICE)
        PHASE4_MODEL_B.load_state_dict(_ckpt["model_state"])
        PHASE4_MODEL_B.eval()
        for p in PHASE4_MODEL_B.parameters():
            p.requires_grad = False
        print(f"✓ Phase 4 Task B loaded (val acc: {_ckpt.get('val_acc',0)*100:.1f}%).")
except Exception as _e:
    print(f"  Phase 4 load skipped: {_e}")

CN_PKL = os.path.join(os.path.dirname(__file__), "conceptnet", "numberbatch_en.pkl")
CN = None
if os.path.exists(CN_PKL):
    with open(CN_PKL, "rb") as f:
        CN = pickle.load(f)
    print(f"✓ ConceptNet loaded ({len(CN):,} words).")

STOPWORDS = {"a","an","the","of","in","on","at","to","for","with","and",
             "or","but","is","are","was","were","i","he","she","it","we",
             "you","they","this","that","my","his","her","our","their"}

# ── Curated inference vocabulary ──────────────────────────────────────────────
# Only clean, single English words that are meaningful for commonsense reasoning
# about emotions, human traits, social dynamics, and figurative language.
# This replaces the raw first-30k CN scan which produced garbage like
# "agathokakological" and "adam_eve".
INFERENCE_VOCAB = [
    # Core emotions
    "anger","joy","fear","sadness","disgust","trust","surprise","anxiety",
    "stress","happiness","grief","shame","pride","guilt","love","hate",
    "envy","jealousy","hope","despair","confidence","excitement","frustration",
    "relief","disappointment","loneliness","regret","satisfaction","admiration",
    # Human character traits
    "corrupt","honest","dishonest","lazy","hardworking","clever","foolish",
    "stubborn","generous","greedy","brave","cowardly","loyal","disloyal",
    "deceptive","trustworthy","selfish","kind","cruel","innocent","naive",
    "wise","reckless","cautious","ambitious","humble","arrogant","patient",
    "impulsive","responsible","irresponsible","reliable","unreliable",
    # Social dynamics
    "conflict","cooperation","betrayal","loyalty","rivalry","friendship",
    "leadership","authority","power","weakness","competition","collaboration",
    "manipulation","control","exclusion","inclusion","rejection","acceptance",
    "isolation","influence","dominance","submission","alliance","opposition",
    # Actions & processes
    "struggle","compete","fight","escape","hide","reveal","succeed","fail",
    "help","harm","deceive","protect","attack","reward","punish","sacrifice",
    "risk","invest","waste","gain","lose","exploit","abuse","support","resist",
    "endure","overcome","abandon","pursue","achieve","avoid","confront",
    # Abstract social concepts
    "success","failure","problem","solution","opportunity","threat","change",
    "growth","decay","progress","obstacle","consequence","responsibility",
    "freedom","constraint","advantage","disadvantage","burden","challenge",
    "pressure","crisis","danger","safety","loss","damage","benefit","reward",
    # Moral / ethical concepts
    "justice","injustice","fairness","corruption","honesty","deception",
    "harm","good","evil","moral","unethical","punishment","integrity","sin",
    # Group & relationships
    "team","group","individual","leader","follower","outsider","community",
    "society","family","relationship","partner","enemy","ally","stranger",
    "colleague","superior","subordinate","rival","victim","aggressor",
    # Figurative meaning domain
    "symbol","implication","meaning","expression","hidden","literal","irony",
    "metaphor","representation","indication","suggestion","inference",
]

# Pre-build CN lookup restricted to the curated vocabulary only
_INFERENCE_VECS = {}   # word → np array; populated after CN loads
if CN is not None:
    for _w in INFERENCE_VOCAB:
        if _w in CN:
            _INFERENCE_VECS[_w] = CN[_w]
    print(f"✓ Inference vocab: {len(_INFERENCE_VECS)}/{len(INFERENCE_VOCAB)} words found in ConceptNet.")

# ──────────────────────────────────────────────────────────────────────────────
# IDIOM DICTIONARY  (meaning + explanation)
# ──────────────────────────────────────────────────────────────────────────────
IDIOM_DB = {
    # Format: idiom → (meaning, visual_figurative_prompt, visual_literal_prompt, origin)
    # visual_figurative_prompt: what to search for as a CLIP image query (concrete, visual)
    # visual_literal_prompt:    what the words literally look like as an image
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
    "spill the beans":      ("to accidentally reveal secret information",
                             "a person accidentally telling someone a secret they should not",
                             "beans being spilled from a container onto the floor",
                             "beans used in secret voting — spilling revealed the count"),
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
    "dutch courage":        ("confidence gained from drinking alcohol",
                             "a nervous person drinking alcohol to feel brave before a challenge",
                             "a Dutch person or windmill with a bottle of alcohol",
                             "Dutch soldiers drank gin before battle"),
    "guinea pig":           ("a person used as a test subject for an experiment",
                             "a person being tested or experimented on in a laboratory",
                             "a guinea pig animal in a laboratory or cage",
                             "guinea pigs were used in lab experiments"),
    "dead wood":            ("useless unproductive people or elements in an organization",
                             "unproductive lazy employees being removed from a company",
                             "dead dry wood branches on a dead tree",
                             "dead wood in a tree contributes nothing"),
    "acid test":            ("a decisive and conclusive test of quality",
                             "a definitive test proving or disproving something beyond doubt",
                             "acid being poured onto metal in a chemistry test",
                             "gold tested with acid to verify authenticity"),
    "bad apple":            ("one corrupt person who spoils the entire group",
                             "a troublemaker being expelled from a team while upset teammates watch",
                             "one rotten brown apple among fresh red apples",
                             "one rotten apple rots the whole barrel"),
    "bells and whistles":   ("attractive but unnecessary extra features",
                             "a product or gadget with many flashy extra unnecessary features",
                             "bells and whistles as musical or mechanical devices",
                             "steam engines: bells and whistles were add-ons"),
    "silver bullet":        ("a simple magical solution to a complex problem",
                             "a single perfect solution that easily solves a difficult problem",
                             "a silver-coloured bullet or ammunition",
                             "werewolves killed by silver bullets; metaphor for perfect fix"),
    "swan song":            ("a final performance, effort or work before ending",
                             "a performer giving their final farewell performance on stage",
                             "a white swan opening its beak and singing",
                             "swans believed to sing beautifully just before dying"),
    "trojan horse":         ("a deceptive strategy that appears helpful but causes harm",
                             "something appearing friendly or useful that is actually a trap",
                             "a large wooden horse sculpture outside ancient Troy city walls",
                             "Greeks hid inside a wooden horse to infiltrate Troy"),
    "white whale":          ("an obsessive unattainable goal that consumes someone",
                             "a person obsessively and desperately chasing an impossible dream",
                             "a large white sperm whale swimming in the ocean",
                             "from Moby Dick: Captain Ahab's obsessive pursuit"),
    "seal of approval":     ("official endorsement or approval from an authority",
                             "an official document or product receiving a stamp of approval",
                             "a seal animal or wax seal stamp on an official document",
                             "wax seals on documents indicated authentic authorisation"),
    "panda car":            ("a black and white police patrol car",
                             "a police patrol car on duty driving on a street",
                             "a black and white car resembling a panda bear pattern",
                             "markings resemble a panda bear colouring"),
    "open goal":            ("an easy opportunity that should not be missed",
                             "a person missing an obvious and easy opportunity right in front of them",
                             "an empty football goal with no goalkeeper",
                             "football: a goal with no goalkeeper is easy to score"),
    "cutting edge":         ("the most advanced and innovative",
                             "a futuristic advanced technology or innovative modern product",
                             "the sharp blade edge of a knife or cutting tool",
                             "sharp edge of a blade = forefront of progress"),
    "dog's dinner":         ("a complete mess; something done very badly",
                             "a chaotic messy disorganised situation or result",
                             "a dog eating a messy bowl of food scraps",
                             "a dog's meal is typically chaotic and unappetising"),
    "devil's advocate":     ("someone who argues the opposing side intentionally",
                             "a person arguing against the popular view in a debate to test ideas",
                             "a devil figure making an argument or presenting a legal case",
                             "Catholic church role: argue against sainthood"),
    "elbow room":           ("enough space to move and act freely",
                             "a person finally having enough open space to move comfortably",
                             "an elbow and arm needing physical space to extend freely",
                             "elbow = personal space; room = adequate distance"),
    "night owl":            ("a person who prefers to stay up late at night",
                             "a person awake and energetic late at night while others sleep",
                             "an owl bird perched on a branch in darkness at night",
                             "owls are nocturnal and active at night"),
    "cat's eyes":           ("small reflective road markers embedded in road surfaces",
                             "small reflective markers glowing on a dark road at night",
                             "the glowing reflective eyes of a cat in the dark",
                             "named after the reflective eyes of cats in darkness"),
    "second nature":        ("a habit so ingrained it feels completely automatic",
                             "a person doing a complex skill effortlessly and automatically",
                             "a person practising something repetitively until natural",
                             "repetition makes behaviour feel natural"),
    "new blood":            ("fresh new people bringing new energy and ideas",
                             "enthusiastic new team members bringing fresh energy and innovation",
                             "blood or a transfusion representing something new entering a system",
                             "new blood infuses energy into a stagnant system"),
    "ancient history":      ("something from so long ago it is no longer relevant",
                             "an old event or relationship that is completely forgotten and irrelevant now",
                             "ancient ruins or archaeological artefacts from thousands of years ago",
                             "ancient = very old; history = past events"),
    "busman's holiday":     ("spending your holiday doing the same thing as your regular job",
                             "a worker spending their day off doing their exact same work anyway",
                             "a bus driver on holiday riding a bus as a passenger",
                             "a bus driver spending holiday riding buses"),
    "agony aunt":           ("a columnist who gives advice on personal problems",
                             "a person giving kind sympathetic advice and guidance to someone upset",
                             "an aunt figure sitting and listening to someone's problems",
                             "agony = suffering shared; aunt = caring maternal figure"),
    "best man":             ("the groom's chief helper at a wedding",
                             "a loyal trusted friend supporting the groom at a wedding ceremony",
                             "a man in a suit standing next to the groom at a wedding",
                             "from wedding tradition: most trusted groomsman"),
    "one-horse race":       ("a contest with only one likely or possible winner",
                             "a competition where one person is so dominant there is no real contest",
                             "a single horse racing alone on a race track with no competitors",
                             "one horse = no real competition"),
    "cold turkey":          ("abrupt complete withdrawal from an addiction without help",
                             "a person suffering through painful withdrawal symptoms shivering and sweating",
                             "a raw cold uncooked turkey carcass on a table",
                             "resembles pale cold skin of someone going through withdrawal"),

    # ── 69 missing dataset idioms (Task A Train + Dev + Extended) ────────────
    "apples and oranges":   ("two things that are too different to be compared fairly",
                             "two people trying to compare completely different and incomparable things",
                             "a red apple and an orange placed side by side",
                             "comparing fundamentally different things is as absurd as comparing fruits"),
    "armchair critic":      ("a person who criticises without any real experience or action",
                             "a person relaxing in an armchair loudly criticising others who are working hard",
                             "a person sitting comfortably in an armchair reading or watching",
                             "critics who never leave their armchair have no practical experience"),
    "baby blues":           ("feeling of sadness or depression after giving birth",
                             "a new mother feeling sad overwhelmed and tearful after having a baby",
                             "a baby wearing or surrounded by blue coloured things",
                             "postpartum sadness; blue = low mood; common in new mothers"),
    "banana republic":      ("a small corrupt country dominated by foreign business interests",
                             "a corrupt government official accepting bribes while citizens suffer poverty",
                             "a tropical country with banana plantations and political chaos",
                             "coined by O Henry in 1904 to describe Central American countries controlled by fruit companies"),
    "beached whale":        ("a very large helpless person or thing stranded somewhere",
                             "a large helpless person lying sprawled unable to move or get up",
                             "a massive whale stranded and lying on a beach unable to return to the sea",
                             "beached whales are helpless and immobile — metaphor for being stuck"),
    "bear market":          ("a period of falling stock prices and economic decline",
                             "investors looking panicked as stock market charts show steep downward falls",
                             "a large bear animal walking through a financial district or stock exchange",
                             "bears swipe downward with their paws; contrasts with bull market"),
    "big cheese":           ("an important and powerful person in an organisation",
                             "a self-important powerful executive commanding authority over everyone",
                             "an enormous impressive wheel of cheese on display",
                             "Urdu 'chiz' meaning thing; misinterpreted as cheese in English"),
    "big fish":             ("an important influential person especially in a small group",
                             "a dominant influential person standing out and commanding respect in a small group",
                             "a large fish swimming among much smaller fish in a pond",
                             "a big fish in a small pond has disproportionate influence"),
    "black box":            ("a device whose internal workings are mysterious or unknown",
                             "a mysterious system or process where the inner workings are completely hidden",
                             "a black rectangular flight recorder box from an aircraft",
                             "aviation: flight recorders are painted orange but called black boxes"),
    "brain surgery":        ("something extremely difficult and complex",
                             "a person struggling with a task so difficult it seems impossibly complex",
                             "a surgeon carefully performing a delicate operation on a human brain",
                             "used sarcastically: this isn't brain surgery = this isn't hard"),
    "brass ring":           ("a highly desirable prize goal or opportunity",
                             "a person reaching and striving ambitiously for a coveted prize or goal",
                             "a shiny brass metal ring held up as a prize or reward",
                             "carousel riders grabbed brass rings for a free ride; became a success metaphor"),
    "bread and butter":     ("the main reliable source of income or basic necessities of life",
                             "a person doing their core everyday essential work that pays the bills",
                             "a slice of bread being spread with butter",
                             "bread and butter are the most basic staple foods — metaphor for basics"),
    "bull market":          ("a period of rising stock prices and economic growth",
                             "excited optimistic investors watching stock market charts rise steeply upward",
                             "a large bull animal charging through a financial district",
                             "bulls thrust upward with their horns; contrasts with bear market"),
    "bun in the oven":      ("a woman who is pregnant",
                             "a pregnant woman with a visibly rounded baby bump",
                             "a bread roll or bun baking inside a kitchen oven",
                             "the womb is the oven; the baby is the bun — domestic metaphor for pregnancy"),
    "busy bee":             ("a very industrious and hardworking person always doing things",
                             "an energetic person rushing between tasks working tirelessly all day",
                             "a bee flying busily between flowers collecting pollen",
                             "bees are famously industrious creatures always working in the hive"),
    "chicken feed":         ("a very small and insignificant amount of money",
                             "a person dismissing a tiny sum of money as completely worthless",
                             "a farmer scattering grain and seeds on the ground to feed chickens",
                             "the small amount of grain fed to chickens is trivial and worthless"),
    "chocolate teapot":     ("something completely useless and not fit for purpose",
                             "a completely useless object or person failing at their basic purpose",
                             "a teapot made entirely of chocolate melting when hot tea is poured in",
                             "a chocolate teapot would melt immediately — utterly useless"),
    "close shave":          ("a narrow escape from danger or disaster",
                             "a person narrowly escaping a serious accident or danger by a fraction",
                             "a barber shaving a person's face very closely with a razor",
                             "a razor blade close to the skin; any closer and it would cut"),
    "copy cat":             ("a person who imitates or copies others without originality",
                             "a person shamelessly copying everything another person does or creates",
                             "a cat perfectly mimicking and copying another cat's exact actions",
                             "cats were thought to imitate; 'copy' was added to emphasise imitation"),
    "dirty money":          ("money obtained through illegal or immoral activities",
                             "a criminal counting cash obtained through illegal corrupt activities",
                             "muddy or stained banknotes and coins covered in filth",
                             "money tainted by crime or corruption is metaphorically dirty"),
    "dirty word":           ("a subject that is taboo or strongly disapproved of in a context",
                             "a person reacting with shock and disgust when a forbidden topic is mentioned",
                             "offensive or profane words written on a surface",
                             "some words are socially unacceptable; the concept extends to taboo topics"),
    "donkey work":          ("the hard tedious and boring part of a task nobody wants to do",
                             "a person doing all the exhausting repetitive grunt work while others relax",
                             "a donkey carrying a heavy load of goods doing physical labour",
                             "donkeys do the heaviest most thankless work; associated with drudgery"),
    "eager beaver":         ("an enthusiastic hardworking person who is overly keen",
                             "an overly enthusiastic person volunteering eagerly for every task",
                             "a beaver busily and energetically building a dam with sticks",
                             "beavers are famously industrious; eager emphasises enthusiasm"),
    "eye candy":            ("something or someone visually attractive but lacking substance",
                             "a very attractive person or object admired for looks alone with no depth",
                             "brightly coloured sweets and candy displayed attractively",
                             "candy appeals only to the eyes just as sweets appeal only to taste"),
    "flea market":          ("an outdoor market selling second-hand cheap goods",
                             "people browsing colourful stalls of second-hand goods at an outdoor market",
                             "a market infested with fleas or a magnified flea on a market stall",
                             "originally from Paris Marche aux Puces; old goods were thought to carry fleas"),
    "flower child":         ("a hippie advocating peace love and non-violence in the 1960s",
                             "a young peaceful hippie wearing flowers in their hair at a protest",
                             "a young child happily surrounded by blooming colourful flowers",
                             "1960s counterculture: protesters placed flowers in gun barrels as peace symbols"),
    "ghost town":           ("a once-busy place that is now completely empty and abandoned",
                             "an empty abandoned town with deserted streets and boarded-up buildings",
                             "a town full of ghosts or a spooky haunted abandoned settlement",
                             "towns deserted after mining booms ended; became eerie ghost-like places"),
    "grass roots":          ("ordinary people at the local level rather than leadership",
                             "ordinary community members organising and working together from the ground up",
                             "the roots of grass plants growing deep into the soil underground",
                             "grass roots are the base foundation of a plant — metaphor for community base"),
    "graveyard shift":      ("a work shift during the late night or early morning hours",
                             "tired workers toiling alone in an empty building at 3am in the dark",
                             "a worker walking past a graveyard during a night shift",
                             "named for the eerie quiet of night shifts; also linked to grave digging"),
    "gravy train":          ("a situation where someone earns money easily with little effort",
                             "a person comfortably profiting from an effortless cushy arrangement",
                             "a train with gravy flowing from it or carrying gravy containers",
                             "railroad slang: gravy = easy money; the train delivers it effortlessly"),
    "green fingers":        ("a natural talent for gardening and growing plants successfully",
                             "a skilled gardener surrounded by thriving lush plants and flowers",
                             "a person's fingers coloured green from handling plants and soil",
                             "British equivalent of American green thumb; plants stain fingers green"),
    "hair of the dog":      ("an alcoholic drink taken as a cure for a hangover",
                             "a person drinking alcohol in the morning to relieve a hangover headache",
                             "a strand of dog fur or a dog with visible hair",
                             "medieval belief: treating a dog bite with hair from the same dog"),
    "heart of gold":        ("an extremely kind generous and caring person",
                             "a genuinely warm kind-hearted person helping others selflessly and lovingly",
                             "a golden heart shape made of shining gold",
                             "gold is precious and pure; a golden heart represents pure kindness"),
    "hen party":            ("a party exclusively for women usually before a wedding",
                             "a group of women celebrating and having fun together at a party",
                             "a group of hens chickens gathered together in a farmyard",
                             "hen = female bird; party of hens = party of women"),
    "high life":            ("a luxurious and extravagant lifestyle enjoyed by the wealthy",
                             "wealthy people enjoying lavish parties expensive food and luxury surroundings",
                             "people living at a high altitude or a life literally lived high up",
                             "living at a high social level with access to expensive pleasures"),
    "honey trap":           ("a trap using romantic or sexual attraction to manipulate someone",
                             "a seductive person luring an unsuspecting victim into a compromising situation",
                             "a jar of honey set as a trap to catch insects or animals",
                             "honey attracts animals into traps — metaphor for seductive entrapment"),
    "hot potato":           ("a controversial sensitive issue that nobody wants to deal with",
                             "politicians and officials desperately passing a controversial problem to each other",
                             "a person struggling to hold a steaming hot potato that burns their hands",
                             "a hot potato burns if held too long — no one wants to handle it"),
    "inner circle":         ("a small exclusive group of trusted advisors close to a leader",
                             "a powerful leader surrounded by a small group of trusted loyal advisors",
                             "people standing inside a drawn circle while others stand outside it",
                             "those inside the circle have access others do not"),
    "ivory tower":          ("a state of privileged isolation from practical reality",
                             "an academic or intellectual isolated from real world problems in luxury",
                             "a tall elegant tower built from white ivory material",
                             "from the Bible Song of Solomon; adopted for academic detachment from reality"),
    "loan shark":           ("a moneylender who charges extremely high illegal interest rates",
                             "a threatening intimidating moneylender demanding repayment with menace",
                             "a shark swimming with money or loan documents in its mouth",
                             "sharks are predatory; loan sharks prey on desperate borrowers"),
    "lounge lizard":        ("a man who frequents fashionable places seeking wealthy women",
                             "a well-dressed man idly lounging in an expensive bar charming wealthy people",
                             "a lizard basking and lounging lazily in the sun",
                             "lizards bask idly in warmth; lounge lizards bask in social settings"),
    "love triangle":        ("a romantic situation involving three people in complicated relationships",
                             "three people tangled in a complicated romantic situation causing jealousy and conflict",
                             "a triangle shape with hearts at each corner representing three people",
                             "a geometric triangle has three points; a love triangle has three people"),
    "low-hanging fruit":    ("an easy task or goal that can be achieved with minimal effort",
                             "a person easily picking the simplest most accessible tasks first",
                             "fruit hanging low on a tree branch easy to reach and pick",
                             "fruit at the bottom is easiest to harvest; metaphor for easy wins"),
    "marching orders":      ("instructions to leave or a dismissal from a job",
                             "a boss firmly dismissing an employee and telling them to leave immediately",
                             "soldiers receiving formal written orders to march and deploy",
                             "military: soldiers given marching orders must leave immediately"),
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
    "pain in the neck":     ("a very annoying and irritating person or thing",
                             "an extremely annoying irritating person causing frustration to everyone",
                             "a person rubbing and wincing in pain from a sore stiff neck",
                             "a euphemism replacing a ruder body part; the neck aches from strain"),
    "pig's ear":            ("something done very badly or clumsily; a mess",
                             "a person making a complete disastrous mess of a task",
                             "a pig's actual ear or a decorative pig ear shape",
                             "British rhyming slang: pig's ear = beer; also means making a mess"),
    "pins and needles":     ("a tingling sensation from a limb falling asleep",
                             "a person wincing with the uncomfortable tingling feeling of a limb waking up",
                             "sharp pins and sewing needles scattered on a surface",
                             "the pricking sensation of returning blood flow feels like pins and needles"),
    "pipe dream":           ("a hope or plan that is completely impossible to achieve",
                             "a person daydreaming about a completely unrealistic impossible fantasy",
                             "a person smoking a pipe while lost in an elaborate daydream",
                             "opium pipe hallucinations were vivid but unreal — metaphor for impossible fantasies"),
    "private eye":          ("a private detective or investigator",
                             "a detective in a trench coat secretly investigating and gathering evidence",
                             "a large human eye with the word private written next to it",
                             "from Pinkerton Detective Agency's logo: an open eye with 'We Never Sleep'"),
    "rat race":             ("the exhausting competitive struggle of modern working life",
                             "exhausted office workers competing desperately in a stressful never-ending routine",
                             "rats running frantically and competitively around a maze or race track",
                             "rats in laboratory mazes run endlessly without reward — metaphor for futile competition"),
    "rat run":              ("a route through residential streets used to avoid traffic",
                             "a driver cutting through quiet residential streets to avoid main road traffic",
                             "rats running through narrow tunnels and passages to find shortcuts",
                             "rats find shortcuts through walls; drivers find shortcuts through backstreets"),
    "red flag":             ("a warning sign that something is wrong or dangerous",
                             "a person noticing a serious warning sign that something is very wrong",
                             "a bright red flag being waved or raised as a signal",
                             "red flags historically warned of danger; now used for warning signs in relationships"),
    "rocket science":       ("something extremely complicated and difficult to understand",
                             "a person completely overwhelmed by something unnecessarily complicated",
                             "a scientist in a lab with complex equations and rocket blueprints",
                             "used sarcastically: this isn't rocket science = this isn't complicated"),
    "secret santa":         ("an anonymous gift exchange where participants are randomly assigned",
                             "office workers exchanging wrapped gifts anonymously at a Christmas party",
                             "a santa claus figure secretly sneaking gifts without being seen",
                             "combines secret anonymous gifting with the Santa Claus tradition"),
    "shrinking violet":     ("an extremely shy timid and self-effacing person",
                             "a very shy timid person hiding away and avoiding all attention",
                             "a small violet flower curling inward and wilting away",
                             "violets are small delicate flowers that seem to shrink from attention"),
    "smoking gun":          ("clear and undeniable proof of guilt or wrongdoing",
                             "a detective holding up undeniable evidence that proves someone's guilt",
                             "a gun with smoke still rising from its barrel after being fired",
                             "a recently fired gun proves someone just shot it — irrefutable evidence"),
    "snake in the grass":   ("a treacherous person who hides their true dangerous intentions",
                             "a smiling friendly person secretly plotting betrayal against their trusting friend",
                             "a dangerous snake hidden and camouflaged in tall grass",
                             "Virgil's Aeneid: a snake lurking in grass strikes unexpectedly"),
    "spring chicken":       ("a young and inexperienced person",
                             "a young naive inexperienced person clearly out of their depth",
                             "a young baby chick hatching in springtime",
                             "young spring chickens are tender and fresh; used to describe youth"),
    "thin ice":             ("a risky or dangerous situation where mistakes have serious consequences",
                             "a person carefully and nervously tiptoeing through a very dangerous situation",
                             "a person walking nervously on cracking thin ice over frozen water",
                             "thin ice can crack and cause someone to fall through into freezing water"),
    "top dog":              ("the most powerful or dominant person in a group",
                             "a confident dominant leader commanding authority and respect from everyone",
                             "a dog standing proudly above other dogs asserting dominance",
                             "in dogfighting the winning dog was on top; adapted to mean leader"),
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
    "white hat":            ("a good ethical person who acts morally and fights for good",
                             "an ethical heroic person standing up against wrongdoing and corruption",
                             "a cowboy wearing a white hat symbolising the good guy hero",
                             "Western films: heroes wore white hats; villains wore black hats"),
    "zebra crossing":       ("a pedestrian crossing marked with black and white stripes",
                             "pedestrians safely crossing a road at a striped pedestrian crossing",
                             "a zebra animal standing on a road crossing",
                             "the black and white stripes resemble a zebra's markings"),

    # ── 15 new idioms from Extended Evaluation dataset ────────────────────────
    "act of god":           ("an uncontrollable natural disaster or event",
                             "a massive storm flood earthquake or natural disaster destroying everything",
                             "a deity or divine figure performing a miraculous act from the sky",
                             "legal term for catastrophic events beyond human control"),
    "big wig":              ("an important or powerful person",
                             "a powerful executive or boss commanding respect in a meeting room",
                             "a person wearing a large elaborate powdered wig like a judge",
                             "historical: elaborate wigs signified high social rank in Europe"),
    "cold feet":            ("sudden nervousness or loss of confidence before doing something",
                             "a nervous anxious person hesitating and backing away from a commitment",
                             "a person's bare feet standing in cold snow or icy water",
                             "soldiers got cold feet literally in trenches; adopted as metaphor for cowardice"),
    "fancy dress":          ("a costume worn at a party or event",
                             "people in colourful elaborate costumes at a fancy dress party",
                             "an elegant expensive formal dress or evening gown",
                             "British English: 'fancy' means elaborate or costume rather than elegant"),
    "field work":           ("practical research or work done outside in the real world",
                             "a scientist or researcher collecting data outdoors in the field",
                             "a person doing physical manual labour work in an open field",
                             "contrasts with lab or desk work; implies hands-on real-world research"),
    "flying saucer":        ("a disc-shaped UFO or alien spacecraft",
                             "a glowing disc-shaped alien spacecraft flying through the night sky",
                             "a saucer or plate flying through the air",
                             "coined in 1947 from pilot Kenneth Arnold's description of UFOs"),
    "green light":          ("official permission or approval to proceed",
                             "a person receiving approval and permission to start a project",
                             "a green traffic light signal on a road",
                             "traffic lights: green means go — adapted to mean approval to proceed"),
    "heart of stone":       ("a cold unfeeling person with no compassion or empathy",
                             "a cold heartless person showing no emotion or compassion whatsoever",
                             "a literal stone carved into the shape of a heart",
                             "stone is hard and cold — metaphor for someone without warm feelings"),
    "hot air":              ("empty meaningless talk with no substance or truth",
                             "a politician or person talking loudly while saying nothing of substance",
                             "hot steam or air rising from a vent or hot air balloon",
                             "inflated speech like a hot air balloon — full of air but no substance"),
    "party animal":         ("a person who loves socialising and partying enthusiastically",
                             "an energetic enthusiastic person dancing and celebrating at a wild party",
                             "a wild animal dressed up or attending a party celebration",
                             "combines the social joy of a party with the wildness of an animal"),
    "peas in a pod":        ("two people who are very similar or always together",
                             "two people who are identical in personality and always inseparable",
                             "two green peas sitting together inside an open pea pod",
                             "peas in the same pod are indistinguishable from each other"),
    "snail mail":           ("traditional slow postal mail as opposed to email",
                             "a slow postal worker delivering letters very slowly compared to email",
                             "a snail carrying a letter or envelope on its shell",
                             "contrasts the slowness of a snail with the speed of email"),
    "sour grapes":          ("pretending not to want something you actually cannot have",
                             "a person dismissively pretending they never wanted something they failed to get",
                             "a person tasting bitter or sour grapes with a grimace",
                             "Aesop fable: fox called grapes sour after failing to reach them"),
    "watering hole":        ("a bar pub or place where people regularly gather to drink",
                             "friends gathering and drinking together at a busy local pub or bar",
                             "wild animals gathering at a waterhole in the savanna to drink",
                             "animals gather at watering holes to drink — metaphor for social drinking spots"),
    "act of god":           ("an uncontrollable natural disaster beyond human control",
                             "a catastrophic storm flood or natural disaster with no warning",
                             "a divine supernatural being performing a miraculous act",
                             "legal and insurance term for unforeseeable catastrophic events"),
}

def lookup_idiom(sentence):
    """Find best-matching idiom in sentence and return meaning + visual prompts."""
    s = sentence.lower()
    best, best_len = None, 0
    for idiom in IDIOM_DB:
        if idiom in s and len(idiom) > best_len:
            best, best_len = idiom, len(idiom)
    if best:
        entry = IDIOM_DB[best]
        if len(entry) == 4:
            meaning, vis_fig, vis_lit, origin = entry
        else:
            meaning, origin = entry
            vis_fig = vis_lit = None
        return best, meaning, origin, vis_fig, vis_lit
    return None, None, None, None, None

# ──────────────────────────────────────────────────────────────────────────────
# OBJECT VOCABULARY (for CLIP zero-shot object detection)
# ──────────────────────────────────────────────────────────────────────────────
OBJECT_VOCAB = [
    # People & body
    "a person","a man","a woman","a child","a hand","a face","a crowd",
    # Animals
    "a dog","a cat","a bird","a horse","an owl","a sheep","a fish","a wolf",
    "a snake","a bear","a cow","a duck","a swan","a panda","a seal",
    # Nature
    "a tree","a flower","grass","water","fire","ice","snow","a rock",
    "a mountain","the sky","clouds","the sun","the ocean","a river",
    # Household / everyday objects
    "a bucket","a ladder","a barrel","a rope","a key","a clock","a book",
    "a candle","a lamp","a table","a chair","a door","a window","a bag",
    # Tools and work
    "a hammer","a brush","a shovel","a screwdriver","a wrench",
    # Food and drink
    "an apple","bread","a cake","meat","an egg","soup","coffee","wine","a bottle",
    # Vehicles
    "a car","a bus","a boat","an airplane","a bicycle",
    # Buildings and places
    "a house","a building","a street","a bridge","a road","a fence","a hospital",
    # Technology
    "a computer","a phone","a screen",
    # Symbolic / abstract visual
    "a medal","a trophy","money","a gun","a sword","a crown","a flag",
    # Body language / states
    "a hug","a fight","someone sleeping","someone crying","someone laughing",
    "someone working hard","someone running","someone falling",
]

SCENE_VOCAB = [
    # Actions & states
    "a person working very hard",
    "someone helping another person",
    "a group of people cooperating",
    "a person feeling exhausted or overwhelmed",
    "someone feeling happy and joyful",
    "a person in a difficult or dangerous situation",
    "someone making a serious mistake",
    "a competitive or rivalry situation",
    "a peaceful and relaxing scene",
    "someone being deceptive or sneaky",
    "a person being ignored or excluded",
    "someone achieving success or victory",
    "a chaotic or disorganised scene",
    "a person being lazy or inactive",
    "someone being overworked or burdened",
    "a risky or threatening situation",
    "a person feeling anxious or worried",
    "someone being generous or kind",
    "a person feeling isolated or alone",
    "people arguing or in conflict",
    "a celebratory or festive scene",
    "a person being stubborn or unyielding",
    "someone hiding their true feelings",
    "a powerful or dominant figure",
    "a slow or inefficient process",
    "a person taking a bold risk",
    "a surprising or unexpected event",
    "someone being rewarded or praised",
    "a vulnerable or weak situation",
    "someone betraying trust",
]

print("Encoding object vocabulary …")
with torch.no_grad():
    OBJ_TOKENS   = clip.tokenize(OBJECT_VOCAB,  truncate=True).to(DEVICE)
    OBJ_EMBEDDINGS = CLIP_MODEL.encode_text(OBJ_TOKENS)
    OBJ_EMBEDDINGS = OBJ_EMBEDDINGS / OBJ_EMBEDDINGS.norm(dim=-1, keepdim=True)
    SCENE_TOKENS = clip.tokenize(SCENE_VOCAB, truncate=True).to(DEVICE)
    SCENE_EMBEDDINGS = CLIP_MODEL.encode_text(SCENE_TOKENS)
    SCENE_EMBEDDINGS = SCENE_EMBEDDINGS / SCENE_EMBEDDINGS.norm(dim=-1, keepdim=True)
print(f"✓ {len(OBJECT_VOCAB)} object labels + {len(SCENE_VOCAB)} scene labels encoded.")

def detect_scene(img_emb, top_k=3, threshold=0.20):
    """Return top scene/action descriptors present in the image via CLIP zero-shot."""
    sims = (SCENE_EMBEDDINGS @ img_emb.unsqueeze(-1)).squeeze(-1)
    topk = sims.topk(min(top_k, len(SCENE_VOCAB)))
    results = []
    for j, i in enumerate(topk.indices.tolist()):
        score = float(topk.values[j])
        if score >= threshold:
            results.append((SCENE_VOCAB[i], score))
    return results

def detect_objects(img_emb, top_k=4, threshold=0.21):
    """
    Return objects detected in an image via CLIP zero-shot.
    Only returns labels whose cosine similarity exceeds `threshold` to
    avoid false positives (e.g. 'someone crying' when no person is present).
    CLIP ViT-B/32 cosine sims for 'a photo of X' typically sit:
      clearly present:  > 0.22
      possibly present: 0.19-0.22
      not present:      < 0.19
    We use 0.21 as a conservative cutoff.
    """
    sims = (OBJ_EMBEDDINGS @ img_emb.unsqueeze(-1)).squeeze(-1)  # (N,)
    topk = sims.topk(top_k)
    results = []
    for j, i in enumerate(topk.indices.tolist()):
        score = float(topk.values[j])
        if score >= threshold:
            label = OBJECT_VOCAB[i].replace("a ","").replace("an ","").replace("the ","").strip()
            results.append((label, score))
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

def encode_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    t   = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = CLIP_MODEL.encode_image(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.squeeze(0)

# ──────────────────────────────────────────────────────────────────────────────
# IAPD PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
def iapd_prompts(sentence, idiom=None, vis_fig=None, vis_lit=None, idiom_meaning=None):
    """
    Generate 3 IAPD perspectives.
    When the idiom is known (vis_fig/vis_lit provided), use concrete visual prompts
    instead of generic templates — this is the key fix for accuracy.
    Generic template: 'an image representing figurative meaning of X'  ← CLIP struggles
    Specific prompt:  'a lazy person watching TV on a couch all day'   ← CLIP understands

    CRITICAL: For the contextual prompt, replace the idiom phrase with its meaning.
    Without this, CLIP reads 'bad apple' in the sentence literally and biases toward
    the literal image of a rotten apple rather than the figurative scene.
    e.g. "he's one bad apple" → "he's one corrupt person who spoils the group"
    """
    phrase = idiom if idiom else sentence
    words  = [w.strip(".,!?\"'") for w in phrase.lower().split() if w.lower() not in STOPWORDS]

    # Literal: use dictionary visual if available, else generic word-based
    if vis_lit:
        literal = vis_lit
    elif len(words) >= 2:
        literal = f"a photo showing {' and '.join(words[:3])}"
    else:
        literal = f"a photo of {phrase}"

    # Figurative: use dictionary visual if available — THIS IS THE KEY FIX
    if vis_fig:
        figurative = vis_fig
    else:
        figurative = f"an image representing the figurative meaning of '{phrase}'"

    # Contextual: de-idiomize the sentence when idiom + meaning are known.
    # Replacing idiom words with their meaning stops CLIP from anchoring to the
    # literal visual referent of the idiom phrase (e.g. an actual apple for "bad apple").
    if idiom and idiom_meaning and idiom in sentence.lower():
        contextual = sentence.lower().replace(idiom, idiom_meaning)
    else:
        contextual = sentence

    return literal, figurative, contextual

# ──────────────────────────────────────────────────────────────────────────────
# CONCEPTNET REASONING
# ──────────────────────────────────────────────────────────────────────────────
def cn_word_associations(words, top_n=5):
    """
    For each content word, find its closest neighbours within the curated
    INFERENCE_VOCAB — a hand-picked set of meaningful English words covering
    emotions, character traits, social dynamics, and abstract concepts.
    This avoids the garbage that raw CN scanning produces (compound terms,
    rare/nonsense words like 'agathokakological', 'adam_eve', etc.).
    """
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

def find_idiom_candidates(sentence, exclude=None, top_k=3):
    """
    Score `sentence` against every IDIOM_DB meaning embedding and return
    the top-k closest idioms (excluding the already-matched one).
    Returns list of (idiom_phrase, meaning, cosine_score).
    """
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


def build_commonsense_chain(
        sentence, idiom, idiom_meaning, idiom_origin,
        best_result, all_results, perspectives,
        objects, scenes, gap_weight, scoring_method,
        known_idiom=False):
    """
    Build the 7-section visual commonsense reasoning chain as structured HTML.

    Sections:
      1. Visual Evidence      — objects + scene actions detected in the top image
      2. Literal Meaning      — what the scene physically depicts
      3. Commonsense Inference — what such a scene typically implies (via ConceptNet)
      4. Best Idiom            — matched idiom + meaning
      5. Why It Fits           — evidence from scores + visual overlap
      6. Other Candidates Rejected — runner-up idioms from IDIOM_DB and why they don't fit
      7. Final Figurative Meaning  — concise conclusion
    """

    def section(icon, title, body):
        return (f'<div class="csec">'
                f'<div class="csec-hd">{icon} <b>{title}</b></div>'
                f'<div class="csec-bd">{body}</div>'
                f'</div>')

    parts = []

    # ── 1. Visual Evidence ────────────────────────────────────────────────────
    obj_labels = [o for o, _ in objects] if objects else []
    scene_labels = [s for s, _ in scenes] if scenes else []
    if obj_labels or scene_labels:
        ev_lines = []
        if obj_labels:
            chips = "".join(f'<span class="cs-chip cs-obj">{o}</span>' for o in obj_labels)
            ev_lines.append(f"<b>Objects detected:</b> {chips}")
        if scene_labels:
            for sl, sc in scenes:
                ev_lines.append(f'<span class="cs-chip cs-act">"{sl}"</span> '
                                f'<span class="cs-score">({sc:.2f})</span>')
        body = "<br>".join(ev_lines) if ev_lines else "No high-confidence elements detected above threshold."
    else:
        body = "No high-confidence visual elements detected above the similarity threshold."
    parts.append(section("👁️", "Visual Evidence", body))

    # ── 2. Literal Meaning ────────────────────────────────────────────────────
    if obj_labels:
        items_str = ", ".join(obj_labels[:4])
        lit_desc = (f"The image literally shows: <b>{items_str}</b>. "
                    f"The figurative/literal similarity gap for the top-ranked image is "
                    f"<b>{best_result.get('gap', 0):+.4f}</b> "
                    f"({'favours figurative interpretation' if best_result.get('gap',0) > 0 else 'favours literal interpretation'}).")
    else:
        lit_desc = (f"The top-ranked image has a figurative–literal gap of "
                    f"<b>{best_result.get('gap', 0):+.4f}</b>.")
    lit_prompt_used = perspectives[0] if perspectives else ""
    lit_desc += f"<br><i>Literal probe: \"{lit_prompt_used[:100]}{'…' if len(lit_prompt_used)>100 else ''}\"</i>"
    parts.append(section("📷", "Literal Meaning", lit_desc))

    # ── 3. Commonsense Inference ──────────────────────────────────────────────
    content_words = [w.strip(".,!?\"'") for w in (idiom or sentence).lower().split()
                     if w.lower() not in STOPWORDS]
    assocs = cn_word_associations(content_words[:4])

    inf_lines = []

    # Scene implication sentence
    if scene_labels:
        scene_str = scene_labels[0]
        inf_lines.append(f"Detected scene: <i>\"{scene_str}\"</i>")
        if any(k in scene_str for k in ("exhausted","hard","overwork","burden")):
            implication = "excessive effort, stress, or being overburdened in a role"
        elif any(k in scene_str for k in ("conflict","arguing","fight","rivalry")):
            implication = "tension, disagreement, or rivalry between parties"
        elif any(k in scene_str for k in ("deceptive","sneaky","betray","hiding")):
            implication = "betrayal, hidden motives, or lack of trust"
        elif any(k in scene_str for k in ("success","victory","achieve","reward")):
            implication = "achievement, recognition, or triumph after effort"
        elif any(k in scene_str for k in ("isolated","alone","excluded","ignored")):
            implication = "loneliness, exclusion, or being marginalised"
        elif any(k in scene_str for k in ("danger","risk","threat","vulnerable")):
            implication = "exposure to risk, instability, or a precarious situation"
        elif any(k in scene_str for k in ("lazy","inactive")):
            implication = "idleness, avoidance of effort, or lack of contribution"
        elif any(k in scene_str for k in ("celebrat","festive","joyful")):
            implication = "shared happiness, recognition, or social bonding"
        elif any(k in scene_str for k in ("generous","kind")):
            implication = "selflessness, goodwill, or altruistic behaviour"
        else:
            implication = "a human situation with social or emotional significance"
        inf_lines.append(f"→ In everyday life this typically implies: <b>{implication}</b>")
        inf_lines.append("")

    # ConceptNet associations — now using curated vocabulary
    if assocs:
        inf_lines.append("<b>Conceptual associations (from ConceptNet):</b>")
        for word, neighbours in assocs.items():
            # Group into clusters for readability
            trait_words   = [n for n in neighbours if n in {
                "corrupt","dishonest","deceptive","selfish","greedy","lazy",
                "stubborn","arrogant","cowardly","cruel","irresponsible","naive",
                "honest","generous","loyal","kind","brave","trustworthy","humble"}]
            emotion_words = [n for n in neighbours if n in {
                "anger","fear","sadness","grief","shame","guilt","envy",
                "jealousy","despair","regret","frustration","anxiety","stress",
                "loneliness","disappointment","pride","joy","happiness","hope",
                "satisfaction","excitement","confidence","admiration","relief"}]
            social_words  = [n for n in neighbours if n in {
                "conflict","betrayal","rivalry","deception","manipulation","exclusion",
                "rejection","isolation","dominance","submission","opposition",
                "cooperation","loyalty","trust","friendship","acceptance"}]
            abstract_words= [n for n in neighbours if n in {
                "failure","corruption","harm","injustice","risk","burden","obstacle",
                "loss","damage","danger","decay","waste","crisis","threat","pressure",
                "success","growth","freedom","benefit","reward","progress","opportunity"}]

            grouped = []
            if trait_words:   grouped.append(f"character: <i>{', '.join(trait_words)}</i>")
            if emotion_words: grouped.append(f"emotions: <i>{', '.join(emotion_words)}</i>")
            if social_words:  grouped.append(f"social dynamics: <i>{', '.join(social_words)}</i>")
            if abstract_words:grouped.append(f"abstract concepts: <i>{', '.join(abstract_words)}</i>")
            # fallback: just list them if no group matched
            if not grouped:
                grouped = [f"<i>{', '.join(neighbours[:5])}</i>"]
            inf_lines.append(f'  "<b>{word}</b>" → {" · ".join(grouped)}')
    else:
        inf_lines.append("ConceptNet associations not available for these words.")

    parts.append(section("🧠", "Commonsense Inference", "<br>".join(inf_lines)))

    # ── 4. Best Idiom ─────────────────────────────────────────────────────────
    if idiom:
        ctx_note = (" <i>(de-idiomised contextual prompt used — idiom replaced with its meaning "
                    "to prevent CLIP anchoring to literal words)</i>") if known_idiom else ""
        id_body = (f'<b>Matched idiom:</b> <span class="cs-chip cs-idiom">"{idiom}"</span><br>'
                   f'<b>Meaning:</b> {idiom_meaning}<br>'
                   f'<b>Origin:</b> <i>{idiom_origin}</i>{ctx_note}<br>'
                   f'<b>Figurative probe:</b> <i>"{perspectives[1][:120]}{"…" if len(perspectives[1])>120 else ""}"</i>')
    else:
        id_body = ("No idiom matched in dictionary — analysed as general figurative expression.<br>"
                   f"<b>Sentence used directly as contextual probe.</b>")
    parts.append(section("💡", "Best Idiom", id_body))

    # ── 5. Why It Fits ────────────────────────────────────────────────────────
    fig_sc   = best_result.get("fig_score", 0)
    lit_sc   = best_result.get("lit_score", 0)
    ctx_sc   = best_result.get("ctx_score", 0)
    gap      = best_result.get("gap", 0)
    flgs     = best_result.get("flgs", 0)
    category = best_result.get("category", "Unknown")

    fit_lines = [
        f"The top-ranked image was classified as <b>{category}</b>.",
        f"Scores for the top image:",
        f"  • Figurative probe similarity: <b>{fig_sc:.4f}</b>",
        f"  • Literal probe similarity:    <b>{lit_sc:.4f}</b>",
        f"  • Contextual probe similarity: <b>{ctx_sc:.4f}</b>",
        f"  • Gap (fig − lit):             <b>{gap:+.4f}</b> {'✅ positive → supports figurative' if gap>0 else '⚠️ negative → supports literal'}",
        f"  • FLGS (final score):          <b>{flgs:.4f}</b>",
        f"  • Scoring engine: <i>{scoring_method}</i>",
    ]
    if idiom and obj_labels:
        idiom_words = set(idiom.lower().split())
        overlap = [o for o in obj_labels if any(w in o.lower() for w in idiom_words)]
        if overlap:
            fit_lines.append(f"  • Visual–idiom overlap: objects <b>{', '.join(overlap)}</b> appear in both image and idiom phrase — supporting figurative alignment")
        else:
            fit_lines.append(f"  • No direct object overlap with idiom words — figurative match driven by semantic similarity")
    parts.append(section("✅", "Why It Fits", "<br>".join(fit_lines)))

    # ── 6. Other Candidates Rejected ─────────────────────────────────────────
    candidates = find_idiom_candidates(sentence, exclude=idiom, top_k=3)
    if candidates:
        rej_lines = []
        for phrase, meaning, score in candidates:
            entry = IDIOM_DB.get(phrase, (meaning, "", "", ""))
            rej_lines.append(
                f'<span class="cs-chip cs-rej">"{phrase}"</span> '
                f'(meaning: <i>{meaning}</i>) — '
                f'<b>sentence similarity: {score:.4f}</b> — '
                f'rejected: lower semantic alignment with input sentence than "{idiom or "matched idiom"}"'
            )
        parts.append(section("❌", "Other Candidates Rejected", "<br>".join(rej_lines)))
    else:
        parts.append(section("❌", "Other Candidates Rejected",
                              "No other idiom candidates found in the dictionary."))

    # ── 7. Final Figurative Meaning ───────────────────────────────────────────
    if idiom and idiom_meaning:
        conclusion = (f'The sentence <i>"{sentence}"</i> uses the idiom '
                      f'<b>"{idiom}"</b> to express: <b>{idiom_meaning}</b>. '
                      f'The top-ranked image ({category}) visually grounds this figurative meaning '
                      f'with a figurative confidence of '
                      f'<b>{best_result.get("confidence_pct", 0):.1f}%</b>.')
    else:
        conclusion = (f'The sentence <i>"{sentence}"</i> is treated as a general figurative expression. '
                      f'The top-ranked image ({category}) shows the strongest visual alignment '
                      f'with the figurative interpretation of the scene.')
    parts.append(section("🎯", "Final Figurative Meaning", conclusion))

    return "".join(parts)


def build_reasoning_text(sentence, idiom, idiom_meaning, fig_image_label,
                          all_results, perspectives, known_idiom=False, gap_weight=0.15,
                          scoring_method="FLGS zero-shot"):
    """Generate a human-readable commonsense reasoning explanation."""
    content_words = [w.strip(".,!?\"'") for w in (idiom or sentence).lower().split()
                     if w.lower() not in STOPWORDS]
    assocs = cn_word_associations(content_words[:4])

    lines = []

    # Step 1: Sentence analysis
    lines.append(f"<b>Step 1 — Sentence Analysis</b>")
    lines.append(f"Input sentence: <i>\"{sentence}\"</i>")
    if idiom:
        lines.append(f"Idiom detected: <b>\"{idiom}\"</b>")
        lines.append(f"Figurative meaning: {idiom_meaning}")
    lines.append("")

    # Step 2: IAPD decomposition
    lines.append(f"<b>Step 2 — IAPD Perspective Decomposition</b>")
    prompt_src = "from idiom dictionary (meaning-specific)" if known_idiom else "generic template (idiom not in dictionary)"
    lines.append(f"<i>Prompts generated: {prompt_src}</i>")
    lines.append(f"🔤 <b>Literal prompt:</b> {perspectives[0]}")
    lines.append(f"💡 <b>Figurative prompt:</b> {perspectives[1]}")
    ctx_note = " <i>(idiom replaced with its meaning to prevent literal anchoring)</i>" if known_idiom else ""
    lines.append(f"📝 <b>Contextual prompt:</b> {perspectives[2]}{ctx_note}")
    lines.append("")

    # Step 3: ConceptNet word associations (curated vocab)
    if assocs:
        lines.append(f"<b>Step 3 — ConceptNet Commonsense Associations</b>")
        for word, neighbours in assocs.items():
            lines.append(f'  \u201c<b>{word}</b>\u201d is conceptually linked to: {", ".join(neighbours[:5])}')
        lines.append("")

    # Step 4: Image scoring logic
    prompt_mode = "meaning-specific prompts (idiom in dictionary)" if known_idiom else "generic template (idiom not in dictionary)"
    lines.append(f"<b>Step 4 — Image Scoring Logic</b>")
    lines.append(f"Scoring engine: <b>{scoring_method}</b>")
    lines.append(f"Prompt mode: <i>{prompt_mode}</i>")
    lines.append("For each image, the model computed:")
    lines.append("  • <b>Figurative score</b>: sim(figurative prompt, image)")
    lines.append("  • <b>Literal score</b>: sim(literal prompt, image)")
    lines.append("  • <b>Gap score</b>: figurative − literal  (positive = more figurative)")
    lines.append(f"  • <b>FLGS</b> = context score + {gap_weight} × gap"
                 + (" &nbsp;<i>(gap weighted higher — specific prompts trusted more)</i>" if gap_weight > 0.15 else ""))
    lines.append("")

    # Step 5: Decision
    winner = next((r for r in all_results if r["rank"] == 1), None)
    if winner:
        lines.append(f"<b>Step 5 — Decision</b>")
        lines.append(f"Image ranked #1 (<b>{winner['category']}</b>) because:")
        lines.append(f"  • Highest FLGS score: {winner['flgs']:.4f}")
        lines.append(f"  • Gap (fig−lit): {winner['gap']:+.4f}  {'↑ favours figurative' if winner['gap']>0 else '↓ favours literal'}")
        # Only show objects that passed the detection threshold (non-empty list)
        confirmed_objs = [o for o, sc in winner['objects']]  # threshold already applied
        if confirmed_objs:
            obj_str = ', '.join(confirmed_objs[:3])
            lines.append(f"  • Objects confirmed in image (sim ≥ 0.21): <b>{obj_str}</b>")
            if idiom and any(c in obj_str for c in content_words):
                lines.append(f"  • These objects align with the idiom words — supporting figurative interpretation")
        else:
            lines.append(f"  • No high-confidence objects detected above threshold")

    return "<br>".join(lines)

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-CATEGORISATION
# ──────────────────────────────────────────────────────────────────────────────
def auto_categorise(results):
    """
    Assign categories based on score patterns.
      Figurative     : highest FLGS overall
      Literal        : highest lit_score, clearly low gap
      Partial Literal: medium gap (fig ≈ lit)
      Random         : lowest scores across all metrics
      Distractor     : the remaining one (5th)
    """
    n = len(results)
    # Sort indices by different criteria
    by_flgs = sorted(range(n), key=lambda i: results[i]["flgs"],     reverse=True)
    by_lit  = sorted(range(n), key=lambda i: results[i]["lit_score"],reverse=True)
    by_gap  = sorted(range(n), key=lambda i: abs(results[i]["gap"]))  # smallest gap first = partial lit

    assigned   = {}
    used_slots = set()

    def assign(idx, cat):
        assigned[idx] = cat
        used_slots.add(idx)

    # 1. Figurative = highest FLGS among images where gap > 0 (fig_score > lit_score).
    #    An image with negative gap has lit_score > fig_score — it cannot be figurative.
    #    If NO image has a positive gap (rare edge case), fall back to highest FLGS.
    pos_gap_by_flgs = [i for i in by_flgs if results[i]["gap"] > 0]
    fig_candidates  = pos_gap_by_flgs if pos_gap_by_flgs else by_flgs
    for i in fig_candidates:
        if i not in used_slots:
            assign(i, "Figurative")
            break

    # 2. Literal = highest lit_score AND not already assigned
    #    Also prefer images with clearly negative gap (lit > fig)
    neg_gap_by_lit = [i for i in by_lit if results[i]["gap"] < 0]
    lit_candidates = neg_gap_by_lit if neg_gap_by_lit else by_lit
    for i in lit_candidates:
        if i not in used_slots:
            assign(i, "Literal")
            break

    # 3. Random = lowest FLGS among remaining
    remaining = [i for i in range(n) if i not in used_slots]
    if remaining:
        worst = min(remaining, key=lambda i: results[i]["flgs"])
        assign(worst, "Random")

    # 4. Partial Literal = among remaining, smallest |gap| (most ambiguous)
    remaining = [i for i in range(n) if i not in used_slots]
    if remaining:
        most_ambig = min(remaining, key=lambda i: abs(results[i]["gap"]))
        assign(most_ambig, "Partial Literal")

    # 5. Distractor = whatever is left
    for i in range(n):
        if i not in used_slots:
            assign(i, "Distractor")

    for i, r in enumerate(results):
        r["category"] = assigned.get(i, "Unknown")

    return results

CATEGORY_COLORS = {
    "Figurative":     "#2e7d32",
    "Literal":        "#e65100",
    "Partial Literal":"#1565c0",
    "Random":         "#6d4c41",
    "Distractor":     "#6a1b9a",
}

# ──────────────────────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figurative Language Analyser</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#222}

/* ── Header ── */
.header{background:linear-gradient(135deg,#1a1a2e 0%,#0f3460 100%);
        color:#fff;padding:36px 24px 28px;text-align:center}
.header h1{font-size:26px;font-weight:800;margin-bottom:6px;letter-spacing:-.3px}
.header p{font-size:13px;opacity:.7;max-width:620px;margin:0 auto}

/* ── Model badge strip ── */
.model-strip{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:18px}
.mbadge{background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.25);
        border-radius:20px;padding:4px 13px;font-size:11px;font-weight:600;color:#e8eaf6}
.mbadge.on{background:rgba(76,175,80,.3);border-color:#81c784;color:#c8e6c9}

/* ── Wrapper & Cards ── */
.wrap{max-width:1060px;margin:28px auto;padding:0 18px}
.card{background:#fff;border-radius:14px;padding:24px;margin-bottom:22px;
      box-shadow:0 2px 16px rgba(0,0,0,.07)}
.card h2{font-size:15px;font-weight:700;color:#333;margin-bottom:16px;
         padding-bottom:10px;border-bottom:2px solid #f0f2f5;display:flex;align-items:center;gap:8px}

/* ── Sentence input ── */
input[type=text]{width:100%;padding:13px 16px;border:2px solid #e0e0e0;
  border-radius:10px;font-size:15px;outline:none;transition:.2s;color:#222}
input[type=text]:focus{border-color:#0f3460;box-shadow:0 0 0 3px rgba(15,52,96,.08)}

/* ── Image upload grid ── */
.img-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:14px}
.slot{border:2px dashed #d0d5dd;border-radius:12px;padding:14px 8px;text-align:center;
      cursor:pointer;transition:.2s;background:#fafbfc;min-height:150px;
      display:flex;flex-direction:column;align-items:center;justify-content:center;gap:6px}
.slot:hover{border-color:#0f3460;background:#f0f4ff}
.slot.loaded{border-style:solid;border-color:#81c784;background:#f1f8e9}
.slot label{font-size:11px;color:#999;font-weight:600;letter-spacing:.3px}
.slot .ico{font-size:28px;color:#ccc}
.slot img{max-width:100%;max-height:96px;border-radius:8px;object-fit:cover;display:none;margin-bottom:2px}
input[type=file]{display:none}
.slot .remove{font-size:10px;color:#e53935;cursor:pointer;margin-top:2px;display:none}
.slot.loaded .remove{display:block}
.slot.loaded .ico{display:none}

/* ── Analyse button ── */
.btn{background:linear-gradient(135deg,#0f3460,#1a6eb5);color:#fff;border:none;
     padding:14px 28px;border-radius:10px;font-size:15px;font-weight:700;
     cursor:pointer;width:100%;margin-top:18px;transition:.2s;letter-spacing:.2px}
.btn:hover{opacity:.92;transform:translateY(-1px)}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none}

/* ── Loading ── */
.loading{text-align:center;padding:42px;color:#888;display:none}
.spin{width:42px;height:42px;border:4px solid #eee;border-top-color:#0f3460;
      border-radius:50%;animation:sp .8s linear infinite;margin:0 auto 16px}
@keyframes sp{to{transform:rotate(360deg)}}
.loading-steps{font-size:12px;color:#aaa;margin-top:8px;line-height:2}

/* ── Detection banner ── */
.banner{padding:14px 18px;border-radius:10px;margin-bottom:14px;font-size:14px;font-weight:600;
        display:flex;align-items:center;gap:10px}
.fig-yes{background:#e8f5e9;color:#1b5e20;border-left:4px solid #4caf50}
.fig-no {background:#fff3e0;color:#bf360c;border-left:4px solid #ff9800}
.conf-pill{background:rgba(0,0,0,.08);border-radius:20px;padding:2px 10px;font-size:12px;font-weight:700}

/* ── Idiom box ── */
.idiom-box{background:linear-gradient(135deg,#f3e5f5,#ede7f6);border-radius:12px;
           padding:18px;margin-top:14px;border-left:4px solid #7b1fa2}
.idiom-title{font-size:19px;font-weight:800;color:#4a148c;margin-bottom:6px}
.idiom-meaning{font-size:14px;color:#333;margin-bottom:6px;line-height:1.5}
.idiom-origin{font-size:12px;color:#888;font-style:italic}

/* ── IAPD perspectives ── */
.perspectives{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px}
.pcard{border-radius:10px;padding:12px 14px;font-size:12px;line-height:1.5}
.pcard-lit{background:#e3f2fd;border-left:3px solid #1565c0}
.pcard-fig{background:#e8f5e9;border-left:3px solid #2e7d32}
.pcard-ctx{background:#fff8e1;border-left:3px solid #f57f17}
.pcard-label{font-weight:700;font-size:11px;letter-spacing:.5px;margin-bottom:4px;opacity:.7}

/* ── Results grid ── */
.res-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
.res-card{border-radius:12px;overflow:hidden;border:2px solid #e8e8e8;
          transition:.25s;position:relative;background:#fff}
.res-card:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.1)}
.res-card.rank1{border-color:#4caf50;box-shadow:0 0 0 3px rgba(76,175,80,.2)}
.res-card img{width:100%;height:140px;object-fit:cover;display:block}
.res-info{padding:11px 12px}
.cat-badge{display:inline-block;padding:3px 10px;border-radius:20px;
           font-size:10px;font-weight:800;color:#fff;margin-bottom:7px;letter-spacing:.3px}
.rank-tag{position:absolute;top:9px;left:9px;background:rgba(0,0,0,.65);
          color:#fff;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700}
.best-tag{position:absolute;top:9px;right:9px;background:#4caf50;
          color:#fff;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:800;letter-spacing:.3px}
.gap-tag{position:absolute;bottom:9px;left:9px;border-radius:10px;
         padding:2px 7px;font-size:10px;font-weight:700}
.gap-pos{background:rgba(46,125,50,.15);color:#2e7d32}
.gap-neg{background:rgba(183,28,28,.12);color:#b71c1c}

.bar-wrap{background:#f0f2f5;border-radius:4px;height:6px;overflow:hidden;margin:5px 0 8px}
.bar-fill{height:100%;border-radius:4px;transition:width .6s ease}
.bar-fig{background:linear-gradient(90deg,#43a047,#81c784)}
.bar-lit{background:linear-gradient(90deg,#e53935,#ef9a9a)}
.bar-dis{background:linear-gradient(90deg,#7b1fa2,#ba68c8)}
.bar-ran{background:linear-gradient(90deg,#6d4c41,#a1887f)}
.bar-par{background:linear-gradient(90deg,#1565c0,#64b5f6)}

/* ── Object chips ── */
.obj-list{margin-top:6px;display:flex;flex-wrap:wrap;gap:3px}
.obj-chip{background:#e3f2fd;color:#1565c0;padding:2px 8px;
          border-radius:10px;font-size:10px;font-weight:600}

/* ── Legacy Reasoning box ── */
.reason-box{background:#fffde7;border-radius:12px;padding:20px 22px;font-size:13px;
            line-height:1.85;border-left:4px solid #fbc02d}
.reason-box b{color:#333}

/* ── Commonsense Chain ── */
.chain-wrap{display:flex;flex-direction:column;gap:10px}
.csec{border-radius:10px;overflow:hidden;border:1px solid #e0e0e0}
.csec-hd{padding:9px 14px;font-size:13px;font-weight:700;
         background:linear-gradient(90deg,#f5f5f5,#fafafa);
         border-bottom:1px solid #e0e0e0;color:#222}
.csec-bd{padding:12px 16px;font-size:13px;line-height:1.8;color:#333;background:#fff}
.csec-bd b{color:#1a237e}
.csec-score{color:#888;font-size:11px}
/* Section accent colours */
.csec:nth-child(1) .csec-hd{border-left:4px solid #1565c0}   /* visual evidence */
.csec:nth-child(2) .csec-hd{border-left:4px solid #6d4c41}   /* literal meaning  */
.csec:nth-child(3) .csec-hd{border-left:4px solid #7b1fa2}   /* commonsense      */
.csec:nth-child(4) .csec-hd{border-left:4px solid #ef6c00}   /* best idiom       */
.csec:nth-child(5) .csec-hd{border-left:4px solid #2e7d32}   /* why it fits      */
.csec:nth-child(6) .csec-hd{border-left:4px solid #c62828}   /* rejected         */
.csec:nth-child(7) .csec-hd{border-left:4px solid #0277bd}   /* final meaning    */
/* Chips inside chain */
.cs-chip{display:inline-block;padding:2px 9px;border-radius:10px;
         font-size:11px;font-weight:600;margin:2px 2px}
.cs-obj{background:#e3f2fd;color:#1565c0}
.cs-act{background:#f3e5f5;color:#7b1fa2}
.cs-idiom{background:#fff8e1;color:#f57f17}
.cs-rej{background:#ffebee;color:#c62828}

/* ── Legend ── */
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.leg-item{display:flex;align-items:center;gap:5px;font-size:11px;font-weight:600;color:#555}
.leg-dot{width:11px;height:11px;border-radius:50%}

#results{display:none}
</style>
</head>
<body>

<div class="header">
  <h1>🔍 Figurative Language Understanding</h1>
  <p>Visual Grounding and Commonsense Reasoning for Multimodal Figurative Language</p>
  <div class="model-strip" id="modelStrip">
    <span class="mbadge">CLIP ViT-B/32</span>
    <span class="mbadge">IAPD Prompting</span>
    <span class="mbadge" id="cnBadge">ConceptNet</span>
    <span class="mbadge" id="p4aBadge">Phase 4 Task A</span>
    <span class="mbadge" id="p4bBadge">Phase 4 Task B</span>
  </div>
</div>

<div class="wrap">

  <div class="card">
    <h2>📝 Input Sentence</h2>
    <input type="text" id="sentence"
      placeholder='Type a sentence with an idiom, e.g. "She finally kicked the bucket."'
      value="We need to remove him from the team; he's just one bad apple spoiling the whole bunch">
  </div>

  <div class="card">
    <h2>🖼️ Upload 5 Images <span style="font-weight:400;color:#999;font-size:13px">— model will auto-categorise</span></h2>
    <p style="font-size:13px;color:#888;margin-bottom:4px">
      Upload 5 images related to the idiom — <strong>do not label them</strong>.
      Include: one figurative, one literal, one partial/conceptual, one random, one distractor.
    </p>
    <div class="img-grid" id="imgGrid"></div>
    <button class="btn" onclick="run()" id="btn">🔍 &nbsp;Analyse</button>
  </div>

  <div class="loading" id="loading">
    <div class="spin"></div>
    <div style="font-weight:600;color:#555">Analysing images…</div>
    <div class="loading-steps">
      🔤 Detecting idiom &nbsp;·&nbsp; 💡 Building IAPD prompts &nbsp;·&nbsp;
      🧠 ConceptNet associations<br>
      🖼️ CLIP encoding &nbsp;·&nbsp; 📊 Phase 4 scoring &nbsp;·&nbsp; 🏷️ Object detection
    </div>
  </div>

  <div id="results">

    <div class="card">
      <h2>🗣️ Sentence Analysis</h2>
      <div id="detBanner"></div>
      <div id="idiomBox"></div>
      <div class="perspectives" id="perspBox"></div>
    </div>

    <div class="card">
      <h2>🏆 Image Ranking &amp; Categorisation</h2>
      <p style="font-size:13px;color:#888;margin-bottom:12px">
        Ranked by figurative relevance · Objects confirmed by CLIP zero-shot detection (sim ≥ 0.21)
      </p>
      <div class="legend">
        <span class="leg-item"><span class="leg-dot" style="background:#2e7d32"></span>Figurative</span>
        <span class="leg-item"><span class="leg-dot" style="background:#e65100"></span>Literal</span>
        <span class="leg-item"><span class="leg-dot" style="background:#1565c0"></span>Partial Literal</span>
        <span class="leg-item"><span class="leg-dot" style="background:#6d4c41"></span>Random</span>
        <span class="leg-item"><span class="leg-dot" style="background:#6a1b9a"></span>Distractor</span>
      </div>
      <div class="res-grid" id="resGrid"></div>
    </div>

    <div class="card">
      <h2>🧩 Visual Commonsense Reasoning Chain</h2>
      <p style="font-size:13px;color:#888;margin-bottom:12px">
        Step-by-step analysis: visual evidence → literal scene → commonsense inference →
        idiom match → evidence → rejected candidates → final figurative meaning
      </p>
      <div class="chain-wrap" id="chainBox"></div>
    </div>

    <div class="card" id="legacyCard">
      <h2>🧠 Detailed Reasoning Log</h2>
      <div class="reason-box" id="reasonBox"></div>
    </div>

  </div>
</div>

<script>
const N = 5;
const urls = Array(N).fill(null);

const CAT_COLORS = {
  "Figurative":"#2e7d32","Literal":"#e65100",
  "Partial Literal":"#1565c0","Random":"#6d4c41","Distractor":"#6a1b9a"
};
const BAR_CLS = {
  "Figurative":"bar-fig","Literal":"bar-lit",
  "Partial Literal":"bar-par","Random":"bar-ran","Distractor":"bar-dis"
};

// ── Build image upload slots ───────────────────────────────────────────────
const grid = document.getElementById('imgGrid');
for(let i=0;i<N;i++){
  grid.innerHTML += `
    <div class="slot" id="slot${i}" onclick="pickFile(${i})">
      <img id="prev${i}" alt="">
      <div class="ico">🖼️</div>
      <label>Image ${i+1}</label>
      <span class="remove" onclick="clearSlot(event,${i})">✕ Remove</span>
      <input type="file" id="f${i}" accept="image/*" onchange="preview(${i})">
    </div>`;
}

function pickFile(i){ document.getElementById(`f${i}`).click(); }

function clearSlot(e, i){
  e.stopPropagation();
  urls[i] = null;
  document.getElementById(`f${i}`).value = '';
  const img = document.getElementById(`prev${i}`);
  img.src=''; img.style.display='none';
  const slot = document.getElementById(`slot${i}`);
  slot.querySelector('.ico').style.display='';
  slot.classList.remove('loaded');
}

function preview(i){
  const file = document.getElementById(`f${i}`).files[0];
  if(!file) return;
  const r = new FileReader();
  r.onload = e => {
    const img = document.getElementById(`prev${i}`);
    img.src = e.target.result;
    img.style.display = 'block';
    urls[i] = e.target.result;
    const slot = document.getElementById(`slot${i}`);
    slot.querySelector('.ico').style.display = 'none';
    slot.classList.add('loaded');
  };
  r.readAsDataURL(file);
}

// ── Load model status badges ───────────────────────────────────────────────
fetch('/status').then(r=>r.json()).then(s=>{
  if(s.conceptnet) document.getElementById('cnBadge').classList.add('on');
  if(s.phase4_a){
    const b = document.getElementById('p4aBadge');
    b.classList.add('on');
    b.textContent = `Phase 4-A ${s.phase4_a_acc}%`;
  }
  if(s.phase4_b){
    const b = document.getElementById('p4bBadge');
    b.classList.add('on');
    b.textContent = `Phase 4-B ${s.phase4_b_acc}%`;
  }
}).catch(()=>{});

// ── Main analysis call ─────────────────────────────────────────────────────
async function run(){
  const sentence = document.getElementById('sentence').value.trim();
  if(!sentence){ alert('Please enter a sentence.'); return; }
  const files = Array.from({length:N},(_,i)=>document.getElementById(`f${i}`).files[0]);
  if(files.some(f=>!f)){ alert('Please upload all 5 images.'); return; }

  document.getElementById('btn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const fd = new FormData();
  fd.append('sentence', sentence);
  files.forEach((f,i) => fd.append(`image${i}`, f));

  try{
    const resp = await fetch('/analyse', {method:'POST', body:fd});
    const data = await resp.json();
    render(data);
  } catch(e){ alert('Error: ' + e); }
  finally{
    document.getElementById('btn').disabled = false;
    document.getElementById('loading').style.display = 'none';
  }
}

// ── Render results ─────────────────────────────────────────────────────────
function render(data){

  // Detection banner
  const b = document.getElementById('detBanner');
  if(data.is_figurative){
    b.className = 'banner fig-yes';
    b.innerHTML = `✅ <strong>Figurative / Idiomatic language detected</strong>
      <span class="conf-pill">Confidence: ${(data.confidence*100).toFixed(1)}%</span>`;
  } else {
    b.className = 'banner fig-no';
    b.innerHTML = `⚠️ <strong>Likely literal language</strong>
      <span class="conf-pill">Confidence: ${(data.confidence*100).toFixed(1)}%</span>`;
  }

  // Idiom box
  const ib = document.getElementById('idiomBox');
  if(data.idiom){
    ib.innerHTML = `<div class="idiom-box">
      <div class="idiom-title">Idiom: &ldquo;${data.idiom}&rdquo;</div>
      <div class="idiom-meaning">📖 Meaning: <strong>${data.idiom_meaning}</strong></div>
      <div class="idiom-origin">🏛️ Origin: ${data.idiom_origin||'—'}</div>
    </div>`;
  } else {
    ib.innerHTML = `<div class="idiom-box">
      <div class="idiom-title" style="color:#555;font-size:15px">
        No known idiom matched — analysing as general figurative expression</div>
    </div>`;
  }

  // IAPD perspectives panel
  const pb = document.getElementById('perspBox');
  if(data.perspectives){
    pb.innerHTML = `
      <div class="pcard pcard-lit">
        <div class="pcard-label">🔤 LITERAL PROMPT</div>${data.perspectives.literal}</div>
      <div class="pcard pcard-fig">
        <div class="pcard-label">💡 FIGURATIVE PROMPT</div>${data.perspectives.figurative}</div>
      <div class="pcard pcard-ctx">
        <div class="pcard-label">📝 CONTEXTUAL PROMPT</div>${data.perspectives.contextual}</div>`;
  }

  // Results cards
  const rg = document.getElementById('resGrid');
  rg.innerHTML = '';
  [...data.results].sort((a,b)=>a.rank-b.rank).forEach(r=>{
    const imgUrl = urls[r.index];
    const col    = CAT_COLORS[r.category] || '#555';
    const barCls = BAR_CLS[r.category]    || 'bar-ran';
    const objs   = (r.objects||[]).slice(0,4).map(o=>`<span class="obj-chip">${o[0]}</span>`).join('');
    const gapSign = r.gap >= 0;
    rg.innerHTML += `
      <div class="res-card ${r.rank===1?'rank1':''}">
        <span class="rank-tag">#${r.rank}</span>
        ${r.rank===1 ? '<span class="best-tag">✓ Best Match</span>' : ''}
        <img src="${imgUrl}" alt="">
        <div class="res-info">
          <span class="cat-badge" style="background:${col}">${r.category.toUpperCase()}</span>
          <div style="font-size:11px;color:#888;margin-bottom:2px">Match: ${r.confidence_pct}%</div>
          <div class="bar-wrap">
            <div class="${barCls}" style="height:100%;border-radius:4px;width:${r.confidence_pct}%"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:#bbb;margin-bottom:4px">
            <span>Fig ${r.fig_score.toFixed(3)}</span>
            <span>Lit ${r.lit_score.toFixed(3)}</span>
            <span class="${gapSign?'':''}">Gap ${r.gap>=0?'+':''}${r.gap.toFixed(3)}</span>
          </div>
          <div class="obj-list">${objs||'<span style="font-size:10px;color:#ccc">no objects above threshold</span>'}</div>
        </div>
      </div>`;
  });

  // Commonsense chain (structured 7-section)
  if(data.chain) {
    document.getElementById('chainBox').innerHTML = data.chain;
  }

  // Legacy reasoning log
  document.getElementById('reasonBox').innerHTML = data.reasoning;

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def status():
    """Return which models are loaded — shown in the UI header."""
    return jsonify({
        "phase4_a": PHASE4_MODEL  is not None,
        "phase4_b": PHASE4_MODEL_B is not None,
        "conceptnet": CN is not None,
        "phase4_a_acc": 73.3 if PHASE4_MODEL  is not None else None,
        "phase4_b_acc": 80.0 if PHASE4_MODEL_B is not None else None,
    })


@app.route("/analyse", methods=["POST"])
def analyse():
    sentence = request.form.get("sentence","").strip()
    if not sentence:
        return jsonify({"error":"No sentence"}), 400

    img_bytes = []
    for i in range(5):
        f = request.files.get(f"image{i}")
        if f is None:
            return jsonify({"error":f"Missing image{i}"}), 400
        img_bytes.append(f.read())

    # ── Idiom lookup ──────────────────────────────────────────────────────────
    idiom, idiom_meaning, idiom_origin, vis_fig, vis_lit = lookup_idiom(sentence)

    # ── Figurative detection ──────────────────────────────────────────────────
    # Use multiple probe prompts and average for more robust detection
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
    # Confidence: scale difference to 0-100% range meaningfully
    diff = fs - ls
    conf = float(min(abs(diff) * 20, 1.0))   # scale so small diffs show reasonable %

    # ── IAPD perspectives — use visual prompts from dictionary when known ─────
    perspectives = iapd_prompts(sentence, idiom, vis_fig, vis_lit, idiom_meaning)
    lit_emb = encode_text(perspectives[0])
    fig_emb = encode_text(perspectives[1])
    ctx_emb = encode_text(perspectives[2])

    # ── Per-image scoring ─────────────────────────────────────────────────────
    # When idiom is known (meaning-specific visual prompts), the gap between
    # fig_score and lit_score is a reliable discriminator — weight it more.
    # When idiom is unknown (generic template), rely mostly on ctx_score.
    gap_weight = 0.40 if (vis_fig is not None) else 0.15

    img_embs = []   # collect for Phase 4
    results = []
    for i, ib in enumerate(img_bytes):
        img_emb = encode_image_bytes(ib)
        img_embs.append(img_emb)

        ctx_sc = float((ctx_emb * img_emb).sum())
        fig_sc = float((fig_emb * img_emb).sum())
        lit_sc = float((lit_emb * img_emb).sum())
        gap    = fig_sc - lit_sc
        flgs   = ctx_sc + gap_weight * gap

        objects = detect_objects(img_emb, top_k=4)
        scenes  = detect_scene(img_emb, top_k=3)

        results.append({
            "index":    i,
            "flgs":     flgs,
            "ctx_sc":   ctx_sc,   # kept for Phase 4 signal
            "ctx_score":ctx_sc,
            "fig_score":fig_sc,
            "lit_score":lit_sc,
            "gap":      gap,
            "objects":  objects,
            "scenes":   scenes,
            "category": "",
        })

    # ── Phase 4 override: use CaptionFusedRanker when checkpoint is loaded ────
    # The trained model scored 73.3% val vs Phase 2's 66.7% by adding caption
    # signals: sim(fig,caption), sim(ctx,caption), sim(lit,caption), sim(cap,img)
    # Since the live demo has no pre-computed captions, we synthesise them:
    #   caption_emb ≈ 0.5*(fig_emb + img_emb) — blends visual + figurative meaning
    # This is a proxy that keeps the caption signal active without a captioning model.
    scoring_method = "FLGS zero-shot"
    if PHASE4_MODEL is not None and len(img_embs) == 5:
        try:
            imgs_t  = torch.stack(img_embs).unsqueeze(0)          # (1, 5, 512)
            # Synthesise per-image caption embedding as visual–figurative blend
            fig_exp = fig_emb.unsqueeze(0).unsqueeze(0).expand(1, 5, -1)
            cap_t   = F.normalize(0.5 * imgs_t + 0.5 * fig_exp, dim=-1)

            lit_exp = lit_emb.unsqueeze(0)   # (1, 512)
            fig_exp1= fig_emb.unsqueeze(0)
            ctx_exp = ctx_emb.unsqueeze(0)

            # 8 scalar signals matching phase4_model.extract_signals()
            s_ctx_img = (ctx_exp.unsqueeze(1) * imgs_t).sum(dim=-1)   # (1,5)
            s_fig_img = (fig_exp1.unsqueeze(1) * imgs_t).sum(dim=-1)
            s_lit_img = (lit_exp.unsqueeze(1) * imgs_t).sum(dim=-1)
            s_fig_cap = (fig_exp1.unsqueeze(1) * cap_t).sum(dim=-1)
            s_ctx_cap = (ctx_exp.unsqueeze(1) * cap_t).sum(dim=-1)
            s_lit_cap = (lit_exp.unsqueeze(1) * cap_t).sum(dim=-1)
            s_cap_img = (cap_t * imgs_t).sum(dim=-1)
            gap_t     = (s_fig_img - s_lit_img)

            signals = torch.stack([
                s_ctx_img, s_fig_img, s_lit_img,
                s_fig_cap, s_ctx_cap, s_lit_cap,
                s_cap_img, gap_t
            ], dim=-1)  # (1, 5, 8)

            with torch.no_grad():
                p4_scores = PHASE4_MODEL(signals).squeeze(0)  # (5,)

            # Blend Phase 4 score with FLGS for stability (0.6 P4 + 0.4 FLGS)
            flgs_raw = torch.tensor([r["flgs"] for r in results])
            flgs_norm = F.normalize(flgs_raw.unsqueeze(0), dim=-1).squeeze(0)
            p4_norm   = F.normalize(p4_scores.unsqueeze(0), dim=-1).squeeze(0)
            blended   = 0.6 * p4_norm + 0.4 * flgs_norm

            for i, r in enumerate(results):
                r["flgs"] = float(blended[i])
            scoring_method = "Phase 4 CaptionFused (73.3%) + FLGS blend"
        except Exception as _p4e:
            scoring_method = f"FLGS zero-shot (P4 error: {_p4e})"

    # ── Auto-categorise ───────────────────────────────────────────────────────
    results = auto_categorise(results)

    # ── Rank ──────────────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["flgs"], reverse=True)
    raw = torch.tensor([r["flgs"] for r in results])
    probs = F.softmax(raw * 10, dim=0).tolist()
    for rank, (r, p) in enumerate(zip(results, probs)):
        r["rank"]           = rank + 1
        r["confidence_pct"] = round(p * 100, 1)

    # ── Commonsense reasoning chain (new structured 7-section output) ─────────
    best_result = results[0] if results else {}
    best_objects = best_result.get("objects", [])
    best_scenes  = best_result.get("scenes",  [])

    reasoning = build_commonsense_chain(
        sentence, idiom, idiom_meaning, idiom_origin,
        best_result, results, perspectives,
        objects=best_objects,
        scenes=best_scenes,
        gap_weight=gap_weight,
        scoring_method=scoring_method,
        known_idiom=(vis_fig is not None),
    )

    # Also keep the legacy step-by-step for the old "Commonsense Reasoning Explanation"
    legacy_reasoning = build_reasoning_text(
        sentence, idiom, idiom_meaning,
        best_result.get("category", ""),
        results, perspectives,
        known_idiom=(vis_fig is not None),
        gap_weight=gap_weight,
        scoring_method=scoring_method
    )

    return jsonify({
        "is_figurative":   is_fig,
        "confidence":      float(conf),
        "idiom":           idiom,
        "idiom_meaning":   idiom_meaning,
        "idiom_origin":    idiom_origin,
        "perspectives":    {"literal": perspectives[0],
                            "figurative": perspectives[1],
                            "contextual": perspectives[2]},
        "results":         results,
        "reasoning":       legacy_reasoning,
        "chain":           reasoning,          # new structured commonsense chain
    })


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Figurative Language Understanding — Demo")
    print("="*52)
    print("  Open: http://127.0.0.1:5000")
    print("="*52+"\n")
    app.run(debug=False, host="127.0.0.1", port=5000)
