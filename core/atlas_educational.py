"""atlas_educational.py

Broad educational / knowledge-domain word-count metrics.

This is an educational counterpart to the original niche-specific metric module.
Instead of hundreds of tiny one-word metrics, it uses large domain lexicons that
can measure how strongly a transcript draws vocabulary from major academic,
scientific, technical, cultural, and educational fields.

Designed to follow the same registration pattern as the original niche module:

    register_educational_metrics(metric_config, classifier_cls)

The supplied classifier class is expected to provide the same shared helpers
used by the original module, especially:

    _get_runtime_text_lower(tokens)
    _compute_pattern_count(tokens, patterns)

This file intentionally favors broad, interpretable categories over hundreds
of isolated word counters. A single metric such as "biology vocabulary density"
therefore represents a substantial vocabulary field rather than one word.

Important interpretation note:
A vocabulary count indicates topical/lexical presence, not that a speaker is
actually teaching, endorsing, practicing, or deeply knowledgeable about a
subject. Domain terms can be polysemous and should be interpreted in context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ============================================================================
# LARGE EDUCATIONAL / ACADEMIC LEXICONS
# ============================================================================

EDUCATIONAL_LEXICONS: Dict[str, set] = {

    # ------------------------------------------------------------------------
    # GENERAL EDUCATION
    # ------------------------------------------------------------------------
    "education": {
        "education", "educational", "school", "classroom", "teacher", "student",
        "learner", "learning", "teaching", "lesson", "course", "curriculum",
        "syllabus", "assignment", "homework", "exam", "examination", "quiz",
        "assessment", "grading", "grade", "rubric", "lecture", "seminar",
        "workshop", "tutorial", "instruction", "instructor", "professor",
        "academic", "academia", "university", "college", "campus", "degree",
        "major", "minor", "undergraduate", "graduate", "doctoral", "research",
        "study", "textbook", "chapter", "reading", "writing", "discussion",
        "class", "coursework", "knowledge", "concept", "skill", "practice",
        "feedback", "peer", "pedagogy", "didactic", "literacy", "scholarship",
        "thesis", "dissertation", "citation", "source", "evidence", "methodology",
        "experiment", "laboratory", "lab", "faculty", "department", "enrollment",
        "credential", "certificate", "degree", "distance", "online", "mooc",
    },

    # ------------------------------------------------------------------------
    # MATHEMATICS
    # ------------------------------------------------------------------------
    "mathematics": {
        "mathematics", "math", "mathematical", "number", "integer", "whole",
        "natural", "real", "complex", "rational", "irrational", "fraction",
        "decimal", "percentage", "ratio", "proportion", "equation", "inequality",
        "variable", "constant", "coefficient", "expression", "function",
        "domain", "range", "graph", "coordinate", "geometry", "algebra",
        "calculus", "derivative", "integral", "limit", "continuity", "sequence",
        "series", "matrix", "vector", "determinant", "tensor", "proof", "theorem",
        "lemma", "corollary", "axiom", "conjecture", "set", "subset", "union",
        "intersection", "cardinality", "probability", "combinatorics",
        "permutation", "combination", "topology", "analysis", "optimization",
        "differential", "equation", "logarithm", "exponential", "trigonometry",
        "sine", "cosine", "tangent", "angle", "polygon", "triangle", "circle",
        "sphere", "statistics", "random", "distribution", "mean", "median",
        "variance", "standard", "deviation", "regression", "hypothesis",
    },

    # ------------------------------------------------------------------------
    # STATISTICS / DATA
    # ------------------------------------------------------------------------
    "statistics_data": {
        "statistics", "statistical", "data", "dataset", "variable", "observation",
        "sample", "population", "parameter", "estimate", "estimator", "mean",
        "median", "mode", "variance", "covariance", "correlation", "distribution",
        "normal", "binomial", "poisson", "uniform", "probability", "likelihood",
        "bayesian", "frequentist", "inference", "hypothesis", "null", "alternative",
        "significance", "pvalue", "confidence", "interval", "effect", "size",
        "power", "sampling", "bias", "outlier", "quartile", "percentile",
        "histogram", "scatterplot", "boxplot", "regression", "linear", "logistic",
        "classification", "clustering", "forecast", "time", "series", "survey",
        "questionnaire", "experiment", "control", "treatment", "replicate",
        "reproducibility", "bootstrap", "resampling", "standardization",
        "normalization", "aggregation", "visualization", "dashboard", "analytics",
        "correlation", "causation", "confounder", "missing", "imputation",
    },

    # ------------------------------------------------------------------------
    # COMPUTER SCIENCE / PROGRAMMING
    # ------------------------------------------------------------------------
    "computer_science": {
        "computer", "computing", "computer science", "algorithm", "algorithms",
        "program", "programming", "code", "coding", "software", "hardware",
        "developer", "development", "application", "app", "system", "systems",
        "architecture", "processor", "cpu", "gpu", "memory", "storage", "binary",
        "bit", "byte", "variable", "function", "method", "class", "object",
        "array", "list", "tuple", "dictionary", "hash", "tree", "graph", "node",
        "edge", "stack", "queue", "recursion", "iteration", "complexity",
        "runtime", "compiler", "interpreter", "syntax", "semantic", "debug",
        "debugging", "exception", "library", "framework", "api", "database",
        "sql", "query", "schema", "table", "index", "transaction", "backend",
        "frontend", "fullstack", "server", "client", "network", "protocol",
        "operating", "linux", "unix", "shell", "terminal", "git", "repository",
        "version", "software", "testing", "unit", "integration", "objectoriented",
        "functional", "concurrency", "parallel", "distributed", "virtualization",
        "container", "docker", "cloud", "computational", "simulation",
    },

    # ------------------------------------------------------------------------
    # AI / MACHINE LEARNING / NLP
    # ------------------------------------------------------------------------
    "ai_machine_learning": {
        "artificial", "intelligence", "ai", "machine", "learning", "deep",
        "neural", "network", "model", "training", "train", "validation", "test",
        "dataset", "feature", "label", "target", "embedding", "transformer",
        "attention", "token", "tokenization", "language", "nlp", "natural",
        "language", "processing", "classification", "regression", "clustering",
        "supervised", "unsupervised", "reinforcement", "agent", "reward",
        "policy", "gradient", "descent", "backpropagation", "loss", "optimizer",
        "epoch", "batch", "neuron", "layer", "activation", "relu", "softmax",
        "convolution", "recurrent", "lstm", "generative", "generation",
        "inference", "prediction", "probability", "accuracy", "precision",
        "recall", "fscore", "roc", "auc", "overfitting", "underfitting",
        "regularization", "dropout", "hyperparameter", "fine-tuning",
        "prompt", "llm", "language", "model", "corpus", "sentiment", "semantic",
        "syntax", "token", "vector", "dimension", "representation",
    },

    # ------------------------------------------------------------------------
    # CYBERSECURITY
    # ------------------------------------------------------------------------
    "cybersecurity": {
        "security", "cybersecurity", "cyber", "privacy", "authentication",
        "authorization", "identity", "password", "credential", "encryption",
        "decryption", "cryptography", "cipher", "hash", "certificate", "tls",
        "ssl", "firewall", "malware", "virus", "worm", "trojan", "ransomware",
        "phishing", "spoofing", "scam", "vulnerability", "exploit", "attack",
        "threat", "risk", "intrusion", "detection", "prevention", "endpoint",
        "network", "packet", "port", "protocol", "access", "permission",
        "privilege", "sandbox", "secure", "authentication", "authorization",
        "zero", "trust", "breach", "incident", "forensics", "audit", "compliance",
        "integrity", "availability", "confidentiality", "hashing", "signature",
        "key", "public", "private", "secret", "token", "session", "injection",
        "sql", "xss", "csrf", "backup", "recovery",
    },

    # ------------------------------------------------------------------------
    # PHYSICS
    # ------------------------------------------------------------------------
    "physics": {
        "physics", "physical", "matter", "energy", "force", "motion", "mass",
        "weight", "velocity", "speed", "acceleration", "momentum", "impulse",
        "gravity", "friction", "pressure", "density", "work", "power", "heat",
        "temperature", "thermodynamics", "entropy", "wave", "frequency",
        "wavelength", "amplitude", "sound", "light", "optics", "photon",
        "electron", "proton", "neutron", "atom", "quantum", "quantum",
        "mechanics", "relativity", "spacetime", "field", "magnetic", "electric",
        "charge", "voltage", "current", "resistance", "circuit", "capacitor",
        "inductor", "radiation", "nuclear", "particle", "plasma", "fluid",
        "mechanics", "kinetic", "potential", "equilibrium", "torque", "angular",
        "rotation", "oscillation", "resonance", "refraction", "reflection",
        "diffraction", "interference",
    },

    # ------------------------------------------------------------------------
    # CHEMISTRY
    # ------------------------------------------------------------------------
    "chemistry": {
        "chemistry", "chemical", "atom", "molecule", "element", "compound",
        "mixture", "solution", "solute", "solvent", "reaction", "reactant",
        "product", "catalyst", "enzyme", "acid", "base", "ph", "salt", "ion",
        "cation", "anion", "bond", "covalent", "ionic", "metallic", "electron",
        "proton", "neutron", "isotope", "periodic", "table", "mole", "molar",
        "concentration", "stoichiometry", "equilibrium", "oxidation",
        "reduction", "redox", "combustion", "precipitate", "organic", "inorganic",
        "carbon", "hydrocarbon", "alkane", "alkene", "alkyne", "aromatic",
        "polymer", "monomer", "functional", "group", "spectroscopy", "titration",
        "chromatography", "distillation", "solubility", "thermochemistry",
        "kinetics", "activation", "energy", "enthalpy", "entropy",
    },

    # ------------------------------------------------------------------------
    # BIOLOGY
    # ------------------------------------------------------------------------
    "biology": {
        "biology", "biological", "organism", "cell", "tissue", "organ",
        "organism", "species", "population", "community", "ecosystem", "biome",
        "evolution", "evolutionary", "natural", "selection", "adaptation",
        "gene", "genome", "genetic", "dna", "rna", "chromosome", "mutation",
        "allele", "heredity", "inheritance", "protein", "enzyme", "metabolism",
        "photosynthesis", "respiration", "mitosis", "meiosis", "membrane",
        "cytoplasm", "nucleus", "ribosome", "organelle", "mitochondria",
        "chloroplast", "bacteria", "archaea", "fungus", "fungi", "plant",
        "animal", "ecology", "behavior", "physiology", "anatomy", "homeostasis",
        "reproduction", "embryology", "development", "taxonomy", "classification",
        "phylogeny", "molecular", "cellular", "microbiology", "biochemistry",
        "biotechnology", "specimen", "microscope", "fieldwork", "habitat",
        "predator", "prey", "food", "web", "symbiosis", "parasite",
    },

    # ------------------------------------------------------------------------
    # GENETICS / MOLECULAR BIOLOGY
    # ------------------------------------------------------------------------
    "genetics_molecular_biology": {
        "genetics", "genetic", "gene", "genome", "genomic", "dna", "rna",
        "mrna", "trna", "rrna", "chromosome", "chromatin", "nucleotide",
        "base", "adenine", "thymine", "cytosine", "guanine", "uracil",
        "replication", "transcription", "translation", "codon", "anticodon",
        "mutation", "variant", "allele", "genotype", "phenotype", "dominant",
        "recessive", "heterozygous", "homozygous", "inheritance", "hereditary",
        "epigenetics", "methylation", "expression", "promoter", "enhancer",
        "operon", "plasmid", "sequencing", "pcr", "crispr", "cas9", "cloning",
        "gene", "editing", "recombinant", "protein", "amino", "acid", "peptide",
        "ribosome", "transcriptome", "proteome", "bioinformatics", "alignment",
        "blast", "genomics", "variant", "annotation",
    },

    # ------------------------------------------------------------------------
    # MICROBIOLOGY / IMMUNOLOGY
    # ------------------------------------------------------------------------
    "microbiology_immunology": {
        "microbiology", "microbe", "microbial", "bacteria", "bacterium",
        "archaea", "fungus", "fungi", "yeast", "virus", "viral", "pathogen",
        "infection", "infectious", "culture", "colony", "agar", "microscope",
        "antibiotic", "antimicrobial", "resistance", "immune", "immunity",
        "immunology", "antibody", "antigen", "lymphocyte", "tcell", "bcell",
        "macrophage", "neutrophil", "cytokine", "inflammation", "vaccine",
        "vaccination", "pathology", "disease", "host", "transmission",
        "epidemic", "pandemic", "outbreak", "sterile", "aseptic", "incubation",
        "replication", "capsid", "genome", "plasmid", "biofilm", "microbiome",
        "gut", "commensal", "symbiosis", "virulence", "toxin", "diagnostic",
    },

    # ------------------------------------------------------------------------
    # MEDICINE / HEALTH / PUBLIC HEALTH
    # ------------------------------------------------------------------------
    "medicine_health": {
        "medicine", "medical", "health", "healthcare", "patient", "clinical",
        "doctor", "physician", "nurse", "nursing", "hospital", "clinic",
        "diagnosis", "diagnostic", "symptom", "disease", "disorder", "condition",
        "treatment", "therapy", "therapeutic", "medicine", "drug", "medication",
        "dose", "dosage", "prescription", "surgery", "surgical", "anatomy",
        "physiology", "pathology", "laboratory", "blood", "heart", "cardiac",
        "lung", "pulmonary", "brain", "neurological", "kidney", "renal",
        "liver", "hepatic", "immune", "infection", "chronic", "acute",
        "prevention", "screening", "risk", "mortality", "morbidity", "epidemiology",
        "public", "health", "nutrition", "rehabilitation", "mental", "wellness",
        "evidence", "trial", "clinical", "placebo", "randomized", "cohort",
        "case", "control", "biomarker", "prognosis", "etiology",
    },

    # ------------------------------------------------------------------------
    # FITNESS / EXERCISE / SPORTS SCIENCE
    # ------------------------------------------------------------------------
    "fitness_exercise": {
        "fitness", "exercise", "workout", "training", "athlete", "athletic",
        "sport", "sports", "strength", "conditioning", "cardio", "aerobic",
        "anaerobic", "endurance", "stamina", "mobility", "flexibility",
        "balance", "coordination", "agility", "speed", "power", "force",
        "muscle", "muscular", "skeleton", "bone", "joint", "tendon", "ligament",
        "heart", "cardiovascular", "respiratory", "recovery", "warmup",
        "cooldown", "stretching", "lifting", "squat", "deadlift", "press",
        "bench", "repetition", "rep", "set", "volume", "intensity", "load",
        "progressive", "overload", "hypertrophy", "strength", "conditioning",
        "running", "cycling", "swimming", "walking", "injury", "rehab",
        "biomechanics", "kinesiology", "physiology", "performance",
    },

    # ------------------------------------------------------------------------
    # NUTRITION
    # ------------------------------------------------------------------------
    "nutrition": {
        "nutrition", "nutritional", "food", "diet", "dietary", "calorie",
        "calories", "energy", "protein", "carbohydrate", "carbs", "fat",
        "lipid", "fiber", "vitamin", "mineral", "micronutrient", "macronutrient",
        "iron", "calcium", "sodium", "potassium", "magnesium", "zinc",
        "folate", "b12", "vitamin", "hydration", "water", "metabolism",
        "digestion", "absorption", "gut", "microbiome", "appetite", "satiety",
        "deficiency", "supplement", "nutrition", "malnutrition", "obesity",
        "diabetes", "cholesterol", "glucose", "insulin", "glycemic", "dietary",
        "meal", "serving", "portion", "ingredient", "nutrient", "wholegrain",
        "vegetable", "fruit", "legume", "grain", "processed", "fermented",
    },

    # ------------------------------------------------------------------------
    # NEUROSCIENCE / PSYCHOLOGY
    # ------------------------------------------------------------------------
    "neuroscience_psychology": {
        "psychology", "psychological", "behavior", "behavioral", "cognition",
        "cognitive", "brain", "neuron", "neural", "synapse", "neurotransmitter",
        "dopamine", "serotonin", "memory", "attention", "perception", "learning",
        "emotion", "motivation", "decision", "language", "consciousness",
        "sleep", "dream", "stress", "anxiety", "depression", "therapy",
        "clinical", "personality", "developmental", "social", "experimental",
        "behaviorism", "conditioning", "reinforcement", "stimulus", "response",
        "working", "memory", "recall", "recognition", "bias", "heuristic",
        "intelligence", "iq", "neuroscience", "neurology", "cortex",
        "hippocampus", "amygdala", "frontal", "temporal", "sensory", "motor",
        "plasticity", "neuroplasticity", "lesion", "electrophysiology",
    },

    # ------------------------------------------------------------------------
    # EARTH SCIENCE / GEOLOGY
    # ------------------------------------------------------------------------
    "earth_science_geology": {
        "earth", "geology", "geological", "rock", "mineral", "crystal",
        "sediment", "sedimentary", "igneous", "metamorphic", "magma", "lava",
        "volcano", "volcanic", "earthquake", "fault", "plate", "tectonics",
        "continental", "oceanic", "crust", "mantle", "core", "erosion",
        "weathering", "deposition", "stratum", "strata", "fossil", "soil",
        "landform", "mountain", "valley", "canyon", "basin", "glacier",
        "ice", "hydrology", "groundwater", "aquifer", "river", "watershed",
        "sedimentology", "petrology", "mineralogy", "geophysics", "seismic",
        "seismology", "geochemistry", "paleoclimate", "carbon", "cycle",
    },

    # ------------------------------------------------------------------------
    # PALEONTOLOGY / EVOLUTIONARY HISTORY
    # ------------------------------------------------------------------------
    "paleontology": {
        "paleontology", "palaeontology", "fossil", "fossils", "fossilization",
        "dinosaur", "dinosaurs", "theropod", "sauropod", "ornithischian",
        "trilobite", "ammonite", "mammoth", "megafauna", "vertebrate",
        "invertebrate", "specimen", "fossiliferous", "stratigraphy", "stratum",
        "formation", "sediment", "sedimentary", "geologic", "geological",
        "extinction", "mass", "evolution", "evolutionary",
        "ancestor", "ancestral", "lineage", "phylogeny", "clade", "taxon",
        "taxonomy", "morphology", "anatomy", "skeleton", "skull", "bone",
        "tooth", "trackway", "coprolite", "amber", "preservation", "museum",
        "fossil", "dating", "radiometric", "jurassic", "cretaceous", "triassic",
        "permian", "cenozoic", "mesozoic", "paleozoic", "prehistoric",
    },

    # ------------------------------------------------------------------------
    # ASTRONOMY / SPACE
    # ------------------------------------------------------------------------
    "astronomy_space": {
        "astronomy", "astronomical", "astrophysics", "space", "universe",
        "galaxy", "galaxies", "star", "stellar", "planet", "planetary",
        "moon", "lunar", "sun", "solar", "orbit", "orbital", "gravity",
        "black", "hole", "neutron", "pulsar", "quasar", "supernova",
        "nebula", "cosmos", "cosmology", "big", "bang", "expansion",
        "dark", "matter", "dark", "energy", "exoplanet", "asteroid",
        "comet", "meteor", "meteorite", "telescope", "observatory", "spectroscopy",
        "redshift", "lightyear", "parsec", "mass", "luminosity", "wavelength",
        "radiation", "relativity", "quantum", "rocket", "spacecraft",
        "satellite", "mission", "launch", "planet", "atmosphere",
    },

    # ------------------------------------------------------------------------
    # ENVIRONMENTAL SCIENCE / CLIMATE
    # ------------------------------------------------------------------------
    "environment_climate": {
        "environment", "environmental", "ecology", "ecosystem", "biodiversity",
        "habitat", "conservation", "sustainability", "sustainable", "climate",
        "climate", "warming", "greenhouse", "carbon", "emissions", "co2",
        "methane", "atmosphere", "weather", "temperature", "precipitation",
        "drought", "flood", "storm", "hurricane", "wildfire", "pollution",
        "air", "water", "soil", "contamination", "waste", "recycling",
        "renewable", "solar", "wind", "hydro", "energy", "fossil", "fuel",
        "deforestation", "forest", "ocean", "marine", "conservation",
        "restoration", "species", "extinction", "carbon", "sequestration",
        "emission", "mitigation", "adaptation", "resilience", "ecosystem",
    },

    # ------------------------------------------------------------------------
    # OCEANOGRAPHY / MARINE SCIENCE
    # ------------------------------------------------------------------------
    "oceanography_marine": {
        "ocean", "marine", "sea", "seawater", "saltwater", "coast", "coastal",
        "shore", "beach", "tide", "tidal", "current", "wave", "coral", "reef",
        "plankton", "phytoplankton", "zooplankton", "fish", "whale", "dolphin",
        "shark", "crustacean", "mollusk", "ecosystem", "habitat", "benthic",
        "pelagic", "abyssal", "deep", "trench", "continental", "shelf",
        "salinity", "temperature", "pressure", "buoyancy", "circulation",
        "upwelling", "downwelling", "sediment", "seafloor", "hydrothermal",
        "vent", "oceanography", "marine", "biology", "acidification", "carbon",
        "current", "bathymetry", "sonar", "submersible", "remotely",
    },

    # ------------------------------------------------------------------------
    # ENGINEERING
    # ------------------------------------------------------------------------
    "engineering": {
        "engineering", "engineer", "design", "prototype", "system", "mechanical",
        "electrical", "civil", "chemical", "aerospace", "industrial",
        "structural", "materials", "manufacturing", "fabrication", "machine",
        "mechanism", "component", "assembly", "load", "stress", "strain",
        "force", "torque", "bearing", "gear", "motor", "actuator", "sensor",
        "circuit", "voltage", "current", "resistance", "power", "signal",
        "control", "feedback", "automation", "robotics", "cad", "simulation",
        "prototype", "tolerance", "specification", "efficiency", "optimization",
        "safety", "failure", "reliability", "testing", "manufacturing",
        "process", "pipeline", "infrastructure", "bridge", "building",
        "construction", "thermal", "fluid", "aerodynamics",
    },

    # ------------------------------------------------------------------------
    # ELECTRONICS / EMBEDDED / ROBOTICS
    # ------------------------------------------------------------------------
    "electronics_embedded_robotics": {
        "electronics", "electronic", "circuit", "microcontroller", "microprocessor",
        "embedded", "firmware", "hardware", "sensor", "actuator", "motor",
        "servo", "stepper", "gpio", "pin", "voltage", "current", "resistance",
        "capacitor", "inductor", "diode", "transistor", "mosfet", "led",
        "breadboard", "solder", "pcb", "oscilloscope", "multimeter", "signal",
        "analog", "digital", "pwm", "uart", "spi", "i2c", "serial", "bluetooth",
        "wifi", "wireless", "radio", "robot", "robotics", "autonomous",
        "controller", "control", "feedback", "encoder", "lidar", "camera",
        "computer", "vision", "navigation", "actuator", "mechanical", "robot",
    },

    # ------------------------------------------------------------------------
    # SOCIAL SCIENCE / SOCIOLOGY
    # ------------------------------------------------------------------------
    "sociology": {
        "sociology", "sociological", "society", "social", "culture", "community",
        "group", "institution", "organization", "class", "status", "identity",
        "norm", "value", "belief", "role", "interaction", "network",
        "inequality", "stratification", "socialization", "family", "education",
        "religion", "work", "labor", "migration", "urban", "rural", "population",
        "demography", "gender", "race", "ethnicity", "class", "power",
        "deviance", "crime", "law", "politics", "economy", "capital",
        "community", "survey", "interview", "ethnography", "qualitative",
        "quantitative", "institution", "modernity", "globalization", "media",
        "social", "movement", "collective", "behavior", "network", "cohort",
    },

    # ------------------------------------------------------------------------
    # ANTHROPOLOGY
    # ------------------------------------------------------------------------
    "anthropology": {
        "anthropology", "anthropological", "culture", "cultural", "ethnography",
        "ethnographic", "society", "community", "kinship", "family", "ritual",
        "religion", "myth", "language", "linguistic", "artifact", "archaeology",
        "excavation", "human", "hominin", "hominid", "evolution", "adaptation",
        "population", "migration", "foraging", "agriculture", "pastoralism",
        "settlement", "tradition", "custom", "identity", "gender", "exchange",
        "economy", "political", "organization", "colonialism", "globalization",
        "material", "culture", "fieldwork", "participant", "observation",
        "interview", "oral", "history", "indigenous", "diaspora", "comparative",
        "biological", "anthropology", "linguistic", "anthropology",
    },

    # ------------------------------------------------------------------------
    # ECONOMICS / BUSINESS / FINANCE
    # ------------------------------------------------------------------------
    "economics": {
        "economics", "economic", "economy", "market", "markets", "supply",
        "demand", "price", "cost", "revenue", "profit", "loss", "consumer",
        "producer", "firm", "industry", "competition", "monopoly", "oligopoly",
        "elasticity", "inflation", "deflation", "unemployment", "employment",
        "gdp", "growth", "recession", "fiscal", "monetary", "tax", "trade",
        "export", "import", "currency", "interest", "rate", "bank", "central",
        "investment", "capital", "labor", "wage", "productivity", "externality",
        "public", "goods", "utility", "equilibrium", "macroeconomics",
        "microeconomics", "econometrics", "regression", "forecast", "policy",
    },

    "business_finance": {
        "business", "management", "company", "corporation", "organization",
        "startup", "entrepreneur", "entrepreneurship", "strategy", "marketing",
        "sales", "customer", "product", "service", "brand", "revenue", "profit",
        "margin", "budget", "finance", "financial", "accounting", "asset",
        "liability", "equity", "cash", "flow", "investment", "portfolio",
        "stock", "bond", "market", "dividend", "interest", "loan", "credit",
        "debt", "risk", "return", "valuation", "balance", "sheet", "income",
        "statement", "audit", "tax", "supply", "chain", "operations", "project",
        "management", "leadership", "human", "resources", "recruiting",
    },

    # ------------------------------------------------------------------------
    # POLITICAL SCIENCE / CIVICS
    # ------------------------------------------------------------------------
    "political_science_civics": {
        "politics", "political", "government", "governance", "state", "nation",
        "citizen", "citizenship", "democracy", "republic", "election", "voting",
        "vote", "ballot", "candidate", "party", "legislature", "legislative",
        "executive", "judicial", "court", "constitution", "law", "policy",
        "public", "administration", "bureaucracy", "federal", "state",
        "local", "municipal", "congress", "senate", "parliament", "president",
        "minister", "diplomacy", "international", "relations", "conflict",
        "war", "peace", "treaty", "sovereignty", "ideology", "liberal",
        "conservative", "socialism", "capitalism", "authoritarian", "rights",
        "freedom", "justice", "representation", "constitution", "institution",
        "comparative", "politics", "public", "opinion",
    },

    # ------------------------------------------------------------------------
    # LAW / CRIMINOLOGY
    # ------------------------------------------------------------------------
    "law_criminology": {
        "law", "legal", "lawyer", "attorney", "court", "judge", "jury",
        "trial", "case", "statute", "legislation", "constitution", "regulation",
        "contract", "liability", "tort", "criminal", "crime", "criminality",
        "evidence", "testimony", "witness", "prosecution", "defense",
        "defendant", "plaintiff", "verdict", "appeal", "sentence", "prison",
        "correction", "police", "investigation", "forensics", "justice",
        "rights", "property", "copyright", "patent", "trademark", "privacy",
        "precedent", "jurisdiction", "federal", "civil", "misdemeanor",
        "felony", "fraud", "theft", "assault", "ethics", "compliance",
    },

    # ------------------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------------------
    "history": {
        "history", "historical", "ancient", "medieval", "modern", "period",
        "era", "century", "millennium", "empire", "kingdom", "dynasty",
        "civilization", "society", "culture", "war", "battle", "revolution",
        "reform", "colonial", "colonialism", "imperial", "independence",
        "migration", "trade", "economy", "religion", "monarchy", "republic",
        "democracy", "archival", "archive", "document", "primary", "source",
        "secondary", "source", "chronology", "timeline", "artifact", "manuscript",
        "letter", "diary", "memoir", "biography", "historiography", "periodization",
        "causation", "continuity", "change", "context", "historian",
    },

    # ------------------------------------------------------------------------
    # ARCHAEOLOGY
    # ------------------------------------------------------------------------
    "archaeology": {
        "archaeology", "archaeological", "archaeologist", "excavation",
        "excavate", "site", "artifact", "artefact", "pottery", "ceramic",
        "sherd", "stone", "tool", "weapon", "burial", "grave", "tomb",
        "settlement", "village", "city", "ruin", "monument", "temple",
        "stratigraphy", "stratum", "layer", "dating", "radiocarbon",
        "chronology", "typology", "material", "culture", "fieldwork",
        "survey", "dig", "context", "provenience", "provenance", "artifact",
        "architecture", "midden", "shell", "bone", "human", "remains",
        "preservation", "conservation", "museum", "heritage", "site",
    },

    # ------------------------------------------------------------------------
    # PHILOSOPHY / LOGIC / ETHICS
    # ------------------------------------------------------------------------
    "philosophy_logic_ethics": {
        "philosophy", "philosophical", "philosopher", "ethics", "ethical",
        "morality", "moral", "virtue", "justice", "reason", "reasoning",
        "logic", "logical", "argument", "premise", "conclusion", "valid",
        "sound", "fallacy", "deduction", "induction", "epistemology",
        "knowledge", "belief", "truth", "justification", "skepticism",
        "metaphysics", "ontology", "existence", "reality", "mind", "consciousness",
        "free", "will", "determinism", "causation", "identity", "person",
        "aesthetics", "beauty", "meaning", "language", "semantics", "pragmatics",
        "political", "philosophy", "utilitarianism", "deontology", "virtue",
        "nihilism", "existentialism", "rationalism", "empiricism", "dualism",
    },

    # ------------------------------------------------------------------------
    # RELIGION / THEOLOGY / MYTHOLOGY
    # ------------------------------------------------------------------------
    "religion_theology_mythology": {
        "religion", "religious", "theology", "theological", "faith", "belief",
        "god", "gods", "deity", "divine", "sacred", "holy", "spiritual",
        "spirituality", "worship", "prayer", "ritual", "ceremony", "temple",
        "church", "mosque", "synagogue", "shrine", "monastery", "scripture",
        "text", "bible", "quran", "torah", "vedas", "sutra", "hadith",
        "tradition", "doctrine", "dogma", "sect", "denomination", "theology",
        "prophet", "saint", "angel", "devil", "afterlife", "salvation",
        "sin", "virtue", "creation", "myth", "mythology", "legend", "folklore",
        "comparative", "religion", "atheism", "agnosticism", "pilgrimage",
        "meditation", "monastic", "reformation",
    },

    # ------------------------------------------------------------------------
    # LINGUISTICS / LANGUAGE
    # ------------------------------------------------------------------------
    "linguistics": {
        "linguistics", "linguistic", "language", "speech", "speaker", "word",
        "vocabulary", "lexicon", "grammar", "syntax", "semantics", "pragmatics",
        "phonetics", "phonology", "morphology", "prosody", "intonation",
        "syllable", "phoneme", "morpheme", "utterance", "discourse", "conversation",
        "corpus", "corpora", "transcript", "translation", "bilingual",
        "multilingual", "dialect", "accent", "sociolinguistic", "psycholinguistic",
        "neurolinguistic", "historical", "comparative", "etymology", "register",
        "style", "genre", "coherence", "cohesion", "ambiguity", "metaphor",
        "idiom", "collocation", "frequency", "token", "type", "ttr", "entropy",
        "predictability", "embedding", "sentiment", "discourse", "narrative",
    },

    # ------------------------------------------------------------------------
    # LITERATURE / WRITING / RHETORIC
    # ------------------------------------------------------------------------
    "literature_writing_rhetoric": {
        "literature", "literary", "novel", "poem", "poetry", "poet", "fiction",
        "nonfiction", "story", "narrative", "character", "plot", "setting",
        "theme", "motif", "symbol", "metaphor", "simile", "imagery", "tone",
        "voice", "style", "genre", "prose", "verse", "stanza", "rhyme",
        "meter", "rhythm", "alliteration", "irony", "satire", "allegory",
        "tragedy", "comedy", "drama", "play", "theater", "rhetoric", "argument",
        "persuasion", "audience", "author", "reader", "narrator", "point",
        "view", "analysis", "interpretation", "criticism", "close", "reading",
        "essay", "thesis", "paragraph", "draft", "revision", "editing",
        "citation", "bibliography",
    },

    # ------------------------------------------------------------------------
    # CULTURE / MEDIA / COMMUNICATION
    # ------------------------------------------------------------------------
    "culture_media_communication": {
        "culture", "cultural", "media", "communication", "journalism", "news",
        "press", "newspaper", "broadcast", "television", "radio", "podcast",
        "video", "film", "movie", "documentary", "social", "media", "platform",
        "audience", "viewer", "reader", "creator", "content", "message",
        "representation", "identity", "community", "fandom", "genre", "popular",
        "culture", "advertising", "publicity", "rhetoric", "narrative",
        "discourse", "framing", "agenda", "medium", "digital", "internet",
        "communication", "interpersonal", "mass", "media", "visual", "literacy",
        "misinformation", "disinformation", "source", "credibility", "context",
    },

    # ------------------------------------------------------------------------
    # ART / ART HISTORY / DESIGN
    # ------------------------------------------------------------------------
    "art_design": {
        "art", "artist", "artistic", "artwork", "painting", "drawing",
        "sculpture", "photography", "printmaking", "ceramic", "textile",
        "architecture", "design", "graphic", "illustration", "composition",
        "color", "colour", "hue", "value", "saturation", "contrast", "perspective",
        "proportion", "form", "shape", "line", "texture", "pattern", "medium",
        "canvas", "brush", "pigment", "gallery", "museum", "exhibition",
        "curator", "aesthetic", "style", "movement", "modernism", "realism",
        "impressionism", "surrealism", "cubism", "abstract", "renaissance",
        "baroque", "portrait", "landscape", "still", "life", "typography",
        "layout", "interface", "user", "experience", "ux", "ui",
    },

    # ------------------------------------------------------------------------
    # MUSIC
    # ------------------------------------------------------------------------
    "music": {
        "music", "musical", "song", "composer", "composition", "melody",
        "harmony", "rhythm", "beat", "tempo", "meter", "key", "scale",
        "chord", "progression", "note", "pitch", "frequency", "octave",
        "interval", "instrument", "vocal", "voice", "singer", "performance",
        "orchestra", "ensemble", "band", "choir", "conductor", "concert",
        "recording", "studio", "microphone", "audio", "sound", "acoustic",
        "electronic", "classical", "jazz", "blues", "rock", "pop", "folk",
        "genre", "timbre", "dynamics", "crescendo", "decrescendo", "notation",
        "staff", "clef", "treble", "bass", "counterpoint", "improvisation",
        "arrangement", "production", "mixing", "mastering",
    },

    # ------------------------------------------------------------------------
    # GEOGRAPHY / GIS
    # ------------------------------------------------------------------------
    "geography_gis": {
        "geography", "geographic", "geographical", "map", "mapping", "cartography",
        "cartographic", "gis", "spatial", "location", "place", "region",
        "territory", "landscape", "population", "demography", "urban",
        "rural", "city", "settlement", "migration", "climate", "weather",
        "latitude", "longitude", "coordinate", "projection", "scale", "legend",
        "topography", "elevation", "terrain", "river", "mountain", "coast",
        "boundary", "border", "country", "continent", "distribution",
        "density", "remote", "sensing", "satellite", "geocoding", "dataset",
        "layer", "raster", "vector", "polygon", "point", "line", "shapefile",
        "geospatial", "analysis", "location",
    },

    # ------------------------------------------------------------------------
    # AGRICULTURE / FOOD SCIENCE
    # ------------------------------------------------------------------------
    "agriculture_food_science": {
        "agriculture", "agricultural", "farming", "farm", "crop", "soil",
        "seed", "plant", "harvest", "irrigation", "fertilizer", "pesticide",
        "herbicide", "livestock", "animal", "pasture", "grazing", "breeding",
        "yield", "cultivation", "horticulture", "botany", "greenhouse",
        "sustainable", "organic", "food", "food", "science", "processing",
        "fermentation", "preservation", "pasteurization", "microbiology",
        "pathogen", "contamination", "safety", "shelf", "life", "packaging",
        "ingredient", "protein", "carbohydrate", "fat", "vitamin", "mineral",
        "sensory", "flavor", "taste", "texture", "agroecology", "crop",
    },

    # ------------------------------------------------------------------------
    # GENERAL SCIENTIFIC METHOD / RESEARCH
    # ------------------------------------------------------------------------
    "scientific_method_research": {
        "science", "scientific", "research", "researcher", "question",
        "hypothesis", "theory", "model", "experiment", "experimental",
        "observation", "observe", "evidence", "data", "dataset", "measurement",
        "measure", "variable", "control", "treatment", "sample", "population",
        "replicate", "replication", "reproducibility", "method", "methodology",
        "protocol", "procedure", "result", "results", "analysis", "analyze",
        "conclusion", "interpretation", "peer", "review", "journal", "paper",
        "publication", "citation", "source", "literature", "systematic",
        "review", "meta", "analysis", "statistical", "significance", "bias",
        "error", "uncertainty", "limitation", "validity", "reliability",
        "causality", "correlation", "mechanism", "prediction", "replicate",
    },

    # ------------------------------------------------------------------------
    # ACADEMIC / RESEARCH LANGUAGE
    # ------------------------------------------------------------------------
    "academic_discourse": {
        "define", "definition", "concept", "principle", "theory", "framework",
        "approach", "method", "methodology", "evidence", "example", "illustrate",
        "demonstrate", "explain", "compare", "contrast", "evaluate", "analyze",
        "synthesize", "interpret", "infer", "derive", "calculate", "estimate",
        "assume", "hypothesis", "claim", "argument", "reason", "because",
        "therefore", "however", "although", "whereas", "specifically",
        "generally", "typically", "approximately", "significant", "relevant",
        "context", "implication", "limitation", "finding", "result", "evidence",
        "source", "reference", "citation", "literature", "research", "study",
        "according", "suggest", "indicate", "demonstrate", "establish",
        "distinguish", "identify", "describe", "summarize",
    },

    # ------------------------------------------------------------------------
    # TUTORIAL / EXPLANATION LANGUAGE
    # ------------------------------------------------------------------------
    "tutorial_explanation": {
        "tutorial", "guide", "lesson", "learn", "learning", "explain",
        "explanation", "understand", "understanding", "step", "steps",
        "example", "examples", "demonstrate", "demonstration", "walkthrough",
        "practice", "exercise", "question", "answer", "solution", "problem",
        "concept", "definition", "meaning", "why", "how", "what", "remember",
        "tip", "tips", "note", "important", "first", "second", "next", "finally",
        "beginner", "advanced", "intermediate", "introduction", "overview",
        "summary", "review", "recap", "mistake", "common", "challenge",
        "strategy", "method", "technique", "apply", "application", "real",
        "world", "example", "practice", "test", "check", "answer",
    },

    # ------------------------------------------------------------------------
    # CODING TUTORIAL LANGUAGE / SOFTWARE DEVELOPMENT
    # ------------------------------------------------------------------------
    "programming_tutorial": {
        "python", "java", "javascript", "typescript", "c", "cpp", "csharp",
        "rust", "go", "ruby", "php", "swift", "kotlin", "code", "coding",
        "programming", "script", "function", "variable", "loop", "for",
        "while", "if", "else", "elif", "condition", "boolean", "string",
        "integer", "float", "array", "list", "tuple", "dictionary", "object",
        "class", "inheritance", "method", "parameter", "argument", "return",
        "import", "module", "package", "library", "framework", "api", "json",
        "http", "request", "response", "server", "client", "database", "sql",
        "query", "html", "css", "dom", "react", "node", "express", "debug",
        "error", "exception", "test", "testing", "git", "github", "repository",
        "commit", "branch", "merge", "deploy", "deployment", "terminal",
    },

    # ------------------------------------------------------------------------
    # GENERAL STEM / ENGINEERING CONCEPTS
    # ------------------------------------------------------------------------
    "stem_general": {
        "science", "technology", "engineering", "mathematics", "stem",
        "experiment", "model", "system", "measurement", "data", "variable",
        "energy", "force", "mass", "motion", "structure", "function",
        "process", "material", "design", "analysis", "calculation",
        "equation", "graph", "simulation", "algorithm", "optimization",
        "accuracy", "precision", "error", "uncertainty", "scale", "ratio",
        "pattern", "structure", "evidence", "observation", "hypothesis",
        "theory", "prediction", "result", "test", "prototype", "technology",
        "device", "system", "model", "solution", "problem", "experiment",
    },
}


# ============================================================================
# PHRASE / DISCOURSE PATTERNS
# ============================================================================

EDUCATIONAL_PHRASES: Dict[str, List[str]] = {
    "definition_phrases": [
        r"\bdefined\s+as\b",
        r"\bmeans\s+that\b",
        r"\brefers\s+to\b",
        r"\bin\s+other\s+words\b",
        r"\bthe\s+definition\s+of\b",
        r"\bknown\s+as\b",
    ],
    "explanation_phrases": [
        r"\bthis\s+means\b",
        r"\bthat\s+means\b",
        r"\bin\s+other\s+words\b",
        r"\bthe\s+reason\s+is\b",
        r"\bthe\s+idea\s+is\b",
        r"\bwhat\s+this\s+means\b",
        r"\bto\s+understand\s+this\b",
        r"\bthe\s+key\s+point\b",
    ],
    "example_phrases": [
        r"\bfor\s+example\b",
        r"\bfor\s+instance\b",
        r"\bsuch\s+as\b",
        r"\bconsider\s+the\s+case\b",
        r"\bas\s+an\s+example\b",
        r"\blet'?s\s+say\b",
        r"\bsuppose\s+that\b",
        r"\bimagine\s+that\b",
    ],
    "instruction_phrases": [
        r"\bfirst\b",
        r"\bnext\b",
        r"\bthen\b",
        r"\bfinally\b",
        r"\bstep\s+\d+\b",
        r"\bclick\s+on\b",
        r"\bgo\s+to\b",
        r"\bselect\b",
        r"\bchoose\b",
        r"\benter\b",
        r"\btype\b",
        r"\binstall\b",
        r"\brun\s+the\b",
    ],
    "reasoning_phrases": [
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bthus\b",
        r"\bhence\b",
        r"\bwhich\s+means\b",
        r"\bthis\s+implies\b",
        r"\bwe\s+can\s+infer\b",
        r"\bit\s+follows\b",
    ],
    "comparison_phrases": [
        r"\bin\s+contrast\b",
        r"\bon\s+the\s+other\s+hand\b",
        r"\bwhereas\b",
        r"\bcompared\s+to\b",
        r"\bsimilar\s+to\b",
        r"\bdifferent\s+from\b",
        r"\bthe\s+difference\s+between\b",
        r"\bthe\s+same\s+as\b",
    ],
    "uncertainty_phrases": [
        r"\bmay\s+be\b",
        r"\bmight\s+be\b",
        r"\bcould\s+be\b",
        r"\bpossibly\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bunlikely\b",
        r"\bapproximately\b",
        r"\broughly\b",
        r"\bit\s+is\s+unclear\b",
        r"\bevidence\s+suggests\b",
    ],
    "research_phrases": [
        r"\bthe\s+study\s+found\b",
        r"\bresearch\s+shows\b",
        r"\bresearch\s+suggests\b",
        r"\bthe\s+authors\b",
        r"\baccording\s+to\b",
        r"\bthe\s+evidence\b",
        r"\bthe\s+results\b",
        r"\bthe\s+findings\b",
        r"\bpeer\s+reviewed\b",
    ],
    "questioning_phrases": [
        r"\bwhy\s+does\b",
        r"\bwhy\s+do\b",
        r"\bhow\s+does\b",
        r"\bhow\s+do\b",
        r"\bwhat\s+is\b",
        r"\bwhat\s+are\b",
        r"\bwhat\s+does\b",
        r"\bwhat\s+would\s+happen\b",
        r"\bcan\s+we\s+explain\b",
    ],
}


# ============================================================================
# COMPILED PATTERNS
# ============================================================================

PHRASE_PATTERNS: Dict[str, List[re.Pattern]] = {
    name: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for name, patterns in EDUCATIONAL_PHRASES.items()
}


# ============================================================================
# METRIC CONFIGURATION
# ============================================================================

EDUCATIONAL_METRIC_CONFIGS: Dict[str, Dict[str, Any]] = {}

for _field in EDUCATIONAL_LEXICONS:
    _safe_name = re.sub(r"[^a-z0-9]+", "_", _field.lower()).strip("_")
    EDUCATIONAL_METRIC_CONFIGS[f"{_safe_name}_lexicon_density"] = {
        "name": f"{_field.replace('_', ' ').title()} Vocabulary",
        "title_suffix": f"Broad {_field.replace('_', ' ')} vocabulary count",
        "compute_method": "_compute_educational_lexicon",
        "lexicon_key": _field,
        "category": "Academic & Knowledge Domains",
    }

for _phrase_group in PHRASE_PATTERNS:
    EDUCATIONAL_METRIC_CONFIGS[f"{_phrase_group}_count"] = {
        "name": f"{_phrase_group.replace('_', ' ').title()} Count",
        "title_suffix": f"Educational discourse: {_phrase_group.replace('_', ' ')}",
        "compute_method": "_compute_educational_phrase",
        "phrase_key": _phrase_group,
        "category": "Educational Discourse",
    }

EDUCATIONAL_METRIC_CONFIGS.update({
    "academic_domain_breadth": {
        "name": "Academic Domain Breadth",
        "title_suffix": "Number of distinct academic/knowledge fields represented",
        "compute_method": "_compute_domain_breadth",
        "category": "Educational Overview",
    },
    "stem_vocabulary_density": {
        "name": "STEM Vocabulary Density",
        "title_suffix": "Broad STEM terminology count",
        "compute_method": "_compute_stem_density",
        "category": "Educational Overview",
    },
    "science_vocabulary_density": {
        "name": "Science Vocabulary Density",
        "title_suffix": "Broad scientific terminology count",
        "compute_method": "_compute_science_density",
        "category": "Educational Overview",
    },
    "tutorial_language_density": {
        "name": "Tutorial Language Density",
        "title_suffix": "Words associated with explanation and instruction",
        "compute_method": "_compute_tutorial_density",
        "category": "Educational Overview",
    },
})


# ============================================================================
# MIXIN
# ============================================================================

class EducationalMetricsMixin:
    """Compute methods for the broad educational metrics."""

    def _educational_runtime_words(self, tokens) -> List[str]:
        if tokens is None:
            return []

        try:
            text = self._get_runtime_text_lower(tokens)
            if isinstance(text, str):
                return re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
        except Exception:
            pass

        words = []
        for token in tokens:
            token = str(token).lower()
            words.extend(re.findall(r"[a-zA-ZÀ-ÿ]+", token))
        return words

    def _compute_educational_lexicon(
        self, tokens, lexicon_key: Optional[str] = None
    ) -> float:
        words = self._educational_runtime_words(tokens)
        lexicon = EDUCATIONAL_LEXICONS.get(lexicon_key or "", set())
        return float(sum(1 for word in words if word in lexicon))

    def _compute_educational_phrase(
        self, tokens, phrase_key: Optional[str] = None
    ) -> float:
        try:
            text = self._get_runtime_text_lower(tokens)
        except Exception:
            text = " ".join(map(str, tokens)).lower()

        patterns = PHRASE_PATTERNS.get(phrase_key or "", [])
        return float(sum(len(pattern.findall(text)) for pattern in patterns))

    def _compute_domain_breadth(self, tokens) -> float:
        words = set(self._educational_runtime_words(tokens))
        return float(sum(
            1 for lexicon in EDUCATIONAL_LEXICONS.values()
            if words.intersection(lexicon)
        ))

    def _compute_stem_density(self, tokens) -> float:
        stem_fields = (
            "mathematics", "statistics_data", "computer_science",
            "ai_machine_learning", "physics", "chemistry", "biology",
            "genetics_molecular_biology", "earth_science_geology",
            "astronomy_space", "environment_climate", "engineering",
            "electronics_embedded_robotics", "stem_general",
        )
        words = self._educational_runtime_words(tokens)
        lexicon = set().union(*(EDUCATIONAL_LEXICONS[name] for name in stem_fields))
        return float(sum(1 for word in words if word in lexicon))

    def _compute_science_density(self, tokens) -> float:
        science_fields = (
            "physics", "chemistry", "biology", "genetics_molecular_biology",
            "microbiology_immunology", "medicine_health",
            "neuroscience_psychology", "earth_science_geology",
            "paleontology", "astronomy_space", "environment_climate",
            "oceanography_marine", "agriculture_food_science",
            "scientific_method_research",
        )
        words = self._educational_runtime_words(tokens)
        lexicon = set().union(*(EDUCATIONAL_LEXICONS[name] for name in science_fields))
        return float(sum(1 for word in words if word in lexicon))

    def _compute_tutorial_density(self, tokens) -> float:
        tutorial_fields = (
            "education", "academic_discourse", "tutorial_explanation",
            "programming_tutorial", "scientific_method_research",
        )
        words = self._educational_runtime_words(tokens)
        lexicon = set().union(*(EDUCATIONAL_LEXICONS[name] for name in tutorial_fields))
        return float(sum(1 for word in words if word in lexicon))


# ============================================================================
# REGISTRATION
# ============================================================================

def register_educational_metrics(metric_config: Any, classifier_cls: Any) -> None:
    """
    Register all educational metrics into the existing metric registry.

    This mirrors the registration strategy in the original niche module:
    mutate the existing registry and attach compute methods to the supplied
    classifier class.
    """
    metric_config.METRICS.update(EDUCATIONAL_METRIC_CONFIGS)

    for method_name, method in EducationalMetricsMixin.__dict__.items():
        if (
            method_name == "_educational_runtime_words"
            or method_name.startswith("_compute_")
        ):
            setattr(classifier_cls, method_name, method)


# ============================================================================
# OPTIONAL STANDALONE ACCESS
# ============================================================================

def get_educational_lexicons() -> Dict[str, set]:
    """Return the domain lexicons for inspection or external tooling."""
    return EDUCATIONAL_LEXICONS


def get_educational_metric_names() -> List[str]:
    """Return all metric IDs generated by this module."""
    return list(EDUCATIONAL_METRIC_CONFIGS)