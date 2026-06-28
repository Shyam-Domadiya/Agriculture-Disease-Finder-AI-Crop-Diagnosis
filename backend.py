"""
Agriculture Disease Finder — FastAPI Backend
=============================================
REST API serving crop disease predictions from crop-specific
Bayesian Networks using pgmpy.

Crops Supported: Tomato, Wheat, Potato, Corn.
Endpoints:
    POST /predict   — Predict disease from crop type + symptoms + environment
    GET  /health    — Health check
    GET  /model-info — Model metadata for all crops

Run with:  python backend.py
    or:    uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


# ------------------------------------------------------------------
# Global Constants
# ------------------------------------------------------------------
ENVIRON_STATES = ["Low", "Normal", "High"]
SYMPTOM_STATES = ["None", "Mild", "Severe"]
SYMPTOM_NODES = [
    "Wilting",
    "WhiteSpots",
    "YellowLeaves",
    "StuntedGrowth",
    "BlackSpots",
    "LeafCurl",
    "FoulSmell",
]
ENVIRON_NODES = ["Temperature", "Humidity"]

# ------------------------------------------------------------------
# Crop-Specific Diseases & Information
# ------------------------------------------------------------------
CROP_DEFS = {
    "tomato": {
        "name": "Tomato",
        "emoji": "🍅",
        "diseases": [
            "Healthy",
            "Early_Blight",
            "Late_Blight",
            "Bacterial_Wilt",
            "Mosaic_Virus",
            "Septoria_Leaf_Spot",
        ],
        "disease_info": {
            "Healthy": {
                "emoji": "🟢",
                "color": "#4CAF50",
                "description": "Your tomato plant looks healthy and thriving!",
                "tip": "Maintain regular watering and monitor for any pest activity.",
                "organic": ["Maintain healthy soil biology with quality compost tea"],
                "chemical": [],
                "prevention": ["Rotate solanaceous crops every 3 years", "Keep watering schedules regular and base-focused"],
            },
            "Early_Blight": {
                "emoji": "🟡",
                "color": "#FFC107",
                "description": "Fungal disease causing brown/black target-like concentric rings on older leaves.",
                "tip": "Prune lower leaves, avoid overhead irrigation, and apply copper fungicide.",
                "organic": [
                    "Prune lower leaves up to 12 inches high to prevent soil splash",
                    "Apply thick organic mulch (straw or bark) around the base",
                    "Spray organic neem oil or copper octanoate solution"
                ],
                "chemical": [
                    "Apply chlorothalonil or mancozeb fungicide at first sign of spots",
                    "Rotate chemical classes over seasons to prevent fungal resistance"
                ],
                "prevention": [
                    "Water strictly at the base of the plant, not overhead",
                    "Space tomato plants at least 24 inches apart to promote airflow"
                ],
            },
            "Late_Blight": {
                "emoji": "🔴",
                "color": "#F44336",
                "description": "Devastating pathogen causing dark water-soaked spots and fuzzy white mold under leaves.",
                "tip": "Destroy infected plants immediately, improve air flow, and apply systemic fungicide.",
                "organic": [
                    "Remove and deeply bury or burn infected plants immediately",
                    "Do not compost infected foliage (spores can survive)",
                    "Spray preventative bio-fungicides containing Bacillus subtilis"
                ],
                "chemical": [
                    "Apply chlorothalonil, copper fungicide, or systemic metalaxyl at first local warning"
                ],
                "prevention": [
                    "Plant only certified disease-free seeds or starts",
                    "Avoid overhead watering during high-humidity periods"
                ],
            },
            "Bacterial_Wilt": {
                "emoji": "🟣",
                "color": "#9C27B0",
                "description": "Soil-borne bacteria blocking water transport, causing rapid green wilting of stems.",
                "tip": "Remove infected plants, rotate crops, sterilize pruning tools, and improve drainage.",
                "organic": [
                    "Pull and discard infected plants immediately to prevent soil spread",
                    "Amend soil with rich organic compost to improve drainage",
                    "Apply beneficial Trichoderma fungi to root zones at planting"
                ],
                "chemical": [
                    "No effective chemical cures exist; focus on soil sanitization and pH management"
                ],
                "prevention": [
                    "Maintain soil pH between 6.2 and 6.8",
                    "Avoid physical root damage when weeding or cultivating",
                    "Rotate tomatoes with non-susceptible crops like corn or beans"
                ],
            },
            "Mosaic_Virus": {
                "emoji": "🌐",
                "color": "#009688",
                "description": "Highly infectious virus causing green/yellow mosaic leaf patterns and leaf distortion.",
                "tip": "Control aphid vectors, remove infected weeds, and plant virus-resistant varieties.",
                "organic": [
                    "Eradicate infected plants immediately to stop systemic transmission",
                    "Use insecticidal soaps or neem oil to control aphid and whitefly vectors"
                ],
                "chemical": [
                    "No chemical viricides exist; manage vectors using systemic insecticides if necessary"
                ],
                "prevention": [
                    "Wash hands and tools with soap or milk before handling healthy plants",
                    "Sow certified virus-free seed varieties",
                    "Control weed hosts around the garden perimeter"
                ],
            },
            "Septoria_Leaf_Spot": {
                "emoji": "🟠",
                "color": "#FF5722",
                "description": "Fungal spots with dark margins and grey centers, causing leaves to yellow and drop.",
                "tip": "Keep foliage dry, apply organic mulch, and use preventive bio-fungicides.",
                "organic": [
                    "Prune infected leaves from lower stem and discard",
                    "Apply straw mulch to block soil-borne spores from splashing upwards",
                    "Apply copper octanoate or organic sulfur sprays"
                ],
                "chemical": [
                    "Apply chlorothalonil or mancozeb at 7-10 day intervals during wet, warm periods"
                ],
                "prevention": [
                    "Practice 3-year crop rotation schedules",
                    "Clean up all garden debris thoroughly in autumn"
                ],
            },
        },
        "disease_cpd": [
            [0.60, 0.50, 0.20, 0.70, 0.75, 0.30, 0.50, 0.40, 0.15],
            [0.05, 0.05, 0.10, 0.05, 0.05, 0.15, 0.10, 0.15, 0.25],
            [0.10, 0.20, 0.45, 0.05, 0.05, 0.20, 0.02, 0.02, 0.05],
            [0.02, 0.02, 0.02, 0.05, 0.05, 0.10, 0.10, 0.15, 0.30],
            [0.18, 0.18, 0.18, 0.10, 0.05, 0.10, 0.20, 0.20, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.15, 0.08, 0.08, 0.15],
        ],
        "symptoms": {
            "Healthy": {
                "Wilting": [0.95, 0.04, 0.01], "WhiteSpots": [0.98, 0.01, 0.01], "YellowLeaves": [0.95, 0.04, 0.01],
                "StuntedGrowth": [0.97, 0.02, 0.01], "BlackSpots": [0.98, 0.01, 0.01], "LeafCurl": [0.96, 0.03, 0.01],
                "FoulSmell": [0.99, 0.01, 0.00]
            },
            "Early_Blight": {
                "BlackSpots": [0.05, 0.25, 0.70], "YellowLeaves": [0.15, 0.45, 0.40], "StuntedGrowth": [0.70, 0.20, 0.10],
                "Wilting": [0.70, 0.20, 0.10]
            },
            "Late_Blight": {
                "BlackSpots": [0.10, 0.30, 0.60], "Wilting": [0.40, 0.40, 0.20], "WhiteSpots": [0.10, 0.40, 0.50],
                "FoulSmell": [0.40, 0.40, 0.20], "YellowLeaves": [0.20, 0.50, 0.30]
            },
            "Bacterial_Wilt": {
                "Wilting": [0.05, 0.15, 0.80], "FoulSmell": [0.20, 0.30, 0.50], "StuntedGrowth": [0.30, 0.40, 0.30]
            },
            "Mosaic_Virus": {
                "LeafCurl": [0.05, 0.25, 0.70], "StuntedGrowth": [0.05, 0.25, 0.70], "YellowLeaves": [0.10, 0.30, 0.60]
            },
            "Septoria_Leaf_Spot": {
                "WhiteSpots": [0.15, 0.45, 0.40], "YellowLeaves": [0.15, 0.50, 0.35], "BlackSpots": [0.20, 0.50, 0.30]
            }
        }
    },
    "wheat": {
        "name": "Wheat",
        "emoji": "🌾",
        "diseases": [
            "Healthy",
            "Leaf_Rust",
            "Stem_Rust",
            "Powdery_Mildew",
            "Loose_Smut",
            "Root_Rot",
        ],
        "disease_info": {
            "Healthy": {
                "emoji": "🟢",
                "color": "#4CAF50",
                "description": "Your wheat field shows no signs of infection.",
                "tip": "Continue monitoring environmental moisture levels.",
                "organic": ["Monitor tillers weekly for early rust warning signs"],
                "chemical": [],
                "prevention": ["Avoid excess nitrogen fertilization", "Optimize seeding rates for proper ventilation"],
            },
            "Leaf_Rust": {
                "emoji": "🟠",
                "color": "#FF5722",
                "description": "Fungal spores producing orange-brown pustules on leaves, restricting photosynthesis.",
                "tip": "Apply triazole fungicides and select rust-resistant wheat seed varieties.",
                "organic": [
                    "Sow rust-resistant wheat seed cultivars",
                    "Integrate mixed crop borders to buffer spore spreads"
                ],
                "chemical": [
                    "Apply triazole or strobilurin foliar fungicide when pustules reach upper leaves"
                ],
                "prevention": [
                    "Destroy volunteer wheat plants in summer to break host cycles",
                    "Avoid excessively early planting dates in the autumn"
                ],
            },
            "Stem_Rust": {
                "emoji": "🔴",
                "color": "#F44336",
                "description": "A highly damaging rust causing dark reddish-brown pustules on stems, leading to lodging.",
                "tip": "Inspect fields early, apply systemic fungicide, and eliminate barberry weeds nearby.",
                "organic": [
                    "Eradicate common barberry bushes (alternate hosts) within 1 mile of the field"
                ],
                "chemical": [
                    "Apply systemic tebuconazole or propiconazole at first spot detection"
                ],
                "prevention": [
                    "Choose cultivars containing Sr genes for stem rust resistance"
                ],
            },
            "Powdery_Mildew": {
                "emoji": "🟡",
                "color": "#FFC107",
                "description": "Fungal growth forming white, fluffy, powdery spots on lower leaves.",
                "tip": "Avoid dense sowing, reduce nitrogen fertilizer overuse, and apply fungicide.",
                "organic": [
                    "Optimize sowing density to prevent high humidity in the canopy",
                    "Apply sulfur dust preventatively to susceptible varieties"
                ],
                "chemical": [
                    "Apply quinoxyfen or triazole fungicides at flag leaf emergence stage"
                ],
                "prevention": [
                    "Avoid excessive nitrogen fertilizer which causes lush, soft tissue",
                    "Rotate wheat with broadleaf crops like canola or peas"
                ],
            },
            "Loose_Smut": {
                "emoji": "⬛",
                "color": "#424242",
                "description": "Fungal disease replacing healthy grain heads with powdery olive-black soot spores.",
                "tip": "Use certified disease-free seed, or apply systemic chemical seed treatments.",
                "organic": [
                    "Perform hot water seed treatment (immerse seeds at 49°C for exactly 10 minutes)"
                ],
                "chemical": [
                    "Apply systemic carboxin or tebuconazole seed treatments prior to sowing"
                ],
                "prevention": [
                    "Sow only certified smut-free seed lots"
                ],
            },
            "Root_Rot": {
                "emoji": "🟣",
                "color": "#9C27B0",
                "description": "Decaying roots and crown due to damp conditions, causing stunted growth and yellow tillers.",
                "tip": "Improve field drainage, practice crop rotation, and avoid planting in cold wet soils.",
                "organic": [
                    "Inoculate seeds with Trichoderma harzianum or Bacillus subtilis bio-agents"
                ],
                "chemical": [
                    "Treat seeds with fludioxonil or difenoconazole fungicides prior to sowing"
                ],
                "prevention": [
                    "Improve field drainage and tile systems",
                    "Maintain a balanced 3-year crop rotation",
                    "Avoid sowing when soil is excessively wet and cold"
                ],
            },
        },
        "disease_cpd": [
            [0.65, 0.55, 0.25, 0.70, 0.75, 0.35, 0.55, 0.45, 0.20],
            [0.05, 0.05, 0.10, 0.08, 0.08, 0.20, 0.10, 0.15, 0.25],
            [0.02, 0.02, 0.05, 0.05, 0.05, 0.15, 0.15, 0.20, 0.30],
            [0.15, 0.25, 0.45, 0.05, 0.05, 0.15, 0.02, 0.02, 0.05],
            [0.08, 0.08, 0.10, 0.07, 0.05, 0.10, 0.08, 0.08, 0.10],
            [0.05, 0.05, 0.05, 0.05, 0.02, 0.05, 0.10, 0.10, 0.10],
        ],
        "symptoms": {
            "Healthy": {
                "Wilting": [0.96, 0.03, 0.01], "WhiteSpots": [0.99, 0.01, 0.00], "YellowLeaves": [0.95, 0.04, 0.01],
                "StuntedGrowth": [0.98, 0.01, 0.01], "BlackSpots": [0.99, 0.01, 0.00], "LeafCurl": [0.99, 0.01, 0.00],
                "FoulSmell": [0.99, 0.01, 0.00]
            },
            "Leaf_Rust": {
                "YellowLeaves": [0.10, 0.40, 0.50], "BlackSpots": [0.15, 0.45, 0.40]
            },
            "Stem_Rust": {
                "StuntedGrowth": [0.20, 0.40, 0.40], "Wilting": [0.30, 0.40, 0.30], "BlackSpots": [0.10, 0.30, 0.60]
            },
            "Powdery_Mildew": {
                "WhiteSpots": [0.05, 0.25, 0.70], "YellowLeaves": [0.30, 0.40, 0.30]
            },
            "Loose_Smut": {
                "BlackSpots": [0.05, 0.15, 0.80], "StuntedGrowth": [0.30, 0.40, 0.30]
            },
            "Root_Rot": {
                "Wilting": [0.10, 0.30, 0.60], "YellowLeaves": [0.20, 0.45, 0.35], "FoulSmell": [0.30, 0.40, 0.30]
            }
        }
    },
    "potato": {
        "name": "Potato",
        "emoji": "🥔",
        "diseases": [
            "Healthy",
            "Late_Blight",
            "Early_Blight",
            "Black_Dot",
            "Common_Scab",
            "Black_Scurf",
        ],
        "disease_info": {
            "Healthy": {
                "emoji": "🟢",
                "color": "#4CAF50",
                "description": "Your potato foliage and roots appear healthy.",
                "tip": "Maintain balanced moisture levels and watch out for beetles.",
                "organic": ["Encourage beneficial ladybirds to control aphid populations"],
                "chemical": [],
                "prevention": ["Keep potato hills fully covered with soil to protect developing tubers"],
            },
            "Late_Blight": {
                "emoji": "🔴",
                "color": "#F44336",
                "description": "The notorious blight causing rotting spots, quick leaf decay, and white fuzzy spores in humid weather.",
                "tip": "Remove infected foliage, use certified seed tubers, and spray copper-based fungicide.",
                "organic": [
                    "Destroy infected cull piles away from fields",
                    "Harvest only during dry weather conditions",
                    "Apply copper-based protectant sprays weekly"
                ],
                "chemical": [
                    "Apply mancozeb, chlorothalonil, or fluazinam preventatively during high-humidity forecasts"
                ],
                "prevention": [
                    "Destroy all volunteer potato plants in spring",
                    "Plant certified disease-free seed tubers"
                ],
            },
            "Early_Blight": {
                "emoji": "🟠",
                "color": "#FF5722",
                "description": "Fungus causing dry, dark brown lesions with target-pattern rings on mature leaves.",
                "tip": "Apply mulch, avoid overhead watering, and fertilize adequately.",
                "organic": [
                    "Ensure balanced soil fertility (avoid nitrogen stress)",
                    "Apply organic compost teas to boost foliar health"
                ],
                "chemical": [
                    "Apply protectant fungicides (mancozeb, chlorothalonil) starting mid-season"
                ],
                "prevention": [
                    "Minimize leaf wetness duration by watering early in the morning",
                    "Incorporate or bury crop debris deep into soil post-harvest"
                ],
            },
            "Black_Dot": {
                "emoji": "🟡",
                "color": "#FFC107",
                "description": "Fungal infection yielding yellow wilted foliage and tiny black spots on stems and roots.",
                "tip": "Ensure proper field drainage, destroy old stems, and rotate crops.",
                "organic": [
                    "Harvest crop immediately once tubers mature to prevent skin silvering",
                    "Amend soil structure with leaf compost to improve aeration"
                ],
                "chemical": [
                    "Use seed-applied fungicides as no effective post-emergence options exist"
                ],
                "prevention": [
                    "Practice rigorous 3-to-4 year rotations with non-hosts",
                    "Ensure balanced nitrogen and phosphorus levels"
                ],
            },
            "Common_Scab": {
                "emoji": "🟤",
                "color": "#8D6E63",
                "description": "Bacterial pathogen causing corky, scabby lesions on tuber surfaces (represented by brown foliage spots).",
                "tip": "Lower soil pH slightly, keep soil damp during tuber initiation, and rotate crops.",
                "organic": [
                    "Incorporate green rye or clover manure into soil",
                    "Maintain soil moisture at field capacity during tuber initiation"
                ],
                "chemical": [
                    "Treat seed tubers with protective mancozeb dust prior to planting"
                ],
                "prevention": [
                    "Maintain soil pH below 5.2 (scab bacteria prefer neutral soils)",
                    "Avoid applying fresh animal manure to potato beds",
                    "Choose scab-resistant potato varieties"
                ],
            },
            "Black_Scurf": {
                "emoji": "🟣",
                "color": "#9C27B0",
                "description": "Rhizoctonia infection causing dark brown scale on tubers, stunting, and upward leaf curling.",
                "tip": "Plant seed in warm soils, avoid deep planting, and harvest early.",
                "organic": [
                    "Harvest tubers as soon as skin sets",
                    "Expose seed tubers to light (chitting) to speed up shoot emergence"
                ],
                "chemical": [
                    "Apply fludioxonil or azoxystrobin seed treatments at planting"
                ],
                "prevention": [
                    "Avoid deep planting in cold, wet spring soils",
                    "Rotate potato crops with barley or oats"
                ],
            },
        },
        "disease_cpd": [
            [0.60, 0.50, 0.20, 0.70, 0.75, 0.30, 0.50, 0.40, 0.15],
            [0.15, 0.25, 0.50, 0.05, 0.05, 0.25, 0.02, 0.02, 0.05],
            [0.05, 0.05, 0.10, 0.05, 0.05, 0.15, 0.10, 0.15, 0.25],
            [0.05, 0.05, 0.05, 0.08, 0.05, 0.10, 0.15, 0.15, 0.20],
            [0.05, 0.05, 0.05, 0.08, 0.05, 0.05, 0.18, 0.18, 0.15],
            [0.10, 0.10, 0.10, 0.04, 0.05, 0.15, 0.05, 0.10, 0.20],
        ],
        "symptoms": {
            "Healthy": {
                "Wilting": [0.95, 0.04, 0.01], "WhiteSpots": [0.98, 0.01, 0.01], "YellowLeaves": [0.95, 0.04, 0.01],
                "StuntedGrowth": [0.97, 0.02, 0.01], "BlackSpots": [0.98, 0.01, 0.01], "LeafCurl": [0.96, 0.03, 0.01],
                "FoulSmell": [0.99, 0.01, 0.00]
            },
            "Late_Blight": {
                "BlackSpots": [0.10, 0.30, 0.60], "Wilting": [0.40, 0.40, 0.20], "WhiteSpots": [0.15, 0.45, 0.40],
                "FoulSmell": [0.35, 0.45, 0.20]
            },
            "Early_Blight": {
                "BlackSpots": [0.05, 0.25, 0.70], "YellowLeaves": [0.15, 0.45, 0.40]
            },
            "Black_Dot": {
                "YellowLeaves": [0.20, 0.40, 0.40], "Wilting": [0.25, 0.45, 0.30], "StuntedGrowth": [0.30, 0.40, 0.30],
                "BlackSpots": [0.40, 0.40, 0.20]
            },
            "Common_Scab": {
                "BlackSpots": [0.10, 0.30, 0.60]
            },
            "Black_Scurf": {
                "StuntedGrowth": [0.20, 0.40, 0.40], "LeafCurl": [0.15, 0.45, 0.40], "BlackSpots": [0.30, 0.45, 0.25]
            }
        }
    },
    "corn": {
        "name": "Corn",
        "emoji": "🌽",
        "diseases": [
            "Healthy",
            "Common_Rust",
            "Gray_Leaf_Spot",
            "Northern_Leaf_Blight",
            "Maize_Dwarf_Mosaic",
            "Stalk_Rot",
        ],
        "disease_info": {
            "Healthy": {
                "emoji": "🟢",
                "color": "#4CAF50",
                "description": "Your maize plants appear strong, tall, and healthy.",
                "tip": "Maintain regular watering and nitrogen schedules.",
                "organic": ["Inoculate soil with beneficial mycorrhizal fungi at seeding"],
                "chemical": [],
                "prevention": ["Monitor soil potassium levels to keep stalk rind thickness healthy"],
            },
            "Common_Rust": {
                "emoji": "🟠",
                "color": "#FF5722",
                "description": "Fungal rust creating small, powdery, cinnamon-brown pustules on leaves.",
                "tip": "Plant resistant hybrids, spray foliar fungicides if severe, and destroy crop residues.",
                "organic": [
                    "Bury corn stalks deep into the soil after harvest to decompose spores"
                ],
                "chemical": [
                    "Apply strobilurin or triazole foliar fungicides if rust appears prior to tasseling"
                ],
                "prevention": [
                    "Plant corn hybrids with high rust resistance scores"
                ],
            },
            "Gray_Leaf_Spot": {
                "emoji": "🟡",
                "color": "#FFC107",
                "description": "Fungus producing long, rectangular, greyish-tan leaf lesions between veins.",
                "tip": "Use tillage to bury old residues, rotate to non-host crops, and apply fungicide.",
                "organic": [
                    "Use crop rotation intervals to allow soil biology to decay leaf debris"
                ],
                "chemical": [
                    "Apply foliar strobilurin or triazole fungicides at first sign of rectangular spots"
                ],
                "prevention": [
                    "Rotate corn with soybeans, alfalfa, or wheat",
                    "Shred old crop residues in autumn to accelerate composting"
                ],
            },
            "Northern_Leaf_Blight": {
                "emoji": "🟤",
                "color": "#A1887F",
                "description": "Fungal pathogen yielding large, grey-green cigar-shaped necrotic lesions.",
                "tip": "Select resistant hybrids, manage crop residue, and rotate crops.",
                "organic": [
                    "Plant corn hybrids containing specific Ht resistance genes"
                ],
                "chemical": [
                    "Apply propiconazole or prothioconazole when leaves show cigar lesions during wet forecasts"
                ],
                "prevention": [
                    "Rotate out of corn for at least one full season",
                    "Practice conservation tillage to bury crop trash"
                ],
            },
            "Maize_Dwarf_Mosaic": {
                "emoji": "🌐",
                "color": "#009688",
                "description": "Viral pathogen causing light-green mottled streaks, leaf curling, and severe stunting.",
                "tip": "Control aphid vectors, eradicate wild Johnsongrass hosts, and sow resistant varieties.",
                "organic": [
                    "Eradicate Johnsongrass weeds (the main overwintering virus reservoir)",
                    "Use reflective mulches in small plots to deter flying aphids"
                ],
                "chemical": [
                    "Apply systemic insecticidal sprays to suppress aphid vector swarms"
                ],
                "prevention": [
                    "Select virus-resistant corn hybrids",
                    "Sow early in spring before aphid vectors peak in flight"
                ],
            },
            "Stalk_Rot": {
                "emoji": "🔴",
                "color": "#F44336",
                "description": "Severe decay of internal stalk tissue causing wilting, foul smell, and lodging.",
                "tip": "Avoid plant overcrowding, avoid overwatering, and optimize potassium nutrients.",
                "organic": [
                    "Balance soil potassium levels to strengthen stalk structures",
                    "Avoid mechanical stalk injuries during cultivation"
                ],
                "chemical": [
                    "No chemical cures exist for stalk decay; apply foliar fungicides early to keep leaves clean"
                ],
                "prevention": [
                    "Select corn hybrids with excellent stay-green and stalk-strength ratings",
                    "Ensure proper seed population densities to prevent plant crowding"
                ],
            },
        },
        "disease_cpd": [
            [0.60, 0.50, 0.20, 0.70, 0.75, 0.30, 0.50, 0.40, 0.15],
            [0.15, 0.20, 0.40, 0.05, 0.05, 0.15, 0.02, 0.02, 0.05],
            [0.05, 0.05, 0.10, 0.05, 0.05, 0.20, 0.10, 0.15, 0.25],
            [0.10, 0.15, 0.20, 0.05, 0.05, 0.15, 0.05, 0.05, 0.10],
            [0.08, 0.08, 0.08, 0.05, 0.05, 0.10, 0.18, 0.18, 0.15],
            [0.02, 0.02, 0.02, 0.10, 0.05, 0.10, 0.15, 0.20, 0.30],
        ],
        "symptoms": {
            "Healthy": {
                "Wilting": [0.96, 0.03, 0.01], "WhiteSpots": [0.99, 0.01, 0.00], "YellowLeaves": [0.95, 0.04, 0.01],
                "StuntedGrowth": [0.98, 0.01, 0.01], "BlackSpots": [0.99, 0.01, 0.00], "LeafCurl": [0.99, 0.01, 0.00],
                "FoulSmell": [0.99, 0.01, 0.00]
            },
            "Common_Rust": {
                "YellowLeaves": [0.20, 0.40, 0.40], "BlackSpots": [0.10, 0.30, 0.60]
            },
            "Gray_Leaf_Spot": {
                "YellowLeaves": [0.15, 0.45, 0.40], "BlackSpots": [0.05, 0.25, 0.70]
            },
            "Northern_Leaf_Blight": {
                "YellowLeaves": [0.20, 0.40, 0.40], "BlackSpots": [0.10, 0.40, 0.50]
            },
            "Maize_Dwarf_Mosaic": {
                "StuntedGrowth": [0.05, 0.25, 0.70], "YellowLeaves": [0.10, 0.40, 0.50], "LeafCurl": [0.10, 0.40, 0.50]
            },
            "Stalk_Rot": {
                "Wilting": [0.10, 0.35, 0.55], "FoulSmell": [0.15, 0.35, 0.50], "StuntedGrowth": [0.30, 0.40, 0.30]
            }
        }
    }
}


# ==================================================================
# Bayesian Network Engine & Model Builder
# ==================================================================

def get_symptom_matrix(profiles: dict, diseases: list[str], symptom: str) -> list[list[float]]:
    """Build a (3, len(diseases)) conditional probability matrix for a symptom."""
    matrix = []
    for state_idx in range(3):
        row = []
        for disease in diseases:
            profile = profiles.get(disease, {})
            if symptom in profile:
                row.append(profile[symptom][state_idx])
            else:
                if disease == "Healthy":
                    default_vals = [0.95, 0.04, 0.01]
                else:
                    default_vals = [0.85, 0.12, 0.03]
                row.append(default_vals[state_idx])
        matrix.append(row)
    return matrix


def build_crop_network(crop_key: str, data: dict) -> DiscreteBayesianNetwork:
    """Build a pgmpy DiscreteBayesianNetwork for the selected crop."""
    diseases = data["diseases"]
    disease_cpd_vals = data["disease_cpd"]
    profiles = data["symptoms"]

    edges = []
    for env in ENVIRON_NODES:
        edges.append((env, "CropDisease"))
    for symptom in SYMPTOM_NODES:
        edges.append(("CropDisease", symptom))

    model = DiscreteBayesianNetwork(edges)

    cpd_temperature = TabularCPD(
        variable="Temperature", variable_card=3,
        values=[[0.25], [0.50], [0.25]],
        state_names={"Temperature": ENVIRON_STATES},
    )
    cpd_humidity = TabularCPD(
        variable="Humidity", variable_card=3,
        values=[[0.25], [0.50], [0.25]],
        state_names={"Humidity": ENVIRON_STATES},
    )

    cpd_disease = TabularCPD(
        variable="CropDisease", variable_card=len(diseases),
        values=disease_cpd_vals,
        evidence=["Temperature", "Humidity"], evidence_card=[3, 3],
        state_names={
            "CropDisease": diseases,
            "Temperature": ENVIRON_STATES,
            "Humidity": ENVIRON_STATES,
        },
    )

    cpds = [cpd_temperature, cpd_humidity, cpd_disease]

    for symptom in SYMPTOM_NODES:
        matrix = get_symptom_matrix(profiles, diseases, symptom)
        cpd_symptom = TabularCPD(
            variable=symptom, variable_card=3,
            values=matrix,
            evidence=["CropDisease"], evidence_card=[len(diseases)],
            state_names={symptom: SYMPTOM_STATES, "CropDisease": diseases},
        )
        cpds.append(cpd_symptom)

    model.add_cpds(*cpds)
    assert model.check_model(), f"Model check failed for crop {crop_key}"
    return model


# Pre-build models and variable elimination inferences for all crops
MODELS = {}
INFERENCES = {}

print("🌿 Pre-building Bayesian Networks for all crops...")
for crop_k, crop_d in CROP_DEFS.items():
    net = build_crop_network(crop_k, crop_d)
    MODELS[crop_k] = net
    INFERENCES[crop_k] = VariableElimination(net)
print("✅ Crop models loaded successfully.")


def predict_disease(crop: str, symptoms: dict, environment: dict) -> dict:
    """Run Variable Elimination inference on the specific crop network."""
    if crop not in INFERENCES:
        raise ValueError(f"Unknown crop: {crop}")

    inference = INFERENCES[crop]
    diseases = CROP_DEFS[crop]["diseases"]
    evidence = {}

    if symptoms:
        for k, v in symptoms.items():
            if v is not None and v != "None" and k in SYMPTOM_NODES:
                evidence[k] = v

    if environment:
        for k, v in environment.items():
            if v is not None and k in ENVIRON_NODES:
                evidence[k] = v

    if not evidence:
        query_result = inference.query(variables=["CropDisease"])
        probs = query_result.values
        return {
            state: round(float(p), 4)
            for state, p in zip(diseases, probs)
        }

    query_result = inference.query(
        variables=["CropDisease"],
        evidence=evidence,
    )
    probabilities = query_result.values
    return {
        state: round(float(prob), 4)
        for state, prob in zip(diseases, probabilities)
    }


# ==================================================================
# FastAPI Web Server Setup
# ==================================================================

app = FastAPI(
    title="Agriculture Disease Finder API",
    description="AI-powered crop-specific disease diagnosis using Bayesian Networks",
    version="3.0.0",
)

# Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    crop: str = "tomato"
    symptoms: dict = {}
    environment: dict = {}


class PredictionResult(BaseModel):
    disease: str
    probability: float
    emoji: str
    color: str
    description: str
    tip: str


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    top_disease: str
    top_probability: float


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "Discrete Bayesian Network (pgmpy)",
        "crops": list(CROP_DEFS.keys()),
        "symptoms_count": len(SYMPTOM_NODES),
    }


@app.get("/model-info")
def model_info():
    """Returns metadata for all crops, including diseases, emojis, descriptions, tips, and symptoms."""
    crops_metadata = {}
    for crop_k, crop_d in CROP_DEFS.items():
        crops_metadata[crop_k] = {
            "name": crop_d["name"],
            "emoji": crop_d["emoji"],
            "diseases": crop_d["diseases"],
            "disease_info": crop_d["disease_info"],
        }

    return {
        "crops": list(CROP_DEFS.keys()),
        "crop_details": crops_metadata,
        "symptoms": SYMPTOM_NODES,
        "symptom_states": SYMPTOM_STATES,
        "environment_nodes": ENVIRON_NODES,
        "environment_states": ENVIRON_STATES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    crop_k = request.crop.lower()
    if crop_k not in CROP_DEFS:
        raise HTTPException(status_code=400, detail=f"Crop '{request.crop}' is not supported. Choose from: {list(CROP_DEFS.keys())}")

    try:
        results = predict_disease(crop_k, request.symptoms, request.environment)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine error: {str(e)}")

    # Sort results by probability (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    top_disease = sorted_results[0][0]
    top_probability = sorted_results[0][1]

    disease_info_map = CROP_DEFS[crop_k]["disease_info"]

    predictions = []
    for disease, prob in sorted_results:
        info = disease_info_map.get(disease, {})
        predictions.append(PredictionResult(
            disease=disease,
            probability=prob,
            emoji=info.get("emoji", "❓"),
            color=info.get("color", "#888"),
            description=info.get("description", ""),
            tip=info.get("tip", ""),
        ))

    return PredictResponse(
        predictions=predictions,
        top_disease=top_disease,
        top_probability=top_probability,
    )


# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("🌿 Starting Crop-Specific Agriculture Disease Finder API...")
    print("📡 API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
