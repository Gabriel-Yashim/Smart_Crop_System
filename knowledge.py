"""Written content used by the pages: crop growing notes, soil categories and condition guides.

The condition guides are general information to help a reader understand a result. They are not
a prescription. Chemical choices, rates and timing should follow local extension advice.
"""

CROP_TIPS = {
    "rice":       "Requires flooded or waterlogged soils. Best grown in humid tropical climates with heavy rainfall.",
    "maize":      "Thrives in well-drained loamy soil. Needs full sunlight and moderate rainfall (500 to 800 mm).",
    "banana":     "Grows best in deep, well-drained loam soils rich in organic matter. Needs high humidity.",
    "mango":      "Prefers deep, well-drained alluvial or loamy soils. Drought-tolerant once established.",
    "grapes":     "Best in deep, well-drained loamy to sandy loam soils. Sensitive to waterlogging.",
    "watermelon": "Thrives in sandy loam soils with good drainage. Requires warm temperatures and low humidity.",
    "coconut":    "Grows best in sandy loam soils near the coast. High tolerance for saline conditions.",
    "coffee":     "Requires well-drained, fertile volcanic soils. Grows best in highlands with consistent rainfall.",
    "cotton":     "Prefers deep, well-drained black or alluvial soils. Sensitive to waterlogging.",
    "potato":     "Requires loose, well-drained sandy loam soil. Cool climates with moderate rainfall.",
    "cashew":     "Grows on sandy coastal soils. Drought-resistant and suitable for marginal lands.",
    "palm tree":  "Thrives in deep, well-drained loamy soils in humid tropical regions with high rainfall.",
}

# name, typical low, typical high (from the original app), axis low, axis high, unit
INPUT_SPECS = [
    ("N", "Nitrogen (N)", 0, 179, "mg/kg"),
    ("P", "Phosphorus (P)", 5, 145, "mg/kg"),
    ("K", "Potassium (K)", 5, 205, "mg/kg"),
    ("temperature", "Temperature", 8, 44, "°C"),
    ("humidity", "Humidity", 0, 100, "%"),
    ("ph", "Soil pH", 3.5, 9.9, ""),
    ("rainfall", "Rainfall", 20, 298, "mm"),
]


def nutrient_level(value, lo, hi):
    pos = (value - lo) / (hi - lo)
    return "Low" if pos < 1 / 3 else "Medium" if pos < 2 / 3 else "High"


def ph_category(ph):
    if ph < 5.5:
        return "Strongly acidic"
    if ph < 6.5:
        return "Slightly acidic"
    if ph <= 7.5:
        return "Neutral"
    if ph <= 8.5:
        return "Alkaline"
    return "Strongly alkaline"


def temperature_category(t):
    return "Cool" if t < 15 else "Mild" if t < 25 else "Warm" if t < 32 else "Hot"


def humidity_category(h):
    return "Dry air" if h < 40 else "Moderate" if h < 70 else "Humid"


def rainfall_category(r):
    return "Low" if r < 75 else "Moderate" if r < 150 else "High" if r < 220 else "Very high"


_HEALTHY = {
    "about": "No visible disease signs were detected on this leaf.",
    "symptoms": "The leaf looks evenly colored, without spots, lesions, mold, curling or unusual patterns.",
    "management": "Keep scouting regularly, water at the base of the plant, remove plant debris, and act early if "
                  "spots or curling appear. A single healthy leaf does not guarantee the whole plant is healthy.",
}

DISEASE_INFO = {
    "Bell_pepper__Bell_pepper_Bacterial_spot": {
        "about": "A bacterial disease caused by Xanthomonas species. It spreads through splashing water, contaminated "
                 "seed and tools, and does best in warm, wet weather.",
        "symptoms": "Small water-soaked spots on leaves that turn dark brown with a rough, slightly raised surface. "
                    "Heavily spotted leaves yellow and drop. Fruit can show raised, scabby spots.",
        "management": "Use clean seed and healthy transplants, avoid overhead irrigation and working in wet fields, remove "
                      "badly infected plants, rotate away from peppers and tomatoes, and use copper-based sprays only as "
                      "advised locally.",
    },
    "Brinjal__Diseased_Brinjal_Leaf_-_Cercospora_Leaf_Spot": {
        "about": "A fungal leaf spot caused by Cercospora species, favoured by warm, humid weather and long periods of "
                 "leaf wetness.",
        "symptoms": "Round to angular brown or gray spots, often with a yellow halo, mostly on older leaves. Severe "
                    "infection causes yellowing and early leaf fall.",
        "management": "Remove infected leaves, avoid wetting the foliage, give plants room for airflow, rotate crops, and "
                      "use a fungicide recommended by your local extension service if the spread continues.",
    },
    "Corn__Corn_Common_rust": {
        "about": "A fungal disease caused by Puccinia sorghi. Spores travel on the wind and the disease is favoured by "
                 "cool, humid weather.",
        "symptoms": "Small, powdery, reddish-brown pustules on both leaf surfaces that later turn dark and break "
                    "through the leaf.",
        "management": "Plant resistant hybrids, plant on time, and scout early in the season. Consider a fungicide only if "
                      "infection is heavy early on and local advice supports it.",
    },
    "Corn__Corn_Gray_leaf_spot": {
        "about": "A fungal disease caused by Cercospora zeae-maydis. It survives in crop residue and is favoured by warm, "
                 "humid weather.",
        "symptoms": "Narrow, rectangular tan to gray lesions that run parallel to the leaf veins and can join together, "
                    "killing large areas of leaf.",
        "management": "Rotate away from corn, manage or bury residue, choose tolerant hybrids, and improve airflow where "
                      "possible. Fungicide timing should follow local advice.",
    },
    "Corn__Corn_Northern_Leaf_Blight": {
        "about": "A fungal disease caused by Exserohilum turcicum, favoured by moderate temperatures and long periods of "
                 "leaf wetness.",
        "symptoms": "Long, cigar-shaped gray-green to tan lesions, often several centimetres long, that start on lower "
                    "leaves and move upward.",
        "management": "Use resistant hybrids, rotate crops, manage residue, and scout early. Apply fungicide only when "
                      "advised locally.",
    },
    "Lettuce__Bacterial": {
        "about": "A group of bacterial diseases of lettuce. They spread through splashing water and do best in warm, wet "
                 "conditions.",
        "symptoms": "Water-soaked spots that turn brown or black, sometimes angular, and may merge. In wet weather the "
                    "affected tissue can become slimy.",
        "management": "Avoid overhead watering, use clean seed, remove infected plants, keep tools clean, rotate crops, "
                      "and harvest in dry conditions.",
    },
    "Lettuce__fungal": {
        "about": "A group of fungal diseases of lettuce, such as downy mildew and leaf spots. They are favoured by cool, "
                 "humid weather and wet leaves.",
        "symptoms": "Yellow or pale patches on the upper leaf, sometimes with white or gray growth underneath, or brown "
                    "spots that expand. Telling the exact disease apart needs a closer look.",
        "management": "Space plants for airflow, water at the base in the morning, remove infected leaves, rotate crops, "
                      "and choose resistant varieties. Ask an extension officer to confirm the specific disease.",
    },
    "Potato__Potato_Early_blight": {
        "about": "A fungal disease caused by Alternaria solani. It is worse when plants are stressed or poorly fed and "
                 "when weather alternates between wet and dry.",
        "symptoms": "Dark brown spots with concentric rings, like a target, on older leaves, with yellowing around them. "
                    "Spots can also appear on stems and tubers.",
        "management": "Rotate crops, keep plants well fed and watered, remove infected leaves, avoid overhead irrigation, "
                      "and use fungicides as advised locally.",
    },
    "Potato__Potato_Late_blight": {
        "about": "A serious disease caused by Phytophthora infestans. It spreads very fast in cool, wet weather and can "
                 "destroy a crop within days.",
        "symptoms": "Large, dark, water-soaked patches on leaves and stems, often with white fuzzy growth underneath in "
                    "humid weather. Tubers can rot.",
        "management": "Act quickly: remove and destroy infected plants, use certified seed, keep foliage dry where you "
                      "can, hill soil over tubers, and follow local fungicide advice. Do not compost infected material.",
    },
    "Tomato__Tomato_Bacterial_spot": {
        "about": "A bacterial disease caused by Xanthomonas species. It spreads through splashing water, seed and tools, "
                 "and prefers warm, wet weather.",
        "symptoms": "Small dark, water-soaked spots on leaves, often with a yellow halo, that become scabby on fruit.",
        "management": "Use clean seed or transplants, avoid overhead watering, keep tools clean, rotate crops, remove "
                      "infected plants, and use copper-based sprays only as advised locally.",
    },
    "Tomato__Tomato_Early_blight": {
        "about": "A fungal disease caused by Alternaria solani. It is worse when plants are stressed or poorly fed and "
                 "when weather alternates between wet and dry.",
        "symptoms": "Dark brown spots with concentric rings, like a target, on older leaves, with yellowing around them. "
                    "Stems and fruit can also be affected.",
        "management": "Rotate crops, stake and mulch plants, remove infected lower leaves, avoid overhead irrigation, and "
                      "use fungicides as advised locally.",
    },
    "Tomato__Tomato_Late_blight": {
        "about": "A serious disease caused by Phytophthora infestans. It spreads very fast in cool, wet weather and can "
                 "destroy plants within days.",
        "symptoms": "Large, dark, greasy-looking patches on leaves and stems, white fuzzy growth underneath in humid "
                    "weather, and firm brown rot on fruit.",
        "management": "Act quickly: remove and destroy infected plants, avoid wet foliage, improve airflow, use "
                      "resistant varieties where available, and follow local fungicide advice.",
    },
    "Tomato__Tomato_Leaf_Mold": {
        "about": "A fungal disease caused by Passalora fulva, common in humid greenhouses and dense plantings.",
        "symptoms": "Pale green to yellow spots on the upper leaf surface with olive-green to gray, velvety mold "
                    "underneath. Leaves later curl and dry.",
        "management": "Lower humidity, improve ventilation and spacing, water at the base, remove infected leaves, and "
                      "choose resistant varieties.",
    },
    "Tomato__Tomato_Mosaic_virus": {
        "about": "A virus that spreads through contaminated hands, tools, seed and plant debris. There is no cure once a "
                 "plant is infected.",
        "symptoms": "Mottled light and dark green patterns on leaves, distorted or narrow leaves, and stunted growth. "
                    "Fruit may ripen unevenly.",
        "management": "Remove and destroy infected plants, wash hands and disinfect tools, clear plant debris, and use "
                      "certified seed and resistant varieties.",
    },
    "Tomato__Tomato_Septoria_leaf_spot": {
        "about": "A fungal disease caused by Septoria lycopersici. It spreads through splashing water and is favoured by "
                 "warm, wet weather.",
        "symptoms": "Many small round spots with dark borders and gray centres, usually starting on lower leaves. "
                    "Badly affected leaves yellow and drop.",
        "management": "Remove infected lower leaves, mulch to stop soil splash, water at the base, rotate crops, and use "
                      "fungicides as advised locally.",
    },
    "Tomato__Tomato_Spider_mites": {
        "about": "Tiny pests (not a disease) that suck plant sap. They multiply fastest in hot, dry, dusty conditions.",
        "symptoms": "Fine yellow speckling on leaves, bronzing or drying, and fine webbing on the underside of leaves in "
                    "heavy infestations.",
        "management": "Wash plants with a strong jet of water, keep plants well watered, use insecticidal soap or a "
                      "suitable miticide as advised locally, and protect natural predators.",
    },
    "Tomato__Tomato_Target_Spot": {
        "about": "A fungal disease caused by Corynespora cassiicola, favoured by warm, humid weather and long periods of "
                 "leaf wetness.",
        "symptoms": "Brown spots with concentric rings and a yellow edge on leaves, which can look like early blight. "
                    "Fruit may show sunken spots.",
        "management": "Improve airflow, avoid overhead watering, remove infected leaves, rotate crops, and use "
                      "fungicides as advised locally.",
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "about": "A virus spread by whiteflies. It cannot be cured once a plant is infected.",
        "symptoms": "Leaves curl upward and turn yellow at the edges, plants are stunted, and flowering and fruit set "
                    "drop sharply.",
        "management": "Control whiteflies (nets, yellow sticky traps, reflective mulch), use resistant varieties, remove "
                      "infected plants early, and keep weeds down.",
    },
}


def condition_info(raw_class_name, healthy):
    """Return the guide for a class name, with a generic healthy entry for healthy classes."""
    if healthy:
        return _HEALTHY
    return DISEASE_INFO.get(raw_class_name)
