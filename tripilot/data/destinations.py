"""
Sample destination data for TripPilot
"""

DESTINATIONS = {
    "paris": {
        "name": "Paris",
        "country": "France",
        "description": "The City of Light - renowned for art, fashion, gastronomy, and culture. Home to iconic landmarks like the Eiffel Tower and the Louvre.",
        "highlights": [
            "Eiffel Tower",
            "Louvre Museum",
            "Notre-Dame Cathedral",
            "Champs-Élysées",
            "Montmartre",
            "Palace of Versailles"
        ],
        "best_time": "April-June, September-November",
        "avg_temperature": {"summer": "25°C", "winter": "5°C"},
        "currency": "EUR",
        "language": "French",
        "timezone": "CET (UTC+1)",
        "rating": 4.8,
        "tags": ["romantic", "cultural", "art", "food", "history"],
        "image_url": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=800"
    },
    "tokyo": {
        "name": "Tokyo",
        "country": "Japan",
        "description": "A dazzling blend of traditional temples and cutting-edge technology. Experience ancient culture alongside futuristic innovation.",
        "highlights": [
            "Senso-ji Temple",
            "Shibuya Crossing",
            "Tokyo Skytree",
            "Tsukiji/Toyosu Fish Market",
            "Meiji Shrine",
            "Akihabara"
        ],
        "best_time": "March-May, September-November",
        "avg_temperature": {"summer": "30°C", "winter": "7°C"},
        "currency": "JPY",
        "language": "Japanese",
        "timezone": "JST (UTC+9)",
        "rating": 4.9,
        "tags": ["technology", "culture", "food", "anime", "temples"],
        "image_url": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=800"
    },
    "barcelona": {
        "name": "Barcelona",
        "country": "Spain",
        "description": "A vibrant Mediterranean city with stunning Gaudí architecture, beautiful beaches, and incredible tapas culture.",
        "highlights": [
            "Sagrada Familia",
            "Park Güell",
            "La Rambla",
            "Gothic Quarter",
            "Casa Batlló",
            "Barceloneta Beach"
        ],
        "best_time": "May-June, September-October",
        "avg_temperature": {"summer": "28°C", "winter": "12°C"},
        "currency": "EUR",
        "language": "Spanish, Catalan",
        "timezone": "CET (UTC+1)",
        "rating": 4.7,
        "tags": ["beach", "architecture", "food", "nightlife", "art"],
        "image_url": "https://images.unsplash.com/photo-1583422409516-2895a77efded?w=800"
    },
    "bali": {
        "name": "Bali",
        "country": "Indonesia",
        "description": "Island paradise known for stunning temples, terraced rice paddies, pristine beaches, and spiritual retreats.",
        "highlights": [
            "Ubud Rice Terraces",
            "Tanah Lot Temple",
            "Seminyak Beach",
            "Mount Batur",
            "Uluwatu Temple",
            "Tegallalang"
        ],
        "best_time": "April-October",
        "avg_temperature": {"summer": "30°C", "winter": "27°C"},
        "currency": "IDR",
        "language": "Indonesian, Balinese",
        "timezone": "WITA (UTC+8)",
        "rating": 4.6,
        "tags": ["beach", "spiritual", "nature", "wellness", "surfing"],
        "image_url": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=800"
    },
    "new york": {
        "name": "New York City",
        "country": "USA",
        "description": "The city that never sleeps - iconic skyline, world-class museums, Broadway shows, and diverse neighborhoods.",
        "highlights": [
            "Statue of Liberty",
            "Central Park",
            "Times Square",
            "Brooklyn Bridge",
            "Metropolitan Museum of Art",
            "Empire State Building"
        ],
        "best_time": "April-June, September-November",
        "avg_temperature": {"summer": "29°C", "winter": "1°C"},
        "currency": "USD",
        "language": "English",
        "timezone": "EST (UTC-5)",
        "rating": 4.7,
        "tags": ["urban", "culture", "food", "shopping", "entertainment"],
        "image_url": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=800"
    },
    "rome": {
        "name": "Rome",
        "country": "Italy",
        "description": "The Eternal City - ancient ruins, Renaissance art, Vatican treasures, and incredible Italian cuisine.",
        "highlights": [
            "Colosseum",
            "Vatican Museums",
            "Trevi Fountain",
            "Pantheon",
            "Roman Forum",
            "Spanish Steps"
        ],
        "best_time": "April-June, September-October",
        "avg_temperature": {"summer": "30°C", "winter": "8°C"},
        "currency": "EUR",
        "language": "Italian",
        "timezone": "CET (UTC+1)",
        "rating": 4.8,
        "tags": ["history", "art", "food", "religion", "architecture"],
        "image_url": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?w=800"
    },
    "london": {
        "name": "London",
        "country": "United Kingdom",
        "description": "Historic royal city with world-class museums, theater, diverse cuisine, and iconic landmarks.",
        "highlights": [
            "Big Ben & Parliament",
            "Tower of London",
            "British Museum",
            "Buckingham Palace",
            "West End Theatre",
            "Tower Bridge"
        ],
        "best_time": "May-September",
        "avg_temperature": {"summer": "23°C", "winter": "6°C"},
        "currency": "GBP",
        "language": "English",
        "timezone": "GMT (UTC+0)",
        "rating": 4.7,
        "tags": ["history", "culture", "theatre", "museums", "royal"],
        "image_url": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?w=800"
    },
    "dubai": {
        "name": "Dubai",
        "country": "United Arab Emirates",
        "description": "Futuristic city of superlatives - tallest buildings, luxury shopping, desert adventures, and ambitious architecture.",
        "highlights": [
            "Burj Khalifa",
            "Dubai Mall",
            "Palm Jumeirah",
            "Dubai Marina",
            "Desert Safari",
            "Gold Souk"
        ],
        "best_time": "November-March",
        "avg_temperature": {"summer": "41°C", "winter": "24°C"},
        "currency": "AED",
        "language": "Arabic, English",
        "timezone": "GST (UTC+4)",
        "rating": 4.5,
        "tags": ["luxury", "shopping", "architecture", "desert", "modern"],
        "image_url": "https://images.unsplash.com/photo-1512453979798-5ea266f8880c?w=800"
    },
    "amsterdam": {
        "name": "Amsterdam",
        "country": "Netherlands",
        "description": "Charming canal city known for historic architecture, world-class museums, cycling culture, and liberal atmosphere.",
        "highlights": [
            "Anne Frank House",
            "Van Gogh Museum",
            "Rijksmuseum",
            "Canal Ring",
            "Vondelpark",
            "Jordaan District"
        ],
        "best_time": "April-May, September-October",
        "avg_temperature": {"summer": "22°C", "winter": "5°C"},
        "currency": "EUR",
        "language": "Dutch",
        "timezone": "CET (UTC+1)",
        "rating": 4.6,
        "tags": ["art", "canals", "cycling", "museums", "liberal"],
        "image_url": "https://images.unsplash.com/photo-1534351590666-13e3e96b5017?w=800"
    },
    "singapore": {
        "name": "Singapore",
        "country": "Singapore",
        "description": "Modern city-state blending futuristic architecture, lush gardens, diverse food scenes, and multicultural heritage.",
        "highlights": [
            "Marina Bay Sands",
            "Gardens by the Bay",
            "Sentosa Island",
            "Orchard Road",
            "Chinatown",
            "Hawker Centers"
        ],
        "best_time": "February-April",
        "avg_temperature": {"summer": "31°C", "winter": "27°C"},
        "currency": "SGD",
        "language": "English, Mandarin, Malay, Tamil",
        "timezone": "SGT (UTC+8)",
        "rating": 4.7,
        "tags": ["modern", "food", "gardens", "clean", "multicultural"],
        "image_url": "https://images.unsplash.com/photo-1525625293386-3f8f99389edd?w=800"
    }
}


def get_destination(name: str) -> dict:
    """
    Get destination data by name.

    Args:
        name: Destination name (case-insensitive)

    Returns:
        Destination data dict or None if not found
    """
    key = name.lower().strip()
    return DESTINATIONS.get(key)


def search_destinations(
    query: str = None,
    tags: list = None,
    min_rating: float = None
) -> list:
    """
    Search destinations by various criteria.

    Args:
        query: Search term for name/description
        tags: List of tags to filter by
        min_rating: Minimum rating threshold

    Returns:
        List of matching destinations
    """
    results = []

    for key, dest in DESTINATIONS.items():
        # Query filter
        if query:
            query_lower = query.lower()
            if query_lower not in dest["name"].lower() and \
               query_lower not in dest["description"].lower() and \
               query_lower not in dest["country"].lower():
                continue

        # Tags filter
        if tags:
            if not any(tag.lower() in dest.get("tags", []) for tag in tags):
                continue

        # Rating filter
        if min_rating:
            if dest.get("rating", 0) < min_rating:
                continue

        results.append(dest)

    return sorted(results, key=lambda x: x.get("rating", 0), reverse=True)
