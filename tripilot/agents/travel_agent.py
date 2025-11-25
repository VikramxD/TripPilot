"""
TripPilot Travel Agent - Core agent with travel planning capabilities
"""
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from enum import Enum
from datetime import datetime, timedelta


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ToolCall:
    """Represents a tool call made by the agent"""
    name: str
    args: dict
    result: Any = None
    duration: float = 0.0
    status: str = "pending"


@dataclass
class AgentStep:
    """Represents a single step in agent execution"""
    step_num: int
    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    observation: str = ""
    state: AgentState = AgentState.THINKING


@dataclass
class AgentResponse:
    """Complete agent response"""
    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    total_time: float = 0.0
    success: bool = True


class TravelAgent:
    """
    AI Travel Planning Agent with tool-use capabilities.

    This agent can:
    - Search for destinations based on preferences
    - Find hotels and accommodations
    - Get restaurant recommendations
    - Create day-by-day itineraries
    - Provide travel tips and insights
    - Check weather forecasts
    """

    TOOLS = {
        "search_destinations": {
            "description": "Search for travel destinations based on criteria",
            "parameters": ["query", "preferences", "budget"]
        },
        "find_hotels": {
            "description": "Find hotels in a specific location",
            "parameters": ["location", "check_in", "check_out", "guests", "budget"]
        },
        "get_restaurants": {
            "description": "Get restaurant recommendations",
            "parameters": ["location", "cuisine", "price_range"]
        },
        "create_itinerary": {
            "description": "Create a day-by-day travel itinerary",
            "parameters": ["destination", "days", "interests", "pace"]
        },
        "get_attractions": {
            "description": "Get top attractions and activities",
            "parameters": ["location", "category", "limit"]
        },
        "check_weather": {
            "description": "Check weather forecast for destination",
            "parameters": ["location", "dates"]
        },
        "get_travel_tips": {
            "description": "Get travel tips and local insights",
            "parameters": ["destination", "topics"]
        },
        "estimate_budget": {
            "description": "Estimate total trip budget",
            "parameters": ["destination", "days", "travel_style"]
        }
    }

    def __init__(self):
        self.current_state = AgentState.IDLE
        self.execution_history: list[AgentResponse] = []

    def _simulate_thinking(self, query: str) -> list[dict]:
        """Simulate agent reasoning to determine actions"""
        # Analyze query to determine which tools to use
        query_lower = query.lower()

        actions = []

        # Destination search
        if any(word in query_lower for word in ["where", "destination", "visit", "travel to", "go to", "trip"]):
            actions.append({
                "thought": "I should search for destinations that match the user's interests and preferences.",
                "tool": "search_destinations",
                "args": self._extract_destination_args(query)
            })

        # Hotel search
        if any(word in query_lower for word in ["hotel", "stay", "accommodation", "book", "lodging"]):
            actions.append({
                "thought": "I need to find suitable hotels for the trip.",
                "tool": "find_hotels",
                "args": self._extract_hotel_args(query)
            })

        # Restaurant search
        if any(word in query_lower for word in ["restaurant", "food", "eat", "dining", "cuisine"]):
            actions.append({
                "thought": "Let me find some great restaurant recommendations.",
                "tool": "get_restaurants",
                "args": self._extract_restaurant_args(query)
            })

        # Itinerary creation
        if any(word in query_lower for word in ["itinerary", "plan", "schedule", "days", "trip plan"]):
            actions.append({
                "thought": "I'll create a detailed day-by-day itinerary for this trip.",
                "tool": "create_itinerary",
                "args": self._extract_itinerary_args(query)
            })

        # Attractions
        if any(word in query_lower for word in ["attraction", "see", "do", "activity", "visit", "sightseeing"]):
            actions.append({
                "thought": "I should find the top attractions and activities in this area.",
                "tool": "get_attractions",
                "args": self._extract_attraction_args(query)
            })

        # Weather
        if any(word in query_lower for word in ["weather", "climate", "temperature", "rain"]):
            actions.append({
                "thought": "Let me check the weather forecast for the travel dates.",
                "tool": "check_weather",
                "args": self._extract_weather_args(query)
            })

        # Travel tips
        if any(word in query_lower for word in ["tip", "advice", "know", "local", "culture"]):
            actions.append({
                "thought": "I'll gather some useful travel tips and local insights.",
                "tool": "get_travel_tips",
                "args": self._extract_tips_args(query)
            })

        # Budget estimation
        if any(word in query_lower for word in ["budget", "cost", "expensive", "cheap", "price", "money"]):
            actions.append({
                "thought": "I should estimate the total budget for this trip.",
                "tool": "estimate_budget",
                "args": self._extract_budget_args(query)
            })

        # Default comprehensive action if no specific match
        if not actions:
            destination = self._extract_destination(query)
            actions = [
                {
                    "thought": f"The user wants to explore {destination}. Let me gather comprehensive travel information.",
                    "tool": "search_destinations",
                    "args": {"query": destination, "preferences": "general", "budget": "moderate"}
                },
                {
                    "thought": "Now I'll find the top attractions to visit.",
                    "tool": "get_attractions",
                    "args": {"location": destination, "category": "all", "limit": 5}
                },
                {
                    "thought": "Let me also get some travel tips for the destination.",
                    "tool": "get_travel_tips",
                    "args": {"destination": destination, "topics": ["culture", "safety", "local customs"]}
                }
            ]

        return actions

    def _extract_destination(self, query: str) -> str:
        """Extract destination from query"""
        destinations = [
            "Paris", "Tokyo", "New York", "Barcelona", "Bali", "Rome",
            "London", "Sydney", "Dubai", "Bangkok", "Amsterdam", "Singapore"
        ]
        for dest in destinations:
            if dest.lower() in query.lower():
                return dest
        # Extract potential location mentions
        words = query.split()
        for i, word in enumerate(words):
            if word.lower() in ["to", "in", "at", "visit"]:
                if i + 1 < len(words):
                    return words[i + 1].strip(".,!?").title()
        return "Paris"  # Default destination

    def _extract_destination_args(self, query: str) -> dict:
        return {
            "query": self._extract_destination(query),
            "preferences": self._extract_preferences(query),
            "budget": self._extract_budget_level(query)
        }

    def _extract_hotel_args(self, query: str) -> dict:
        return {
            "location": self._extract_destination(query),
            "check_in": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
            "check_out": (datetime.now() + timedelta(days=35)).strftime("%Y-%m-%d"),
            "guests": 2,
            "budget": self._extract_budget_level(query)
        }

    def _extract_restaurant_args(self, query: str) -> dict:
        cuisines = ["italian", "japanese", "french", "local", "seafood", "vegetarian"]
        cuisine = "local"
        for c in cuisines:
            if c in query.lower():
                cuisine = c
                break
        return {
            "location": self._extract_destination(query),
            "cuisine": cuisine,
            "price_range": self._extract_budget_level(query)
        }

    def _extract_itinerary_args(self, query: str) -> dict:
        # Extract number of days
        import re
        days_match = re.search(r'(\d+)\s*days?', query.lower())
        days = int(days_match.group(1)) if days_match else 5
        return {
            "destination": self._extract_destination(query),
            "days": days,
            "interests": self._extract_preferences(query),
            "pace": "moderate"
        }

    def _extract_attraction_args(self, query: str) -> dict:
        return {
            "location": self._extract_destination(query),
            "category": "all",
            "limit": 5
        }

    def _extract_weather_args(self, query: str) -> dict:
        return {
            "location": self._extract_destination(query),
            "dates": (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        }

    def _extract_tips_args(self, query: str) -> dict:
        return {
            "destination": self._extract_destination(query),
            "topics": ["culture", "safety", "transportation", "local customs"]
        }

    def _extract_budget_args(self, query: str) -> dict:
        import re
        days_match = re.search(r'(\d+)\s*days?', query.lower())
        days = int(days_match.group(1)) if days_match else 5
        return {
            "destination": self._extract_destination(query),
            "days": days,
            "travel_style": self._extract_budget_level(query)
        }

    def _extract_preferences(self, query: str) -> str:
        prefs = []
        if any(w in query.lower() for w in ["culture", "museum", "history", "art"]):
            prefs.append("cultural")
        if any(w in query.lower() for w in ["food", "cuisine", "restaurant", "eat"]):
            prefs.append("culinary")
        if any(w in query.lower() for w in ["adventure", "outdoor", "hike", "nature"]):
            prefs.append("adventure")
        if any(w in query.lower() for w in ["relax", "beach", "spa", "peaceful"]):
            prefs.append("relaxation")
        if any(w in query.lower() for w in ["family", "kid", "children"]):
            prefs.append("family-friendly")
        return ", ".join(prefs) if prefs else "general sightseeing"

    def _extract_budget_level(self, query: str) -> str:
        if any(w in query.lower() for w in ["luxury", "expensive", "premium", "5-star"]):
            return "luxury"
        if any(w in query.lower() for w in ["budget", "cheap", "affordable", "backpack"]):
            return "budget"
        return "moderate"

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a tool and return results"""
        tool_functions = {
            "search_destinations": self._tool_search_destinations,
            "find_hotels": self._tool_find_hotels,
            "get_restaurants": self._tool_get_restaurants,
            "create_itinerary": self._tool_create_itinerary,
            "get_attractions": self._tool_get_attractions,
            "check_weather": self._tool_check_weather,
            "get_travel_tips": self._tool_get_travel_tips,
            "estimate_budget": self._tool_estimate_budget,
        }

        if tool_name in tool_functions:
            return tool_functions[tool_name](args)
        return {"error": f"Unknown tool: {tool_name}"}

    def _tool_search_destinations(self, args: dict) -> dict:
        """Search for travel destinations"""
        time.sleep(0.3)  # Simulate API call
        query = args.get("query", "")

        destinations_db = {
            "paris": {
                "name": "Paris, France",
                "description": "The City of Light - renowned for art, fashion, gastronomy, and culture",
                "highlights": ["Eiffel Tower", "Louvre Museum", "Notre-Dame", "Champs-Élysées"],
                "best_time": "April-June, September-November",
                "rating": 4.8,
                "image": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?w=400"
            },
            "tokyo": {
                "name": "Tokyo, Japan",
                "description": "A dazzling blend of traditional temples and cutting-edge technology",
                "highlights": ["Senso-ji Temple", "Shibuya Crossing", "Tokyo Skytree", "Tsukiji Market"],
                "best_time": "March-May, September-November",
                "rating": 4.9,
                "image": "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf?w=400"
            },
            "barcelona": {
                "name": "Barcelona, Spain",
                "description": "A vibrant city with stunning architecture, beaches, and tapas culture",
                "highlights": ["Sagrada Familia", "Park Güell", "La Rambla", "Gothic Quarter"],
                "best_time": "May-June, September-October",
                "rating": 4.7,
                "image": "https://images.unsplash.com/photo-1583422409516-2895a77efded?w=400"
            },
            "bali": {
                "name": "Bali, Indonesia",
                "description": "Island paradise known for temples, rice terraces, and beaches",
                "highlights": ["Ubud Rice Terraces", "Tanah Lot Temple", "Seminyak Beach", "Mount Batur"],
                "best_time": "April-October",
                "rating": 4.6,
                "image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?w=400"
            },
            "new york": {
                "name": "New York City, USA",
                "description": "The city that never sleeps - iconic skyline, Broadway, and diverse neighborhoods",
                "highlights": ["Statue of Liberty", "Central Park", "Times Square", "Brooklyn Bridge"],
                "best_time": "April-June, September-November",
                "rating": 4.7,
                "image": "https://images.unsplash.com/photo-1496442226666-8d4d0e62e6e9?w=400"
            }
        }

        key = query.lower().replace(",", "").strip()
        if key in destinations_db:
            return {"destinations": [destinations_db[key]], "total_found": 1}

        # Return multiple suggestions
        return {
            "destinations": list(destinations_db.values())[:3],
            "total_found": len(destinations_db),
            "message": f"Showing top destinations matching your preferences"
        }

    def _tool_find_hotels(self, args: dict) -> dict:
        """Find hotels in a location"""
        time.sleep(0.4)
        location = args.get("location", "Paris")
        budget = args.get("budget", "moderate")

        price_ranges = {"budget": (50, 100), "moderate": (120, 250), "luxury": (300, 800)}
        min_price, max_price = price_ranges.get(budget, (120, 250))

        hotels = [
            {
                "name": f"Grand Hotel {location}",
                "rating": 4.5 + random.random() * 0.4,
                "price_per_night": random.randint(min_price, max_price),
                "amenities": ["Free WiFi", "Pool", "Spa", "Restaurant"],
                "location": "City Center",
                "reviews": random.randint(500, 2000)
            },
            {
                "name": f"Boutique Inn {location}",
                "rating": 4.3 + random.random() * 0.4,
                "price_per_night": random.randint(min_price, max_price),
                "amenities": ["Free WiFi", "Breakfast", "Rooftop Bar"],
                "location": "Old Town",
                "reviews": random.randint(200, 800)
            },
            {
                "name": f"Modern Suites {location}",
                "rating": 4.4 + random.random() * 0.4,
                "price_per_night": random.randint(min_price, max_price),
                "amenities": ["Free WiFi", "Gym", "Kitchen", "Workspace"],
                "location": "Business District",
                "reviews": random.randint(300, 1000)
            }
        ]

        return {"hotels": hotels, "location": location, "budget_category": budget}

    def _tool_get_restaurants(self, args: dict) -> dict:
        """Get restaurant recommendations"""
        time.sleep(0.3)
        location = args.get("location", "Paris")
        cuisine = args.get("cuisine", "local")

        restaurants = [
            {
                "name": f"Le Petit {location}",
                "cuisine": cuisine.title(),
                "rating": 4.6 + random.random() * 0.3,
                "price_level": "$$",
                "specialty": f"Traditional {cuisine} cuisine",
                "must_try": "Chef's tasting menu"
            },
            {
                "name": f"Bistro Central",
                "cuisine": cuisine.title(),
                "rating": 4.4 + random.random() * 0.4,
                "price_level": "$$$",
                "specialty": "Modern fusion",
                "must_try": "Seasonal specials"
            },
            {
                "name": f"Street Food Market",
                "cuisine": "Various",
                "rating": 4.3 + random.random() * 0.4,
                "price_level": "$",
                "specialty": "Local street food",
                "must_try": "Everything!"
            }
        ]

        return {"restaurants": restaurants, "location": location}

    def _tool_create_itinerary(self, args: dict) -> dict:
        """Create a travel itinerary"""
        time.sleep(0.5)
        destination = args.get("destination", "Paris")
        days = args.get("days", 5)
        interests = args.get("interests", "general")

        itinerary = []
        activities_pool = [
            {"name": "Historical Walking Tour", "duration": "3 hours", "type": "cultural"},
            {"name": "Local Food Market Visit", "duration": "2 hours", "type": "culinary"},
            {"name": "Museum Exploration", "duration": "4 hours", "type": "cultural"},
            {"name": "Scenic Viewpoint Visit", "duration": "1.5 hours", "type": "sightseeing"},
            {"name": "Traditional Cooking Class", "duration": "3 hours", "type": "culinary"},
            {"name": "Neighborhood Discovery Walk", "duration": "2.5 hours", "type": "exploration"},
            {"name": "Sunset Cruise/Tour", "duration": "2 hours", "type": "romantic"},
            {"name": "Local Artisan Workshop", "duration": "2 hours", "type": "cultural"},
            {"name": "Day Trip to Nearby Town", "duration": "Full day", "type": "adventure"},
            {"name": "Street Art & Hidden Gems Tour", "duration": "3 hours", "type": "exploration"},
        ]

        for day in range(1, min(days + 1, 8)):
            day_activities = random.sample(activities_pool, min(3, len(activities_pool)))
            itinerary.append({
                "day": day,
                "theme": f"Day {day}: " + random.choice(["Exploration", "Culture", "Adventure", "Relaxation", "Discovery"]),
                "activities": [
                    {"time": "Morning", "activity": day_activities[0]["name"], "duration": day_activities[0]["duration"]},
                    {"time": "Afternoon", "activity": day_activities[1]["name"], "duration": day_activities[1]["duration"]},
                    {"time": "Evening", "activity": day_activities[2]["name"] if len(day_activities) > 2 else "Dinner at local restaurant", "duration": "2-3 hours"},
                ],
                "tips": f"Wear comfortable shoes and bring a camera!"
            })

        return {
            "destination": destination,
            "total_days": days,
            "itinerary": itinerary,
            "note": "This itinerary can be customized based on your preferences and pace."
        }

    def _tool_get_attractions(self, args: dict) -> dict:
        """Get top attractions"""
        time.sleep(0.3)
        location = args.get("location", "Paris")
        limit = args.get("limit", 5)

        attractions = [
            {"name": f"Historic Landmark of {location}", "category": "Landmark", "rating": 4.8, "visit_duration": "2-3 hours"},
            {"name": f"{location} National Museum", "category": "Museum", "rating": 4.7, "visit_duration": "3-4 hours"},
            {"name": f"Old Town Quarter", "category": "Neighborhood", "rating": 4.6, "visit_duration": "2-3 hours"},
            {"name": f"Central Park & Gardens", "category": "Nature", "rating": 4.5, "visit_duration": "1-2 hours"},
            {"name": f"Local Market Square", "category": "Market", "rating": 4.4, "visit_duration": "1-2 hours"},
            {"name": f"Observation Deck", "category": "Viewpoint", "rating": 4.6, "visit_duration": "1 hour"},
            {"name": f"Art District", "category": "Cultural", "rating": 4.3, "visit_duration": "2-3 hours"},
        ]

        return {"attractions": attractions[:limit], "location": location}

    def _tool_check_weather(self, args: dict) -> dict:
        """Check weather forecast"""
        time.sleep(0.2)
        location = args.get("location", "Paris")

        weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Clear"]
        forecast = []

        for i in range(7):
            forecast.append({
                "day": (datetime.now() + timedelta(days=30 + i)).strftime("%A, %b %d"),
                "condition": random.choice(weather_conditions),
                "high": random.randint(18, 28),
                "low": random.randint(10, 18),
                "precipitation": f"{random.randint(0, 40)}%"
            })

        return {
            "location": location,
            "forecast": forecast,
            "recommendation": "Pack layers and bring a light jacket for cooler evenings."
        }

    def _tool_get_travel_tips(self, args: dict) -> dict:
        """Get travel tips"""
        time.sleep(0.3)
        destination = args.get("destination", "Paris")

        tips = {
            "transportation": [
                f"Use public transit - it's efficient and affordable in {destination}",
                "Download offline maps before your trip",
                "Consider getting a tourist transit pass for unlimited travel"
            ],
            "safety": [
                "Keep valuables secure and be aware of pickpockets in tourist areas",
                "Stay in well-lit areas at night",
                "Register with your embassy for travel alerts"
            ],
            "culture": [
                "Learn a few basic phrases in the local language",
                "Respect local customs and dress codes at religious sites",
                "Tipping customs vary - research before you go"
            ],
            "money_saving": [
                "Book attractions online in advance for discounts",
                "Eat where locals eat for authentic and affordable meals",
                "Visit museums on free entry days"
            ],
            "packing": [
                "Pack a universal power adapter",
                "Bring comfortable walking shoes",
                "Keep important documents in multiple locations"
            ]
        }

        return {"destination": destination, "tips": tips}

    def _tool_estimate_budget(self, args: dict) -> dict:
        """Estimate trip budget"""
        time.sleep(0.3)
        destination = args.get("destination", "Paris")
        days = args.get("days", 5)
        style = args.get("travel_style", "moderate")

        daily_costs = {
            "budget": {"accommodation": 60, "food": 30, "activities": 25, "transport": 15},
            "moderate": {"accommodation": 150, "food": 60, "activities": 50, "transport": 25},
            "luxury": {"accommodation": 400, "food": 150, "activities": 100, "transport": 50}
        }

        costs = daily_costs.get(style, daily_costs["moderate"])
        daily_total = sum(costs.values())

        return {
            "destination": destination,
            "days": days,
            "travel_style": style,
            "daily_breakdown": costs,
            "daily_total": daily_total,
            "trip_total": daily_total * days,
            "currency": "USD",
            "note": "Prices are estimates and may vary based on season and availability."
        }

    def _generate_final_response(self, query: str, steps: list[AgentStep]) -> str:
        """Generate a comprehensive final response based on gathered information"""
        # Collect all tool results
        all_results = {}
        for step in steps:
            for tool_call in step.tool_calls:
                if tool_call.result:
                    all_results[tool_call.name] = tool_call.result

        # Build response
        response_parts = []

        # Destination info
        if "search_destinations" in all_results:
            dest_data = all_results["search_destinations"]
            if "destinations" in dest_data and dest_data["destinations"]:
                dest = dest_data["destinations"][0]
                response_parts.append(f"## {dest.get('name', 'Destination')}\n")
                response_parts.append(f"*{dest.get('description', '')}*\n")
                if "highlights" in dest:
                    response_parts.append(f"\n**Highlights:** {', '.join(dest['highlights'])}\n")
                if "best_time" in dest:
                    response_parts.append(f"**Best Time to Visit:** {dest['best_time']}\n")

        # Attractions
        if "get_attractions" in all_results:
            attr_data = all_results["get_attractions"]
            if "attractions" in attr_data:
                response_parts.append("\n### Top Attractions\n")
                for attr in attr_data["attractions"][:5]:
                    response_parts.append(f"- **{attr['name']}** ({attr['category']}) - {attr['visit_duration']}\n")

        # Hotels
        if "find_hotels" in all_results:
            hotel_data = all_results["find_hotels"]
            if "hotels" in hotel_data:
                response_parts.append("\n### Recommended Hotels\n")
                for hotel in hotel_data["hotels"][:3]:
                    response_parts.append(f"- **{hotel['name']}** - ${hotel['price_per_night']}/night ({hotel['location']})\n")

        # Restaurants
        if "get_restaurants" in all_results:
            rest_data = all_results["get_restaurants"]
            if "restaurants" in rest_data:
                response_parts.append("\n### Restaurant Picks\n")
                for rest in rest_data["restaurants"][:3]:
                    response_parts.append(f"- **{rest['name']}** ({rest['cuisine']}) - {rest['price_level']}\n")

        # Itinerary
        if "create_itinerary" in all_results:
            itin_data = all_results["create_itinerary"]
            if "itinerary" in itin_data:
                response_parts.append(f"\n### {itin_data['total_days']}-Day Itinerary\n")
                for day in itin_data["itinerary"][:3]:  # Show first 3 days
                    response_parts.append(f"\n**{day['theme']}**\n")
                    for act in day["activities"]:
                        response_parts.append(f"- {act['time']}: {act['activity']} ({act['duration']})\n")
                if itin_data['total_days'] > 3:
                    response_parts.append(f"\n*...and {itin_data['total_days'] - 3} more days of adventure!*\n")

        # Weather
        if "check_weather" in all_results:
            weather_data = all_results["check_weather"]
            if "forecast" in weather_data:
                response_parts.append("\n### Weather Outlook\n")
                for day in weather_data["forecast"][:3]:
                    response_parts.append(f"- {day['day']}: {day['condition']}, {day['high']}°/{day['low']}°C\n")

        # Budget
        if "estimate_budget" in all_results:
            budget_data = all_results["estimate_budget"]
            response_parts.append(f"\n### Budget Estimate ({budget_data.get('travel_style', 'moderate').title()})\n")
            response_parts.append(f"- **Daily:** ${budget_data.get('daily_total', 0)}\n")
            response_parts.append(f"- **Total ({budget_data.get('days', 5)} days):** ${budget_data.get('trip_total', 0)}\n")

        # Travel Tips
        if "get_travel_tips" in all_results:
            tips_data = all_results["get_travel_tips"]
            if "tips" in tips_data:
                response_parts.append("\n### Quick Tips\n")
                all_tips = []
                for category, tip_list in tips_data["tips"].items():
                    all_tips.extend(tip_list[:1])  # One tip from each category
                for tip in all_tips[:4]:
                    response_parts.append(f"- {tip}\n")

        if response_parts:
            return "\n".join(response_parts)

        return "I've gathered information about your travel query. Feel free to ask more specific questions about destinations, hotels, activities, or itineraries!"

    def run(self, query: str) -> Generator[tuple[AgentState, Any], None, AgentResponse]:
        """
        Run the agent on a query, yielding state updates.

        Yields tuples of (state, data) for UI updates.
        """
        start_time = time.time()
        response = AgentResponse(query=query)

        # Phase 1: Thinking
        self.current_state = AgentState.THINKING
        yield (AgentState.THINKING, {"message": "Analyzing your travel query..."})

        actions = self._simulate_thinking(query)

        # Phase 2: Execute tools
        for i, action in enumerate(actions, 1):
            step = AgentStep(step_num=i, thought=action["thought"])

            # Yield thought
            self.current_state = AgentState.THINKING
            yield (AgentState.THINKING, {"step": i, "thought": action["thought"]})

            # Execute tool
            self.current_state = AgentState.TOOL_CALLING
            tool_call = ToolCall(name=action["tool"], args=action["args"])
            yield (AgentState.TOOL_CALLING, {"tool": action["tool"], "args": action["args"]})

            # Get result
            tool_start = time.time()
            result = self._execute_tool(action["tool"], action["args"])
            tool_call.result = result
            tool_call.duration = time.time() - tool_start
            tool_call.status = "success"

            step.tool_calls.append(tool_call)
            step.observation = json.dumps(result, indent=2)[:500]  # Truncate for display

            yield (AgentState.TOOL_CALLING, {"tool": action["tool"], "result": result, "duration": tool_call.duration})

            response.steps.append(step)

        # Phase 3: Generate final response
        self.current_state = AgentState.GENERATING
        yield (AgentState.GENERATING, {"message": "Composing your travel recommendations..."})

        final_answer = self._generate_final_response(query, response.steps)
        response.final_answer = final_answer
        response.total_time = time.time() - start_time

        # Complete
        self.current_state = AgentState.COMPLETE
        yield (AgentState.COMPLETE, {"answer": final_answer, "time": response.total_time})

        self.execution_history.append(response)
        return response
