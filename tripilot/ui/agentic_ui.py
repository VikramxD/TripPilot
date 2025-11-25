"""
TripPilot Agentic UI - Interactive Travel Planning Interface

This module creates a Gradio-based UI that visualizes the agent's
thinking process, tool calls, and provides an interactive chat experience.
"""

import gradio as gr
import json
import time
from typing import Generator
from ..agents.travel_agent import TravelAgent, AgentState


# Custom CSS for the agentic UI
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
}

/* Agent status indicator */
.agent-status {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-radius: 12px;
    font-weight: 600;
    margin-bottom: 16px;
}

.status-idle {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    color: #3730a3;
}

.status-thinking {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    color: #92400e;
    animation: pulse 2s infinite;
}

.status-tool {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    color: #065f46;
}

.status-complete {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    color: #1e40af;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

/* Thought bubble styling */
.thought-bubble {
    background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%);
    border-left: 4px solid #eab308;
    padding: 16px;
    margin: 12px 0;
    border-radius: 0 12px 12px 0;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* Tool call card */
.tool-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.tool-name {
    font-weight: 700;
    color: #166534;
    font-size: 1.1em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tool-args {
    background: #ffffff;
    padding: 12px;
    border-radius: 8px;
    margin-top: 10px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
    color: #374151;
}

/* Result card */
.result-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #93c5fd;
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
}

/* Step indicator */
.step-indicator {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    color: white;
    border-radius: 50%;
    font-weight: 700;
    font-size: 0.95em;
    margin-right: 12px;
}

/* Timeline styling */
.timeline {
    border-left: 3px solid #c4b5fd;
    padding-left: 24px;
    margin-left: 16px;
}

.timeline-item {
    position: relative;
    padding-bottom: 20px;
}

.timeline-item::before {
    content: '';
    position: absolute;
    left: -30px;
    top: 4px;
    width: 14px;
    height: 14px;
    background: #8b5cf6;
    border-radius: 50%;
    border: 3px solid white;
    box-shadow: 0 0 0 3px #c4b5fd;
}

/* Chat styling */
.chat-message {
    padding: 16px;
    border-radius: 16px;
    margin: 8px 0;
    max-width: 85%;
}

.user-message {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    margin-left: auto;
}

.assistant-message {
    background: #f3f4f6;
    color: #1f2937;
}

/* Loading animation */
.loading-dots {
    display: inline-flex;
    gap: 4px;
}

.loading-dots span {
    width: 8px;
    height: 8px;
    background: #8b5cf6;
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}

.loading-dots span:nth-child(1) { animation-delay: -0.32s; }
.loading-dots span:nth-child(2) { animation-delay: -0.16s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* Destination card */
.destination-card {
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}

.destination-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

/* Suggestion chips */
.suggestion-chip {
    display: inline-block;
    padding: 8px 16px;
    background: #f3e8ff;
    color: #7c3aed;
    border-radius: 20px;
    margin: 4px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.9em;
}

.suggestion-chip:hover {
    background: #7c3aed;
    color: white;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: #7c3aed;
}

.metric-label {
    color: #6b7280;
    font-size: 0.9em;
}
"""


class TripPilotUI:
    """
    Agentic UI for TripPilot - visualizes agent thinking and tool use.
    """

    def __init__(self):
        self.agent = TravelAgent()
        self.chat_history = []
        self.current_execution = []

    def format_thought(self, thought: str, step: int) -> str:
        """Format a thought bubble with step indicator"""
        return f"""
<div class="timeline-item">
    <div style="display: flex; align-items: flex-start;">
        <span class="step-indicator">{step}</span>
        <div class="thought-bubble" style="flex: 1;">
            <strong>Thinking...</strong><br/>
            {thought}
        </div>
    </div>
</div>
"""

    def format_tool_call(self, tool_name: str, args: dict, result: dict = None, duration: float = None) -> str:
        """Format a tool call card"""
        tool_icons = {
            "search_destinations": "🔍",
            "find_hotels": "🏨",
            "get_restaurants": "🍽️",
            "create_itinerary": "📅",
            "get_attractions": "🎯",
            "check_weather": "⛅",
            "get_travel_tips": "💡",
            "estimate_budget": "💰"
        }

        icon = tool_icons.get(tool_name, "🔧")
        args_str = json.dumps(args, indent=2)

        result_html = ""
        if result:
            # Format result nicely
            if "destinations" in result:
                result_html = self._format_destinations(result["destinations"])
            elif "hotels" in result:
                result_html = self._format_hotels(result["hotels"])
            elif "restaurants" in result:
                result_html = self._format_restaurants(result["restaurants"])
            elif "attractions" in result:
                result_html = self._format_attractions(result["attractions"])
            elif "itinerary" in result:
                result_html = self._format_itinerary(result["itinerary"])
            elif "forecast" in result:
                result_html = self._format_weather(result["forecast"])
            elif "tips" in result:
                result_html = self._format_tips(result["tips"])
            elif "daily_breakdown" in result:
                result_html = self._format_budget(result)
            else:
                result_html = f"<pre>{json.dumps(result, indent=2)[:500]}</pre>"

        duration_str = f"<span style='color: #6b7280; font-size: 0.85em;'>⏱️ {duration:.2f}s</span>" if duration else ""

        return f"""
<div class="tool-card">
    <div class="tool-name">{icon} {tool_name.replace('_', ' ').title()} {duration_str}</div>
    <details>
        <summary style="cursor: pointer; color: #6b7280; margin-top: 8px;">View parameters</summary>
        <div class="tool-args">{args_str}</div>
    </details>
    {f'<div class="result-card" style="margin-top: 12px;">{result_html}</div>' if result_html else ''}
</div>
"""

    def _format_destinations(self, destinations: list) -> str:
        """Format destination results"""
        html = "<div style='display: grid; gap: 12px;'>"
        for dest in destinations[:3]:
            rating = dest.get('rating', 0)
            stars = "⭐" * int(rating)
            html += f"""
<div style="background: white; padding: 16px; border-radius: 12px; border: 1px solid #e5e7eb;">
    <strong style="font-size: 1.1em; color: #1f2937;">{dest.get('name', 'Unknown')}</strong>
    <div style="color: #6b7280; margin: 8px 0;">{dest.get('description', '')[:100]}...</div>
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;">
        <span style="background: #fef3c7; color: #92400e; padding: 4px 8px; border-radius: 6px; font-size: 0.85em;">{stars} {rating:.1f}</span>
        <span style="background: #dbeafe; color: #1e40af; padding: 4px 8px; border-radius: 6px; font-size: 0.85em;">🗓️ {dest.get('best_time', 'Year-round')}</span>
    </div>
</div>
"""
        html += "</div>"
        return html

    def _format_hotels(self, hotels: list) -> str:
        """Format hotel results"""
        html = "<div style='display: grid; gap: 10px;'>"
        for hotel in hotels[:3]:
            html += f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: white; padding: 12px 16px; border-radius: 10px; border: 1px solid #e5e7eb;">
    <div>
        <strong>{hotel['name']}</strong>
        <div style="color: #6b7280; font-size: 0.9em;">📍 {hotel['location']}</div>
    </div>
    <div style="text-align: right;">
        <div style="font-weight: 700; color: #059669;">${hotel['price_per_night']}<span style="font-weight: 400; color: #6b7280;">/night</span></div>
        <div style="color: #eab308;">{'⭐' * int(hotel['rating'])}</div>
    </div>
</div>
"""
        html += "</div>"
        return html

    def _format_restaurants(self, restaurants: list) -> str:
        """Format restaurant results"""
        html = "<div style='display: grid; gap: 10px;'>"
        for rest in restaurants[:3]:
            html += f"""
<div style="background: white; padding: 12px 16px; border-radius: 10px; border: 1px solid #e5e7eb;">
    <div style="display: flex; justify-content: space-between;">
        <strong>{rest['name']}</strong>
        <span style="color: #059669;">{rest['price_level']}</span>
    </div>
    <div style="color: #6b7280; font-size: 0.9em; margin-top: 4px;">
        🍴 {rest['cuisine']} • Must try: {rest.get('must_try', 'Chef special')}
    </div>
</div>
"""
        html += "</div>"
        return html

    def _format_attractions(self, attractions: list) -> str:
        """Format attraction results"""
        html = "<div style='display: grid; gap: 8px;'>"
        for attr in attractions[:5]:
            html += f"""
<div style="display: flex; align-items: center; gap: 12px; background: white; padding: 10px 14px; border-radius: 8px; border: 1px solid #e5e7eb;">
    <span style="font-size: 1.5em;">🎯</span>
    <div style="flex: 1;">
        <strong>{attr['name']}</strong>
        <div style="color: #6b7280; font-size: 0.85em;">{attr['category']} • {attr['visit_duration']}</div>
    </div>
    <span style="color: #eab308;">⭐ {attr['rating']:.1f}</span>
</div>
"""
        html += "</div>"
        return html

    def _format_itinerary(self, itinerary: list) -> str:
        """Format itinerary results"""
        html = "<div style='display: grid; gap: 12px;'>"
        for day in itinerary[:3]:
            activities_html = ""
            for act in day["activities"]:
                activities_html += f"<div style='padding: 6px 0; border-bottom: 1px solid #f3f4f6;'><span style='color: #7c3aed; font-weight: 600;'>{act['time']}</span>: {act['activity']}</div>"

            html += f"""
<div style="background: white; padding: 14px; border-radius: 10px; border: 1px solid #e5e7eb;">
    <div style="font-weight: 700; color: #4338ca; margin-bottom: 8px;">{day['theme']}</div>
    {activities_html}
</div>
"""
        if len(itinerary) > 3:
            html += f"<div style='text-align: center; color: #6b7280; font-style: italic;'>+ {len(itinerary) - 3} more days...</div>"
        html += "</div>"
        return html

    def _format_weather(self, forecast: list) -> str:
        """Format weather results"""
        html = "<div style='display: flex; gap: 8px; overflow-x: auto; padding: 8px 0;'>"
        weather_icons = {
            "Sunny": "☀️", "Partly Cloudy": "⛅", "Cloudy": "☁️",
            "Light Rain": "🌧️", "Clear": "🌤️"
        }
        for day in forecast[:5]:
            icon = weather_icons.get(day['condition'], "🌡️")
            html += f"""
<div style="background: white; padding: 12px; border-radius: 10px; text-align: center; min-width: 80px; border: 1px solid #e5e7eb;">
    <div style="font-size: 1.8em;">{icon}</div>
    <div style="font-weight: 600; font-size: 0.85em; margin: 4px 0;">{day['day'].split(',')[0]}</div>
    <div style="color: #ef4444;">{day['high']}°</div>
    <div style="color: #3b82f6;">{day['low']}°</div>
</div>
"""
        html += "</div>"
        return html

    def _format_tips(self, tips: dict) -> str:
        """Format travel tips"""
        html = "<div style='display: grid; gap: 8px;'>"
        icons = {"transportation": "🚇", "safety": "🛡️", "culture": "🎭", "money_saving": "💵", "packing": "🎒"}
        for category, tip_list in list(tips.items())[:3]:
            icon = icons.get(category, "💡")
            html += f"""
<div style="background: white; padding: 10px 14px; border-radius: 8px; border: 1px solid #e5e7eb;">
    <strong>{icon} {category.replace('_', ' ').title()}</strong>
    <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #4b5563;">
        {''.join(f'<li style="margin: 4px 0;">{tip}</li>' for tip in tip_list[:2])}
    </ul>
</div>
"""
        html += "</div>"
        return html

    def _format_budget(self, budget: dict) -> str:
        """Format budget estimate"""
        breakdown = budget.get('daily_breakdown', {})
        return f"""
<div style="background: white; padding: 16px; border-radius: 10px;">
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;">
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="font-size: 1.8em; font-weight: 700; color: #059669;">${budget.get('daily_total', 0)}</div>
            <div style="color: #6b7280; font-size: 0.85em;">Daily Budget</div>
        </div>
        <div style="text-align: center; padding: 12px; background: #f3f4f6; border-radius: 8px;">
            <div style="font-size: 1.8em; font-weight: 700; color: #7c3aed;">${budget.get('trip_total', 0)}</div>
            <div style="color: #6b7280; font-size: 0.85em;">Total ({budget.get('days', 5)} days)</div>
        </div>
    </div>
    <div style="margin-top: 12px; font-size: 0.9em; color: #6b7280;">
        🏨 Accommodation: ${breakdown.get('accommodation', 0)} •
        🍽️ Food: ${breakdown.get('food', 0)} •
        🎯 Activities: ${breakdown.get('activities', 0)} •
        🚇 Transport: ${breakdown.get('transport', 0)}
    </div>
</div>
"""

    def get_status_html(self, state: AgentState) -> str:
        """Generate status indicator HTML"""
        status_config = {
            AgentState.IDLE: ("status-idle", "🟣 Ready", "Ask me anything about travel!"),
            AgentState.THINKING: ("status-thinking", "🧠 Thinking", "Analyzing your request..."),
            AgentState.TOOL_CALLING: ("status-tool", "🔧 Working", "Gathering information..."),
            AgentState.GENERATING: ("status-tool", "✍️ Writing", "Composing response..."),
            AgentState.COMPLETE: ("status-complete", "✅ Complete", "Here's what I found!"),
            AgentState.ERROR: ("status-idle", "❌ Error", "Something went wrong"),
        }

        css_class, title, subtitle = status_config.get(state, ("status-idle", "Ready", ""))

        return f"""
<div class="agent-status {css_class}">
    <div>
        <div style="font-size: 1.1em;">{title}</div>
        <div style="font-size: 0.85em; opacity: 0.8;">{subtitle}</div>
    </div>
</div>
"""

    def process_message(self, message: str, history: list) -> Generator:
        """
        Process a user message and yield updates for the UI.
        """
        if not message.strip():
            yield history, "", self.get_status_html(AgentState.IDLE)
            return

        # Add user message to history
        history = history + [[message, None]]

        # Initialize execution log
        execution_html = "<div class='timeline'>"

        # Run agent and yield updates
        for state, data in self.agent.run(message):
            if state == AgentState.THINKING:
                if "thought" in data:
                    execution_html += self.format_thought(data["thought"], data.get("step", 1))
                yield history, execution_html + "</div>", self.get_status_html(state)

            elif state == AgentState.TOOL_CALLING:
                if "result" in data:
                    execution_html += self.format_tool_call(
                        data["tool"],
                        data.get("args", {}),
                        data.get("result"),
                        data.get("duration")
                    )
                yield history, execution_html + "</div>", self.get_status_html(state)

            elif state == AgentState.GENERATING:
                yield history, execution_html + "</div>", self.get_status_html(state)

            elif state == AgentState.COMPLETE:
                # Update history with final answer
                history[-1][1] = data["answer"]
                execution_html += f"""
<div class="timeline-item">
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 16px; border-radius: 12px; margin-top: 12px;">
        <strong>✅ Task completed in {data['time']:.2f}s</strong>
    </div>
</div>
"""
                yield history, execution_html + "</div>", self.get_status_html(state)

        # Final yield
        yield history, execution_html + "</div>", self.get_status_html(AgentState.IDLE)

    def get_example_queries(self) -> list:
        """Return example queries for the UI"""
        return [
            ["Plan a 5-day trip to Paris with focus on art and cuisine"],
            ["Find budget-friendly hotels in Tokyo for next month"],
            ["What are the top attractions to visit in Barcelona?"],
            ["Create a romantic itinerary for Bali"],
            ["I want to visit New York - give me restaurants and travel tips"],
            ["Estimate the budget for a week in Rome, moderate style"],
            ["What's the weather like in Barcelona and what should I pack?"],
            ["I'm looking for a beach destination with good food"]
        ]


def create_ui() -> gr.Blocks:
    """Create and return the Gradio interface"""

    ui = TripPilotUI()

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="TripPilot - AI Travel Planner",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="blue",
            neutral_hue="slate"
        )
    ) as demo:
        # Header
        gr.HTML("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 2.5em; margin: 0; background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
        ✈️ TripPilot
    </h1>
    <p style="color: #6b7280; margin-top: 8px; font-size: 1.1em;">
        AI-Powered Travel Planning Assistant
    </p>
</div>
        """)

        with gr.Row():
            # Left column - Chat interface
            with gr.Column(scale=3):
                # Agent status
                status_display = gr.HTML(
                    value=ui.get_status_html(AgentState.IDLE),
                    elem_id="agent-status"
                )

                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=False,
                    avatar_images=(None, "https://em-content.zobj.net/source/apple/391/robot_1f916.png"),
                    bubble_full_width=False
                )

                # Input area
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask me about travel destinations, hotels, itineraries...",
                        show_label=False,
                        scale=5,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                # Example queries
                gr.HTML("<div style='margin-top: 16px; color: #6b7280; font-size: 0.9em;'>Try asking:</div>")
                examples = gr.Examples(
                    examples=ui.get_example_queries(),
                    inputs=msg_input,
                    label=""
                )

            # Right column - Agent execution visualization
            with gr.Column(scale=2):
                gr.HTML("""
<div style="padding: 12px 16px; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 12px; margin-bottom: 16px;">
    <h3 style="margin: 0; color: #5b21b6;">🔮 Agent Reasoning</h3>
    <p style="margin: 8px 0 0 0; color: #7c3aed; font-size: 0.9em;">Watch the AI think and use tools in real-time</p>
</div>
                """)

                execution_display = gr.HTML(
                    value="<div style='text-align: center; padding: 40px; color: #9ca3af;'>Agent execution steps will appear here...</div>",
                    elem_id="execution-log"
                )

        # Footer
        gr.HTML("""
<div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #e5e7eb; color: #9ca3af; font-size: 0.85em;">
    <p>🚀 Powered by AI • Built with Gradio</p>
    <p>TripPilot demonstrates agentic AI for travel planning with tool use and reasoning visualization</p>
</div>
        """)

        # Event handlers
        def respond(message, history):
            """Handle message submission"""
            for hist, exec_html, status_html in ui.process_message(message, history):
                yield hist, exec_html, status_html, ""

        # Connect events
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, execution_display, status_display, msg_input]
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, execution_display, status_display, msg_input]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
