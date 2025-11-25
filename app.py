#!/usr/bin/env python3
"""
TripPilot - AI-Powered Travel Planning with Agentic UI

Launch the application with:
    python app.py

Or with custom settings:
    python app.py --port 7860 --share
"""

import argparse
import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tripilot.ui import create_ui


def main():
    """Main entry point for TripPilot application"""

    parser = argparse.ArgumentParser(
        description="TripPilot - AI-Powered Travel Planning Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Run locally on default port
  python app.py --port 8080        # Run on custom port
  python app.py --share            # Create public shareable link
  python app.py --debug            # Enable debug mode
        """
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with auto-reload"
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )

    args = parser.parse_args()

    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ✈️  TripPilot - AI-Powered Travel Planning                  ║
║                                                               ║
║   An Agentic UI demonstrating AI reasoning and tool use       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

    print(f"🚀 Starting TripPilot...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {args.share}")
    print(f"   Debug: {args.debug}")
    print()

    # Create and launch the UI
    demo = create_ui()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        inbrowser=not args.no_browser,
        show_error=True
    )


if __name__ == "__main__":
    main()
