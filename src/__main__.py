"""CLI for gridmind."""
import sys, json, argparse
from .core import Gridmind

def main():
    parser = argparse.ArgumentParser(description="GridMind — Building Energy Optimizer. AI-powered HVAC and energy optimization for commercial buildings.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = Gridmind()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.generate(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"gridmind v0.1.0 — GridMind — Building Energy Optimizer. AI-powered HVAC and energy optimization for commercial buildings.")

if __name__ == "__main__":
    main()
