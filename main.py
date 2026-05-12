"""Main entry point."""
import argparse
import sys


def run_cli():
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    import subprocess
    print("Starting Streamlit...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


def run_evaluation():
    from dotenv import load_dotenv
    import yaml
    load_dotenv()
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    try:
        from src.evaluation.evaluator import SystemEvaluator
        SystemEvaluator(cfg).run_full_evaluation()
    except Exception:
        from src.autogen_orchestrator import AutoGenOrchestrator
        o = AutoGenOrchestrator(cfg)
        r = o.process_query("What are the key principles of accessible UI design in HCI?")
        print("\nResponse:\n", r.get("response", ""))
        print("\nMetadata:", r.get("metadata", {}))


def run_autogen():
    import subprocess
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "web", "evaluate", "autogen"], default="autogen")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        run_evaluation()
    else:
        run_autogen()


if __name__ == "__main__":
    main()
