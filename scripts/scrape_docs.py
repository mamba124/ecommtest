"""
One-shot scraper for Anthropic documentation.
Usage: python scripts/scrape_docs.py --output data/raw_docs --max-pages 50
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.scraper import DocScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Anthropic documentation")
    parser.add_argument("--output", default="data/raw_docs", help="Output directory")
    parser.add_argument("--max-pages", type=int, default=50)
    args = parser.parse_args()

    scraper = DocScraper(
        base_url="https://docs.anthropic.com",
        output_dir=args.output,
        max_pages=args.max_pages,
    )
    scraper.scrape()
