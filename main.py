import time

from transformers import pipeline

MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

HEADLINES = [
    "Company raises full year profit forecast after record quarterly sales",
    "Shares plunge as regulators open investigation into accounting practices",
    "The board will announce its quarterly results on Thursday",
]


def run_once(classifier, label):
    start = time.perf_counter()
    results = classifier(HEADLINES)
    elapsed = time.perf_counter() - start
    print(f"\n{label}: {elapsed:.3f} seconds")
    for text, result in zip(HEADLINES, results):
        print(f"  {result['label']:<9} {result['score']:.3f}  {text}")


def main():
    load_start = time.perf_counter()
    classifier = pipeline("text-classification", model=MODEL_ID)
    print(f"Model load time: {time.perf_counter() - load_start:.3f} seconds")

    run_once(classifier, "First run")
    run_once(classifier, "Second run")


if __name__ == "__main__":
    main()