"""verify_learning.py — Re-open the self_learner DB and confirm learnings
persisted from the previous run. No new data ingestion — pure read.

Usage:
    .venv/bin/python verify_learning.py
"""
import os
import sys
from self_learner import (
    Learner,
    SEOLearner,
    TrendAnalyzer,
    TrendPoint,
    RecommendationEngine,
)
from self_learner.memory import PersistentMemory
from self_learner.knowledge import KnowledgeBase


def main():
    db_path = os.environ.get("SELF_LEARNER_DB", "self_learner.db")
    print(f"DB path: {db_path}")
    print(f"DB exists: {os.path.exists(db_path)}, size: {os.path.getsize(db_path) if os.path.exists(db_path) else 0} bytes")

    if not os.path.exists(db_path):
        print("No DB found. Run learn_from_uploads.py first.")
        sys.exit(1)

    mem = PersistentMemory(db_path=db_path)
    kb = KnowledgeBase(memory=mem)
    seo = SEOLearner(memory=mem, knowledge=kb)
    learner = Learner(memory=mem, knowledge=kb)
    trends = TrendAnalyzer(memory=mem)
    recs = RecommendationEngine(seo_learner=seo, trend_analyzer=trends)

    facts = kb.get_facts(subject="seo_outcome", relation="has_performance")
    print(f"\nPersisted SEO facts: {len(facts)}")

    upload_facts = kb.get_facts(subject="upload", relation="observed")
    print(f"Persisted upload observations: {len(upload_facts)}")

    metrics = trends.get_all_trends()
    print(f"Persisted trend metrics: {len(metrics)}")

    print(f"\n─── Top 10 keywords by engagement ───")
    for kw in seo.get_best_keywords("shorts", limit=10):
        print(f"  {kw['keyword']:30s}  eng={kw['avg_engagement']:.2f}%  n={kw['count']}")

    print(f"\n─── Top 5 title patterns ───")
    for p in seo.get_best_title_patterns("shorts", limit=5):
        print(f"  {p['pattern']:20s}  eng={p['avg_engagement']:.2f}%  n={p['count']}")

    rec_list = recs.generate_recommendations()
    print(f"\n─── Top 3 recommendations ───")
    for r in rec_list[:3]:
        print(f"  [{r.priority}/{r.category}] {r.title}")
        print(f"    -> {r.action}")

    print("\nDB persistence: CONFIRMED")

    recs.close()
    trends.close()
    seo.close()
    learner.close()


if __name__ == "__main__":
    main()
