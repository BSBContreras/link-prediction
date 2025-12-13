import pandas as pd
import itertools
import numpy as np
from collections import defaultdict, Counter

authors_df = pd.read_csv("database/authorships.csv")
print("Unique authors:", len(authors_df["author_id"].unique()))
print("Unique works:", len(authors_df["work_id"].unique()))
print("Size of authors_df:", len(authors_df))


works_df = pd.read_csv("database/works.csv")
print("Size of works_df:", len(works_df))


merged_df = authors_df.merge(
    works_df[["id", "publication_date"]], left_on="work_id", right_on="id"
)
merged_df["publication_date"] = pd.to_datetime(
    merged_df["publication_date"], errors="coerce"
)
merged_df = merged_df.dropna(subset=["publication_date", "author_id"]).drop(
    columns=["id"]
)


print("Size of merged_df:", len(merged_df))


unique_works = (
    merged_df[["work_id", "publication_date"]]
    .drop_duplicates()
    .sort_values("publication_date")
)
split_idx = int(len(unique_works) * 0.8)

train_work_ids = set(unique_works.iloc[:split_idx]["work_id"])
test_work_ids = set(unique_works.iloc[split_idx:]["work_id"])

train_df = merged_df[merged_df["work_id"].isin(train_work_ids)]
test_df = merged_df[merged_df["work_id"].isin(test_work_ids)]

print("Size of unique_works:", len(unique_works))
print("Train work ids:", len(train_work_ids))
print("Test work ids:", len(test_work_ids))


def build_graph(df):
    graph = defaultdict(set)
    for _, group in df.groupby("work_id"):
        authors = group["author_id"].tolist()

        if len(authors) > 1:
            for u, v in itertools.permutations(authors, 2):
                graph[u].add(v)

    return graph


train_graph = build_graph(train_df)
print("Number of Authors in Train Graph:", len(train_graph))
print(
    f"Mean of coauthors per author: {np.mean([len(v) for v in train_graph.values()]):.2f}"
)

test_graph_raw = build_graph(test_df)
print("Number of Authors in Test Graph:", len(test_graph_raw))
print(
    f"Mean of coauthors per author: {np.mean([len(v) for v in test_graph_raw.values()]):.2f}"
)


test_ground_truth = defaultdict(set)

for author, coauthors in test_graph_raw.items():
    # Pega quem o autor colaborou no futuro
    future_coauthors = coauthors

    # Remove quem ele JÁ conhecia no passado (não é predição nova)
    past_coauthors = train_graph.get(author, set())
    new_links = future_coauthors - past_coauthors

    if new_links:
        test_ground_truth[author] = new_links

print(f"Authors in Train Graph: {len(train_graph)}")
print(f"Authors with new connections in Test Graph: {len(test_ground_truth)}")


popularity_counter = Counter()
for author, neighbors in train_graph.items():
    popularity_counter[author] = len(neighbors)


most_popular_authors = [auth for auth, count in popularity_counter.most_common()]


def recommend_coauthors(author_id, graph, top_n=10):
    recommendations = []

    if author_id in graph:
        current_coauthors = graph[author_id]
        candidates = []

        for neighbor in current_coauthors:
            neighbors_of_neighbor = graph.get(neighbor, set())
            for candidate in neighbors_of_neighbor:
                if candidate != author_id and candidate not in current_coauthors:
                    candidates.append(candidate)

        # Pega os melhores baseados em vizinhos em comum
        recommendations = [c[0] for c in Counter(candidates).most_common(top_n)]
    else:
        # Caso Cold Start: Se o autor não está no grafo, current_coauthors é vazio
        current_coauthors = set()

    if len(recommendations) < top_n:
        for pop_author in most_popular_authors:
            if (
                pop_author != author_id
                and pop_author not in recommendations
                and pop_author not in current_coauthors
            ):

                recommendations.append(pop_author)

                if len(recommendations) >= top_n:
                    break

    return recommendations


author_id_sample = train_df.sample(1, random_state=42).iloc[0]["author_id"]
recommendations = recommend_coauthors(author_id_sample, train_graph, top_n=10)

current_coauthors = train_graph[author_id_sample]

print("Author:", author_id_sample)


K = 5
precisions = []
recalls = []

# Avaliamos APENAS autores que realmente formaram novas conexões
# (Não faz sentido avaliar quem parou de publicar ou só trabalhou com velhos amigos)
for author_id, actual_new_coauthors in test_ground_truth.items():

    recommendations = recommend_coauthors(author_id, train_graph, top_n=K)

    hits = len(set(recommendations) & actual_new_coauthors)

    # Precision: Dos que eu recomendei, quantos eram verdadeiros?
    p = hits / len(recommendations)

    # Recall: Dos que existiam de verdade, quantos eu encontrei?
    r = hits / len(actual_new_coauthors)

    precisions.append(p)
    recalls.append(r)

avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)

if (avg_precision + avg_recall) > 0:
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
else:
    f1_score = 0

print("-" * 40)
print(f"Resultados da Avaliação (Top-{K}):")
print(f"Precision: {avg_precision:.4f} {avg_precision*100:.1f}%")
print(f"Recall:    {avg_recall:.4f} {avg_recall*100:.1f}%")
print(f"F1-Score:  {f1_score:.4f} {f1_score*100:.1f}%")
print("-" * 40)


K_values = [1, 5, 10, 20, 50]
results = {}

for K in K_values:
    precisions = []
    recalls = []
    mrrs = []
    ndcgs = []

    # Avaliamos APENAS autores que realmente formaram novas conexões
    for author_id, actual_new_coauthors in test_ground_truth.items():
        recommendations = recommend_coauthors(author_id, train_graph, top_n=K)

        hits = len(set(recommendations) & actual_new_coauthors)

        # Precision: Dos que eu recomendei, quantos eram verdadeiros?
        p = hits / len(recommendations) if len(recommendations) > 0 else 0

        # Recall: Dos que existiam de verdade, quantos eu encontrei?
        r = hits / len(actual_new_coauthors) if len(actual_new_coauthors) > 0 else 0

        precisions.append(p)
        recalls.append(r)

        # MRR: Foca na posição do PRIMEIRO item relevante
        rr = 0
        for i, item in enumerate(recommendations):
            if item in actual_new_coauthors:
                rr = 1 / (i + 1)
                break
        mrrs.append(rr)

        # NDCG: Avalia a lista toda e penaliza acertos distantes do topo
        dcg = 0
        for i, item in enumerate(recommendations):
            if item in actual_new_coauthors:
                dcg += 1 / np.log2(i + 2)

        idcg = 0
        num_possible_hits = min(len(actual_new_coauthors), K)
        for i in range(num_possible_hits):
            idcg += 1 / np.log2(i + 2)

        ndcg_score = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg_score)

    # Médias finais
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_mrr = np.mean(mrrs)
    avg_ndcg = np.mean(ndcgs)

    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1_score = 0

    results[K] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": f1_score,
        "mrr": avg_mrr,
        "ndcg": avg_ndcg,
    }

    print(
        f"K={K}: Precision={avg_precision:.4f}, "
        f"Recall={avg_recall:.4f}, "
        f"F1={f1_score:.4f}, "
        f"MRR={avg_mrr:.4f}, "  # Novo output
        f"NDCG={avg_ndcg:.4f}"  # Novo output
    )

print("\n" + "=" * 60)
print("RESUMO DETALHADO DAS MÉTRICAS")
print("=" * 60)
print(
    f"{'K':<5} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'MRR':<10} {'NDCG':<10}"
)
print("-" * 60)
for K in K_values:
    res = results[K]
    print(
        f"{K:<5} {res['precision']*100:<10.2f} {res['recall']*100:<10.2f} {res['f1_score']*100:<10.2f} {res['mrr']*100:<10.2f} {res['ndcg']*100:<10.2f}"
    )
print("=" * 60)
