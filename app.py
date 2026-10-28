import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph-Based Recommendation System",
    page_icon="🔗",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/interactions.csv")
    return df


@st.cache_data
def build_graph(df_tuple):
    """Build a bipartite graph from (user_id, item_id) pairs."""
    df = pd.DataFrame(df_tuple[1], columns=df_tuple[0])
    G = nx.Graph()

    users = df["user_id"].unique().tolist()
    items = df["item_id"].unique().tolist()

    # Add nodes with a 'bipartite' attribute so we can colour them later
    for u in users:
        G.add_node(u, node_type="user")
    for i in items:
        G.add_node(i, node_type="item")

    # Add edges
    for _, row in df.iterrows():
        G.add_edge(row["user_id"], row["item_id"])

    return G


def get_user_items(G, user_id):
    """Return the set of items a user interacted with."""
    if user_id not in G:
        return set()
    return {n for n in G.neighbors(user_id) if G.nodes[n]["node_type"] == "item"}


def find_similar_users(G, user_id, user_items):
    """
    Similar users = other users who share ≥1 item with the selected user.
    Returns a dict: {other_user: set_of_common_items}
    """
    similar = {}
    for item in user_items:
        for neighbour in G.neighbors(item):
            if neighbour == user_id or G.nodes[neighbour]["node_type"] != "user":
                continue
            if neighbour not in similar:
                similar[neighbour] = set()
            similar[neighbour].add(item)
    return similar


def recommend_items(G, user_id, user_items, similar_users_dict, top_n=5):
    """
    Recommend items the selected user has NOT interacted with,
    ranked by how many similar users interacted with each item.
    Ties are broken alphabetically by item_id.
    """
    scores = defaultdict(int)
    for sim_user, common_items in similar_users_dict.items():
        sim_user_items = get_user_items(G, sim_user)
        for item in sim_user_items:
            if item not in user_items:
                scores[item] += 1

    # Sort: descending score, then ascending item name for consistent tie-break
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_n]


def draw_graph(G, selected_user, similar_users_dict, recommended_items):
    """Draw a focused subgraph for the selected user."""
    highlighted_nodes = {selected_user}
    similar_set = set(similar_users_dict.keys())
    rec_set = {item for item, _ in recommended_items}

    # Build subgraph: selected user + their items + similar users + their items
    sub_nodes = {selected_user}
    for item in get_user_items(G, selected_user):
        sub_nodes.add(item)
    for sim_user in similar_set:
        sub_nodes.add(sim_user)
        for item in get_user_items(G, sim_user):
            sub_nodes.add(item)

    SG = G.subgraph(sub_nodes)

    # Colour map
    color_map = []
    for node in SG.nodes():
        if node == selected_user:
            color_map.append("#E74C3C")          # red  – selected user
        elif node in similar_set:
            color_map.append("#3498DB")          # blue – similar users
        elif node in rec_set:
            color_map.append("#2ECC71")          # green – recommended items
        elif SG.nodes[node]["node_type"] == "item":
            color_map.append("#F39C12")          # orange – existing items
        else:
            color_map.append("#95A5A6")          # grey – other users

    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(SG, seed=42)
    nx.draw_networkx(
        SG, pos=pos, ax=ax,
        node_color=color_map,
        node_size=700,
        font_size=8,
        font_color="white",
        font_weight="bold",
        edge_color="#CCCCCC",
        with_labels=True,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#E74C3C", label="Selected User"),
        mpatches.Patch(color="#3498DB", label="Similar Users"),
        mpatches.Patch(color="#F39C12", label="Existing Items"),
        mpatches.Patch(color="#2ECC71", label="Recommended Items"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.set_title("User–Item Bipartite Graph (Focused View)", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.title("🔗 Graph-Based Recommendation System")
    st.markdown("**Using User–Item Networks**")
    st.divider()

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Could not find `data/interactions.csv`. Make sure it exists.")
        st.stop()

    # Build graph (cache requires hashable key, so pass as tuple-of-tuples)
    G = build_graph((tuple(df.columns), [tuple(r) for r in df.values]))

    all_users = sorted(df["user_id"].unique().tolist())

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        selected_user = st.selectbox("Select a User", all_users)
        top_n = st.radio("Number of Recommendations", [3, 5], index=1)
        st.divider()
        st.markdown(
            "**How it works**\n"
            "1. A bipartite graph connects users to items.\n"
            "2. Similar users share items with the selected user.\n"
            "3. Items liked by similar users but not the selected user are recommended.\n"
            "4. Items are ranked by how many similar users interacted with them."
        )

    # ── User's existing items ─────────────────────────────────────────────────
    user_items = get_user_items(G, selected_user)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📦 Items for {selected_user}")
        if not user_items:
            st.warning(f"{selected_user} has no recorded interactions.")
        else:
            for item in sorted(user_items):
                st.markdown(f"- **{item}**")

    # ── Similar users ─────────────────────────────────────────────────────────
    similar_users_dict = find_similar_users(G, selected_user, user_items)

    with col2:
        st.subheader("👥 Similar Users")
        if not user_items:
            st.info("Cannot find similar users — no interactions recorded.")
        elif not similar_users_dict:
            st.info(f"No users share items with {selected_user}.")
        else:
            sim_df = pd.DataFrame(
                [
                    {
                        "User": u,
                        "Common Items": len(items),
                        "Shared Items": ", ".join(sorted(items)),
                    }
                    for u, items in sorted(
                        similar_users_dict.items(),
                        key=lambda x: (-len(x[1]), x[0]),
                    )
                ]
            )
            st.dataframe(sim_df, width='stretch', hide_index=True)

    st.divider()

    # ── Recommendations ───────────────────────────────────────────────────────
    st.subheader(f"🎯 Top {top_n} Recommendations for {selected_user}")

    if not user_items:
        st.warning("No interactions recorded — cannot generate recommendations.")
    elif not similar_users_dict:
        st.warning("No similar users found — cannot generate recommendations.")
    else:
        recommendations = recommend_items(
            G, selected_user, user_items, similar_users_dict, top_n=top_n
        )

        if not recommendations:
            st.info("All items have already been interacted with — nothing new to recommend.")
        else:
            rec_df = pd.DataFrame(
                [
                    {"Rank": i + 1, "Item": item, "Score (Similar Users)": score}
                    for i, (item, score) in enumerate(recommendations)
                ]
            )
            st.dataframe(rec_df, width='stretch', hide_index=True)

            # Show top 3 explicitly if top_n == 5
            if top_n == 5 and len(recommendations) >= 3:
                st.markdown("**Top 3 (highlighted):**")
                for i, (item, score) in enumerate(recommendations[:3]):
                    st.markdown(f"&nbsp;&nbsp;🥇🥈🥉"[i] + f" **{item}** — recommended by {score} similar user(s)")

    st.divider()

    # ── Graph visualization ───────────────────────────────────────────────────
    st.subheader("🌐 Graph Visualization")

    if not user_items:
        st.info("No graph to show — selected user has no interactions.")
    else:
        recommendations_for_graph = recommend_items(
            G, selected_user, user_items, similar_users_dict, top_n=5
        )
        fig = draw_graph(G, selected_user, similar_users_dict, recommendations_for_graph)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.caption("Graph-Based Recommendation System | College Project Demo")


if __name__ == "__main__":
    main()
