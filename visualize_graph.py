import networkx as nx
from typing import List, Dict, Any, Tuple
from py2neo import Graph as Py2NeoGraph
import re
from pyvis.network import Network
import tempfile
import os


uri = "neo4j+s://a36f6471.databases.neo4j.io"
username = "neo4j"
password = "E7QyiOiiDAi0SjQY7eXvUyjNPNzEHa4sYsMdlTqc1gI"

graph = Py2NeoGraph(uri, auth=(username, password))


def fuzzy_node_type(key: str) -> str:
    key_lower = key.lower()
    if "paper" in key_lower or "title" in key_lower:
        return "Papers"
    elif "author" in key_lower:
        return "Authors"
    elif "year" in key_lower:
        return "Years"
    elif "source" in key_lower or "journal" in key_lower or "conference" in key_lower:
        return "Sources"
    elif "keyword" in key_lower or "key" in key_lower:
        return "Keywords"
    else:
        return key.capitalize()


RELATIONSHIP_MAP = {
    ("Authors", "Papers"): "WROTE",
    ("Papers", "Authors"): "WROTE",
    ("Papers", "Papers"): "CITES",
    ("Papers", "Years"): "PUBLISHED_IN_YEAR",
    ("Years", "Papers"): "PUBLISHED_IN_YEAR",
    ("Papers", "Sources"): "PUBLISHED_IN_SOURCE",
    ("Sources", "Papers"): "PUBLISHED_IN_SOURCE",
    ("Papers", "Keywords"): "KEYWORD",
    ("Keywords", "Papers"): "KEYWORD",
}


def get_relationship_and_direction(label1, label2):
    """
    返回关系名称和边的方向 (from_label, to_label)
    """
    if (label1, label2) in RELATIONSHIP_MAP:
        return RELATIONSHIP_MAP[(label1, label2)], (label1, label2)
    elif (label2, label1) in RELATIONSHIP_MAP:
        return RELATIONSHIP_MAP[(label2, label1)], (label2, label1)
    else:
        return "RELATED", (label1, label2)


def context_and_entities_to_graph(context_result, corrected_entities):
    
    G = nx.DiGraph()
    added_nodes = set()

    for entity_type, values in corrected_entities.items():
        for value in values:
            node_id = f"{entity_type}:{value}"
            if node_id not in added_nodes:
                G.add_node(node_id, label=entity_type, name=value)
                added_nodes.add(node_id)

    for record in context_result:
        keys = list(record.keys())
        for key in keys:
            val = record[key]
            if not val:
                continue
            label = fuzzy_node_type(key)
            node_id = f"{label}:{val}"
            if node_id not in added_nodes:
                G.add_node(node_id, label=label, name=val)
                added_nodes.add(node_id)

        if len(keys) > 1:
            for i, key1 in enumerate(keys):
                val1 = record[key1]
                if not val1:
                    continue
                label1 = fuzzy_node_type(key1)
                id1 = f"{label1}:{val1}"
                for j in range(i+1, len(keys)):
                    val2 = record[keys[j]]
                    if not val2:
                        continue
                    label2 = fuzzy_node_type(keys[j])
                    id2 = f"{label2}:{val2}"

                    rel, direction = get_relationship_and_direction(label1, label2)
                    if direction == (label1, label2):
                        if not G.has_edge(id1, id2):
                            G.add_edge(id1, id2, label=rel)
                    else:
                        if not G.has_edge(id2, id1):
                            G.add_edge(id2, id1, label=rel)

    published_in_year_edges = set()
    published_in_source_edges = set()
    for u, v, attr in G.edges(data=True):
        if attr.get('label') == "PUBLISHED_IN_YEAR":
            published_in_year_edges.add((u,v))
        if attr.get('label') == "PUBLISHED_IN_SOURCE":
            published_in_source_edges.add((u,v))

    context_nodes = {f"{fuzzy_node_type(k)}:{v}" for record in context_result for k,v in record.items() if v}
    corrected_nodes = {f"{k}:{v}" for k, vals in corrected_entities.items() for v in vals}

    all_nodes = added_nodes

    for ent_node in corrected_nodes:
        ent_label, ent_name = ent_node.split(":", 1)
        for ctx_node in context_nodes:
            ctx_label, ctx_name = ctx_node.split(":", 1)
            if ent_node == ctx_node:
                continue  

            if ent_node not in all_nodes or ctx_node not in all_nodes:
                continue

            if (ent_label == "Papers" and ctx_label == "Years") or (ent_label == "Years" and ctx_label == "Papers"):
                if (ent_node, ctx_node) in published_in_year_edges or (ctx_node, ent_node) in published_in_year_edges:
                    continue
            if (ent_label == "Papers" and ctx_label == "Sources") or (ent_label == "Sources" and ctx_label == "Papers"):
                if (ent_node, ctx_node) in published_in_source_edges or (ctx_node, ent_node) in published_in_source_edges:
                    continue

            rel, direction = get_relationship_and_direction(ent_label, ctx_label)
            if direction == (ent_label, ctx_label):
                if not G.has_edge(ent_node, ctx_node):
                    G.add_edge(ent_node, ctx_node, label=rel)
            else:
                if not G.has_edge(ctx_node, ent_node):
                    G.add_edge(ctx_node, ent_node, label=rel)

    paper_nodes = [n for n, d in G.nodes(data=True) if d.get("label") == "Papers"]
    if len(paper_nodes) > 1:
        paper_titles = [G.nodes[p]["name"] for p in paper_nodes]
        titles_list_str = ", ".join([f"'{t}'" for t in paper_titles])
        cypher = f"""
        MATCH (p1:Paper)-[:CITES]->(p2:Paper)
        WHERE p1.title IN [{titles_list_str}] AND p2.title IN [{titles_list_str}]
        RETURN p1.title AS citing, p2.title AS cited
        """
        cite_results = graph.run(cypher).data()
        for row in cite_results:
            citing = row.get("citing")
            cited = row.get("cited")
            id_citing = f"Papers:{citing}"
            id_cited = f"Papers:{cited}"
            if id_citing in G and id_cited in G and not G.has_edge(id_citing, id_cited):
                G.add_edge(id_citing, id_cited, label="CITES")

    return G


def visualize_graph(G):
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")

    net.set_options("""
        var options = {
          "nodes": {
            "font": {
              "size": 20,
              "face": "arial",
              "color": "#000000"
            },
            "physics": true
          },
          "interaction": {
            "dragNodes": true
          },
          "manipulation": {
            "enabled": false
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 120,
              "springConstant": 0.08,
              "avoidOverlap": 1
            },
            "minVelocity": 0.75,
            "stabilization": {
              "enabled": true,
              "iterations": 1000,
              "updateInterval": 25
            }
          }
        }
    """)

    for node_id, data in G.nodes(data=True):
        net.add_node(
            node_id,
            label=data["name"],
            title=data["label"],
            group=data["label"],
            size=20,
            font={"size": 20, "color": "black"}
        )

    for u, v, data in G.edges(data=True):
        net.add_edge(u, v, label=data["label"])

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.write_html(tmp_file.name)
    return tmp_file.name