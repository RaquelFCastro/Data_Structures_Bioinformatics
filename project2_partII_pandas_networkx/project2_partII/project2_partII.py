"""
Project 2 — Biological Data Analysis (Part II)

Implement all functions below. Do not change their signatures.
"""

import json
import csv
import math
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — pandas - GWAS Catalogue
# ─────────────────────────────────────────────────────────────────────────────

gwas = pd.read_csv('data/gwas_catalog.tsv', sep = '\t')
print(gwas.shape)
print(gwas.head())

def mostStudiedTraits(gwas):
    """
    T3.1 — Returns a DataFrame with the traits with >= 3 studies, sorted by descending number of studies.
    """
    #group by number of unique study accession, sort descending, keep over 3 
    no_studies=gwas.groupby('DISEASE/TRAIT')['STUDY ACCESSION'].nunique().reset_index(name='n_studies')
    sort_studies = no_studies.sort_values('n_studies', ascending=False)
    over3= sort_studies[sort_studies['n_studies']>=3]

    #group by snps
    n_snps = gwas.groupby('DISEASE/TRAIT')['SNPS'].nunique().reset_index(name='n_snps')
    
    #merge the two by DISEASE/TRAIT
    result= over3.merge(n_snps, on='DISEASE/TRAIT') 

    return result

print (mostStudiedTraits(gwas))

def mostSignificantPerChrom(gwas):
    """
    T3.2 — Returns a dictionary with the most significant SNP on each chromosome (ie., with lowest p-value).
    """
    table = gwas.loc[gwas.groupby('CHR_ID')['P-VALUE'].idxmin(),['CHR_ID', 'DISEASE/TRAIT', 'SNPS','P-VALUE']]
    chrid = list(table['CHR_ID'])
    snpid = list(table['SNPS'])
    trait = list(table['DISEASE/TRAIT'])
    pvalue = list(table['P-VALUE'])

    result={}
    
    for c, s, t, p in zip(chrid, snpid, trait, pvalue):
        result[c] = (s, t, p)

    return result
#test 
sig_per_chrom = mostSignificantPerChrom(gwas)
print("Most significant SNP per chromosome:")
for chrom in sorted(sig_per_chrom, key = lambda x: int(x)):
    snp, trait, pval = sig_per_chrom[chrom]
    print(f"  chr{chrom}: {snp} | {trait} | p={pval:.2e}")

def publicationTrend(gwas):
    """
    T3.3 — Returns a DataFrame indexed by year with columns ['n_studies', 'n_unique_traits', 'n_unique_snps'], 
    restricted to years with >= 5 studies. Sorted by year.
    """
    # TODO
    pass

"""
T3.3 — Plot n_studies over time
"""
# TODO

def genomicHotspots(gwas):
    """
    T3.4 — Returns the top-5 1-Mbp windows with the most distinct associated traits, as a list of tuples 
    (chr, window_start, window_end, n_traits).
    """
    gwas['window_start']= (gwas['CHR_POS']//1000000)*1000000
    gwas['window_end'] = gwas['window_start']+ 1000000

    result = gwas.groupby(['CHR_ID','window_start','window_end'])
    window = result['DISEASE/TRAIT'].nunique().reset_index()
    window.columns = ['chr','window_start','window_end','n_traits']

    window_sorted = window.sort_values(by=['n_traits', 'chr', 'window_start'], ascending=[False, True, True],key=lambda col: col.astype(int) if col.name == 'chr' else col)

    top5 = window_sorted.head(5)

    top5_tuples = list(top5.itertuples(index=False, name=None))

    return top5_tuples

# Test
#hotspots = genomicHotspots(gwas)
#print("Top 5 genomic hotspots:")
#for chrom, ws, we, nt in hotspots:
    #print(f"  chr{chrom}:{ws:,}–{we:,}  →  {nt} distinct trait(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — NetworkX - Protein–Protein Interaction Network
# ─────────────────────────────────────────────────────────────────────────────

def buildPPINetwork(filepath, min_score = 400):
    """
    T4.1 — Reads the PPI network TSV file and returns an undirected weighted graph (weight = combined_score), 
    containing only interactions with combined_score >= min_score.
    """
    data = pd.read_csv(filepath, sep = '\t')

    G = nx.Graph()
    
    data = data[data['combined_score'] >= min_score]
    
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_edge(row['protein_a'], row['protein_b'], weight=row['combined_score'])
    
    return G

#test
G = buildPPINetwork('data/ppi_network.tsv')
print((G.number_of_nodes()), G.number_of_edges())


def topHubs(graph):
    """
    T4.1 — Returns a list of the 10 proteins with the highest degree, sorted by descending degree.
    """
    hubs = sorted(G.degree(), key=lambda x: x[1], reverse=True)

    return hubs[:10]

#test
hubs = topHubs(G)
print("Top 10 network hubs:")
for protein, deg in hubs:
    print(f"  {protein}: {deg} interaction partners")

def networkComponents(graph):
    """
    T4.2 — Returns (n_components, largest_size, component_sizes_sorted_desc).
    """
    components = list(nx.connected_components(graph))
    sizes = sorted([len(c) for c in components], reverse=True)
    
    return (len(components), sizes[0], sizes)
    

#test 
n_comp, largest, sizes = networkComponents(G)
print(f"Connected components: {n_comp}")
print(f"Largest component: {largest} proteins")
print(f"Component sizes: {sizes}")

def shortestInteractionPath(graph, protein_a, protein_b):
    """
    T4.3 — Returns the list of proteins along the shortest (unweighted) path between protein_a and protein_b. 
    Returns None if no path exists.
    """
    try:
        return nx.shortest_path(graph, source=protein_a, target=protein_b)
    except nx.NetworkXNoPath:
        return None
    except nx.NodeNotFound:
        return None
    
# Test
path = shortestInteractionPath(G, "TP53", "EGFR")
if path:
    print(" → ".join(path))
    print(f"Path length: {len(path)-1} edge(s)")
else:
    print("No path found.")

def breastCancerModule(graph, clinvar_filepath):
    """
    T4.4 — Extracts the breast cancer disease module from the PPI network.
    Returns (subgraph, modularity_score), or (subgraph, None) if too small.
    """
    #Get the breast cancer genes that are pathogenic/likely pathogenic
    with open(clinvar_filepath, 'r') as f:
        data = json.load(f)

    result = data['result']
    uids = result['uids']

    breast_cancer_genes = set()
    for uid in uids:
        variant = result[uid]
        significance = variant['clinical_significance']['description']
        conditions = [c['name'] for c in variant['conditions']]
        genes = [g['symbol'] for g in variant['genes']]

        if significance in ('Pathogenic', 'Likely pathogenic'):
            for cond in conditions:
                if 'breast cancer' in cond.lower():
                    breast_cancer_genes.update(genes)
                    break

    # Build subgraph
    nodes_in_graph = [g for g in breast_cancer_genes if g in graph.nodes]
    subgraph = graph.subgraph(nodes_in_graph).copy()

    # Subgraph meets requirements
    if subgraph.number_of_nodes() < 2 or subgraph.number_of_edges() == 0:
        return (subgraph, None)

    # Detect communities and compute modularity
    communities = list(nx_comm.greedy_modularity_communities(subgraph))
    mod_score = nx_comm.modularity(subgraph, communities)

    return (subgraph, mod_score)

# Test
subG, mod = breastCancerModule(G, 'data/clinvar_variants.json')
print(f"Breast cancer module: {subG.number_of_nodes()} proteins, {subG.number_of_edges()} interactions")
if mod is not None:
    print(f"Modularity score: {mod:.4f}")
else:
    print("Module too small for community detection.")


# Visualise the subgraph
fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(subG, seed=42)
nx.draw_networkx(subG, pos=pos, ax=ax,
                 node_color='salmon', node_size=800,
                 font_size=9, edge_color='gray', width=1.5)
ax.set_title("Breast Cancer Disease Module (PPI Network)", fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('breast_cancer_module.png', dpi=150)
plt.show()