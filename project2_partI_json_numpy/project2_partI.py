"""
Project 2 — Biological Data Analysis

Implement all functions below. Do not change their signatures.
"""

import json
import csv
import math
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — JSON: Genomic Variants (ClinVar)
# ─────────────────────────────────────────────────────────────────────────────

CHROM_LENGTHS = {
    "1":248956422,"2":242193529,"3":198295559,"4":190214555,"5":181538259,
    "6":170805979,"7":159345973,"8":145138636,"9":138394717,"10":133797422,
    "11":135086622,"12":133275309,"13":114364328,"14":107043718,"15":101991189,
    "16":90338345,"17":83257441,"18":80373285,"19":58617616,"20":64444167,
    "21":46709983,"22":50818468
}

def mostPathogenicGene(filepath):
    """T1.1 — Returns (gene_symbol, count) for the gene with the most Pathogenic variants."""
    with open (filepath, 'r') as f:
        data = json.load(f)
        
        result = data['result']
        pathogenic = defaultdict(set)
        
        for uid in result['uids']:
            if uid not in result:
                continue
            
            entry = result[uid]

            outcome = entry['clinical_significance']['description']
            if outcome != 'Pathogenic':
                continue

            for genes in entry ['genes']:
                symbol= genes['symbol']
                if symbol:
                    pathogenic[symbol].add(uid)

        if not pathogenic:
            return (None, 0)
        
        smbl = max(pathogenic, key = lambda g: len(pathogenic[g]))
        return (smbl, len(pathogenic[smbl]))

print(mostPathogenicGene('data/clinvar_variants.json'))

def variantDensityByChrom(filepath):

    with open(filepath, 'r') as f:
        data = json.load(f)
        result = data['result']
        chrm=defaultdict(set)

        for uid in result['uids']:
            if uid not in result:
                continue

            entry = result[uid]

            for var in entry.get('variation_set', []):
                for loc in var.get('variation_loc',[]):
                    chr= loc.get('chr')
                    if chr:
                        chrm[chr].add(uid)
            
            density_var={}
            for chr in chrm:
                density_var[chr] = round(len(chrm[chr])*1000000/CHROM_LENGTHS[chr], 4)
            
            sorted_dict = dict(sorted(density_var.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_dict

print(variantDensityByChrom('data/clinvar_variants.json'))

def topPathogenicConditions(filepath):
    with open (filepath, 'r') as f:
        data = json.load(f)
        result = data ['result']
        conditions = defaultdict(set) 

        for uid in result['uids']:
            entry = result[uid]
            outcome = entry['clinical_significance']['description']

            if outcome not in ('Pathogenic', 'Likely pathogenic'):
                continue

            for con in entry.get('conditions', []): 
                name = con.get('name')
                if name:
                    conditions[name].add(uid)
            
            top = []

            for c in conditions:
                if len(conditions[c])<6:
                    continue
            
                top.append((c, len(conditions[c])))

        sorted_top= sorted(top, key= lambda x: x[1], reverse=True)           
        return sorted_top

print(topPathogenicConditions('data/clinvar_variants.json'))

def variantIndex(filepath):
    """T1.4 — Returns nested dict {gene: {chr: {clinical_significance: [titles]}}}."""
    nested = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(list)
        )
    )

    with open(filepath, "r") as f:
        data = json.load(f)

    result = data["result"]

    for uid in result["uids"]:
        if uid not in result:
            continue

        entry = result[uid]

        title = entry.get("title")
        clinical = entry.get("clinical_significance", {}).get("description")

        if not title or not clinical:
            continue

        genes = entry.get("genes", [])
        variation_set = entry.get("variation_set", [])

        for gene in genes:
            symbol = gene.get("symbol")
            if not symbol:
                continue

            for var in variation_set:
                for loc in var.get("variation_loc", []):
                    chrom = loc.get("chr")
                    if not chrom:
                        continue

                    nested[symbol][chrom][clinical].append(title)

    return nested

#print(variantIndex("data/clinvar_variants.json"))
idx = variantIndex('data/clinvar_variants.json')
print(len(idx['BRCA2']['13']['Pathogenic']))


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — NumPy: Gene Expression Analysis (TCGA)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib.pyplot as plt

# Read expression matrix
import csv

with open('data/expression_matrix.csv', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    sample_ids = header[1:]            # list of sample IDs
    gene_ids   = []
    expr_rows  = []
    for row in reader:
        gene_ids.append(row[0])
        expr_rows.append([float(x) for x in row[1:]])

expr_matrix = np.array(expr_rows)     # shape: (n_genes, n_samples)
gene_ids    = np.array(gene_ids)
sample_ids  = np.array(sample_ids)

# Read sample metadata
import pandas as pd
metadata = pd.read_csv('data/sample_metadata.csv')

# Boolean masks for tumor and normal samples
tumor_mask  = np.array([s in metadata[metadata['condition']=='tumor']['sample_id'].values
                         for s in sample_ids])
normal_mask = np.array([s in metadata[metadata['condition']=='normal']['sample_id'].values
                         for s in sample_ids])

print(f"Expression matrix shape: {expr_matrix.shape}")
print(f"Tumor samples: {tumor_mask.sum()} | Normal samples: {normal_mask.sum()}")

def geneExpressionStats(expr_matrix):
    """T2.1 — Returns (means, stds): two 1D arrays of shape (n_genes,)."""
    
    mean= np.mean(expr_matrix, axis=1)
    strd= np.std(expr_matrix, axis=1)
    
    return (mean, strd)

# Test T2.1

means, stds = geneExpressionStats(expr_matrix)
print("Top 5 most variable genes (highest std):")
top_var_idx = np.argsort(stds)[::-1][:5]
for i in top_var_idx:
    print(f"  {gene_ids[i]}: mean={means[i]:.3f}, std={stds[i]:.3f}")

# T2.2 - Top Differentially Expressed Genes

def topDifferentialGenes(expr_matrix, gene_ids, tumor_mask, normal_mask):
    """T2.2 — Returns list of 10 tuples (gene_id, delta_mean) sorted by descending |delta_mean|."""
    normal_means = np.mean(expr_matrix[:, normal_mask], axis=1)
    tumor_means = np.mean(expr_matrix[:, tumor_mask], axis=1)
    
    delta_means = tumor_means - normal_means

    top10_ids = np.argsort(np.abs(delta_means))[::-1][:10]
    
    return [(gene_ids[i], round(delta_means[i], 3)) for i in top10_ids]

# Test T2.2

top10 = topDifferentialGenes(expr_matrix, gene_ids, tumor_mask, normal_mask)
print("Top 10 differentially expressed genes:")
#for gene, delta in top10:
    #direction = "▲ overexpressed" if delta > 0 else "▼ underexpressed"
    #print(f"  {gene}: Δmean = {delta:+.3f}  ({direction} in tumor)")

# T2.3 - Sample Correlation Matrix

def sampleCorrelationMatrix(expr_matrix):
    """T2.3 — Returns Pearson correlation matrix of shape (n_samples, n_samples). NumPy only."""
    gene_means = np.mean(expr_matrix, axis=1, keepdims=True)
    centered = expr_matrix - gene_means 
    
    cov_matrix = np.dot(centered.T, centered) / (centered.shape[0] - 1)
    
    stds = np.sqrt(np.diag(cov_matrix))
    
    corr_matrix = cov_matrix / np.outer(stds, stds)
    
    return corr_matrix

df = pd.read_csv('data/expression_matrix.csv', index_col=0)
expr_matrix = df.to_numpy(dtype=float)
corr = sampleCorrelationMatrix(expr_matrix)

# Quick tests on the correlation matrix

print(f"Correlation matrix shape: {corr.shape}")
print(f"Min correlation: {corr.min():.4f} | Max (off-diagonal): {corr[corr < 0.9999].max():.4f}")

# Plotting the correlation matrix as a heatmap

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_title("Sample Correlation Matrix")
ax.set_xlabel("Samples")
ax.set_ylabel("Samples")

n_tumor = tumor_mask.sum()
ax.axhline(n_tumor - 0.5, color='black', linewidth=2)
ax.axvline(n_tumor - 0.5, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('sample_correlation_heatmap.png', dpi=150)
plt.show()

# T2.4 - Gene Classification by z-score

def zscoreNormalize(expr_matrix):
    """T2.4a — Returns z-score normalised matrix (per gene, across all samples)."""
    gene_means = np.mean(expr_matrix, axis=1, keepdims=True)
    gene_stds = np.std(expr_matrix, axis=1, keepdims=True)

    z_matrix = (expr_matrix - gene_means) / gene_stds
    
    return z_matrix

def classifyGenes(z_matrix, gene_ids, tumor_mask):
    """T2.4b — Returns {gene_id: 'overexpressed'|'underexpressed'|'stable'}."""
    tumor_means = np.mean(z_matrix[:, tumor_mask], axis=1) 
    
    classif = {}
    for i, gene in enumerate(gene_ids):
        if tumor_means[i] > 0.5:
            classif[gene] = 'overexpressed'
        elif tumor_means[i] < -0.5:
            classif[gene] = 'underexpressed'
        else:
            classif[gene] = 'stable'
    
    return classif

z = zscoreNormalize(expr_matrix)
classes = classifyGenes(z, gene_ids, tumor_mask)
print(classes['ENSG00000141510'])
