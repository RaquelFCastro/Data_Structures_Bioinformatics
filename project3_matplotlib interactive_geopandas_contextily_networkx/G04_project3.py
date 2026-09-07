"""
Project 3 — Biological Data Visualization

Implement all functions below. Do not change their signatures.
"""

# ─────────────────────────────────────────────────────────────────────────────
# ADD Packages
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.colors import ListedColormap
from matplotlib.widgets import RadioButtons, CheckButtons
import networkx as nx
import matplotlib.patches as mpatches
from collections import defaultdict

# ------------------------------------------------------------------------------
# Read expression matrix and samples
expr = pd.read_csv('data/expression_matrix.csv', index_col='gene_id')
metadata = pd.read_csv('data/sample_metadata.csv', index_col='sample_id')

# Align samples
samples = expr.columns.intersection(metadata.index)
expr = expr[samples]
metadata = metadata.loc[samples]

tumor_samples  = metadata[metadata['condition'] == 'tumor'].index
normal_samples = metadata[metadata['condition'] == 'normal'].index

# Per-gene statistics
mean_tumor  = expr[tumor_samples].mean(axis=1)
mean_normal = expr[normal_samples].mean(axis=1)
log2fc      = mean_tumor - mean_normal   # log2 fold-change (already in log2 space)

# Per-gene p-value (Welch t-test, one per gene)
pvalues = expr.apply(
    lambda row: stats.ttest_ind(
        row[tumor_samples], row[normal_samples], equal_var=False
    ).pvalue,
    axis=1
)
neg_log10_p = -np.log10(pvalues.clip(lower=1e-300))

print(f"Genes: {expr.shape[0]}  |  Tumor samples: {len(tumor_samples)}  |  Normal samples: {len(normal_samples)}")

# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Static Charts: Differential Gene Expression (TCGA)
# ─────────────────────────────────────────────────────────────────────────────

#T1.1
def volcanoPlot(ax, log2fc, neg_log10_p):
    # Assigning the x and y values to the correct variables
    x = log2fc
    y = neg_log10_p

    # Creating a mask with the established thresholds
    mask_plus = (log2fc>1) & (neg_log10_p>2)
    mask_minus = (log2fc<-1) & (neg_log10_p>2)
    mask_notsig = ((log2fc>-1) & (log2fc<1) & (neg_log10_p<2))

    # Plotting the volcano plot
    ax.scatter(x[mask_plus], y[mask_plus], color= 'red', alpha=0.5)
    ax.scatter(x[mask_minus], y[mask_minus], color = 'blue', alpha= 0.5)
    ax.scatter(x[mask_notsig], y[mask_notsig], color = 'grey', alpha= 0.5)
    ax.axvline(1, color= 'black', linestyle = '--', linewidth= 1)
    ax.axvline(-1, color= 'black', linestyle = '--', linewidth= 1)
    ax.axhline(2, color= 'black', linestyle = '--', linewidth=1)
    ax.set_xlabel('log2fc')
    ax.set_ylabel('neg_log10_p')
    ax.set_title('Volcano Plot')

    # Annotate the plot for the five most significant genes (neg_log10_p)
    top5 = np.argsort(y)[-5:]
    for i in top5:
        ax.text(log2fc[i], neg_log10_p[i], neg_log10_p.index[i], size=7)

    # Summary text
    up= x[mask_plus]
    down= x[mask_minus]
    ax.text(-3.8, 13, f'Genes Overexpressed: {len(up)} \nGenes Underexpressed: {len(down)}', size=9)

# Test
fig, ax = plt.subplots(figsize=(8, 6))
volcanoPlot(ax, log2fc, neg_log10_p)
plt.tight_layout()
plt.savefig('volcano_plot.png', dpi=150)
plt.show()

#T1.2
def expressionHeatmap(ax, expr, metadata):
    """T1.2 — Draws a heatmap of the top 20 most differentially expressed genes.
    Samples ordered: tumour (by cancer type) then normal.
    Genes ordered by log2FC (most overexpressed at top).
    Z-score normalised per gene, diverging colourmap (RdBu_r).
    """
    tumor_samples  = metadata[metadata['condition'] == 'tumor']
    normal_samples = metadata[metadata['condition'] == 'normal']

    mean_tumor  = expr[tumor_samples].mean(axis=1)
    mean_normal = expr[normal_samples].mean(axis=1)
    log2fc      = mean_tumor - mean_normal

    abs_fc= log2fc.abs().sort_values(ascending=False).head(20).index
    ordered_genes = log2fc.loc[abs_fc].sort_values(ascending=False).index

    #ordering the metadata (tumour + normal)
    tumor_ordered = tumor_samples.sort_values(by='cancer_type')

    normal_ordered = normal_samples.sort_values(by='cancer_type')

    ordered_samples = (list(tumor_ordered.index) +list(normal_ordered.index))
    
    #mask for the heatmap
    heatmap = expr.loc[ordered_genes, ordered_samples]

    #z-score normalization
    z_score_norm= heatmap.sub(heatmap.mean(axis=1), axis=0).div(heatmap.std(axis=1), axis=0)

    #tumour vs normal annotation
    condition_bar = np.array([
        1 if s in tumor_samples.index else 0
        for s in ordered_samples
    ]).reshape(1, -1)

    # create small axis above heatmap
    top_ax = ax.inset_axes([0, 1.02, 1, 0.04])

    top_ax.imshow(
        condition_bar,
        aspect='auto',
        cmap=ListedColormap(['steelblue', 'firebrick'])
    )

    top_ax.set_xticks([])
    top_ax.set_yticks([])

    n_tumor = len(tumor_samples)
    n_normal = len(normal_samples)

    top_ax.text(
        n_tumor / 2,
        0.1,
        'Tumour',
        ha='center',
        va='center',
        color='white',
        fontsize=9,
        fontweight='bold'
    )

    top_ax.text(
        n_tumor + n_normal / 2,
        0.1,
        'Normal',
        ha='center',
        va='center',
        color='white',
        fontsize=9,
        fontweight='bold'
    )

    #draw heatmap
    sns.heatmap(
            z_score_norm,
            ax=ax,
            cmap='RdBu_r',
            center=0,
            cbar=True,
            xticklabels=False,
            yticklabels=True
        )

    ax.set_title('Top 20 Differentially Expressed Genes')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Genes')

#test
fig, ax = plt.subplots(figsize=(12, 7))
expressionHeatmap(ax, expr, metadata)
plt.tight_layout()
plt.savefig('expression_heatmap.png', dpi=150)
plt.show()

#T1.3
def expressionSummaryFigure(expr, metadata):
    """T1.3 (Bonus) — Combined figure: volcano plot (left) + heatmap (right).
    Saves to 'expression_summary.png'.
    """
    fig, axes = plt.subplots(
    1, 2,
    figsize=(18, 7),
    gridspec_kw={'width_ratios': [1, 1.2]}
)

    # left = volcano
    volcanoPlot(axes[0], log2fc, neg_log10_p)

    # right = heatmap
    expressionHeatmap(axes[1], expr, metadata)

    plt.tight_layout()

    plt.savefig('expression_summary.png', dpi=150, bbox_inches='tight')

    plt.show()

# Test
expressionSummaryFigure(expr, metadata)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Dynamic Charts: Genomic Variant Surveillance
# ─────────────────────────────────────────────────────────────────────────────

snps = pd.read_csv('data/snp_surveillance.csv')
populations = sorted(snps['population'].unique())
chromosomes = sorted(snps['chromosome'].unique(), key=lambda x: (int(x) if x.isdigit() else 100, x))
consequences = sorted(snps['consequence'].unique())
years = sorted(snps['year'].unique())

print(snps.head())
print(f"\n{len(snps)} records | {snps['snp_id'].nunique()} unique SNPs | {len(populations)} populations")

#T2.1
def drawPopulationTrend(ax, chromosome, consequence):

    snps = pd.read_csv('data/snp_surveillance.csv')
    subset = snps[(snps['chromosome'].astype(str) == str(chromosome)) &
                  (snps['consequence'] == consequence)]

    #handles no snp reports for that chromossome and consequence
    if subset.empty:
        ax.text(0.5, 0.5, 'No data for this selection.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Chr {chromosome} | {consequence}')
        return
    
    #get the mean of snps allele frequency
    trend = (subset.groupby(['year', 'population'])['allele_frequency']
             .mean().reset_index())
    
    #create two dictionaries for the colors and markers of the 5 different population subsets
    colors = {
    "AFR": "tab:blue",
    "EUR": "tab:orange",
    "EAS": "tab:green",
    "SAS": "tab:red",
    "AMR": "tab:purple",
    }

    markers = {
    "AFR": "o",
    "EUR": "s",
    "EAS": "^",
    "SAS": "D",
    "AMR": "v",
    }
    
    #for every subset of the population, plot the corresponding mean allele frequency for that year
    for pop in sorted(trend['population'].unique()):
        d = trend[trend['population'] == pop].sort_values('year')
        ax.plot(d['year'], d['allele_frequency'],
                label=pop,
                color=colors.get(pop, None),
                marker=markers.get(pop, 'o'),
                linewidth=1.8, markersize=5, alpha=0.5)
 
    
    n_snps = subset['snp_id'].nunique()
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Mean Allele Frequency', fontsize=11)
    ax.set_title(f'Allele Frequency Trend — Chr {chromosome} | {consequence}\n'
                 f'({n_snps} SNPs)', fontsize=11, fontweight='bold')
    ax.legend(title='Population', fontsize=9, loc='best')
    ax.set_ylim(0, 1)

# Test
fig, ax = plt.subplots(figsize=(9, 5))
drawPopulationTrend(ax, '17', 'missense')
plt.tight_layout()
plt.savefig('population_trend.png', dpi=150)
plt.show()

#T2.2
def drawSurveillanceDashboard():

    current_chr = chromosomes[0]
    current_cons = consequences[0]

    fig = plt.figure(figsize=(12, 7))

    # Main trend plot
    ax_trend = fig.add_axes([0.25, 0.30, 0.70, 0.60])

    # Distribution plot
    ax_dist = fig.add_axes([0.25, 0.08, 0.70, 0.15])

    # Radio buttons
    ax_chr = fig.add_axes([0.03, 0.40, 0.15, 0.45])
    radio_chr = RadioButtons(ax_chr, chromosomes)

    ax_cons = fig.add_axes([0.03, 0.10, 0.15, 0.20])
    radio_cons = RadioButtons(ax_cons, consequences)

    def update():
        ax_trend.clear()
        ax_dist.clear()

        # Main trend plot (T2.1)
        drawPopulationTrend(ax_trend, current_chr, current_cons)

        # Data for current selection
        subset = snps[
            (snps["chromosome"].astype(str) == current_chr) &
            (snps["consequence"] == current_cons)
        ]

        if len(subset) == 0:
            ax_dist.text(
                0.5, 0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax_dist.transAxes
            )
        else:
            # Histogram
            ax_dist.hist(
                subset["allele_frequency"],
                bins=15
            )
            ax_dist.set_title("Allele Frequency Distribution")
            ax_dist.set_xlabel("Allele Frequency")
            ax_dist.set_ylabel("Count")

            # Summary statistics
            stats_text = (
                f"SNPs: {len(subset)}\n"
                f"Mean AF: {subset['allele_frequency'].mean():.4f}"
            )

            ax_trend.text(
                0.02, 0.98,
                stats_text,
                transform=ax_trend.transAxes,
                va="top",
                bbox=dict(facecolor="white", alpha=0.8)
            )

        fig.canvas.draw_idle()

    def chromosome_changed(label):
        nonlocal current_chr
        current_chr = label
        update()

    def consequence_changed(label):
        nonlocal current_cons
        current_cons = label
        update()

    radio_chr.on_clicked(chromosome_changed)
    radio_cons.on_clicked(consequence_changed)

    update()
    plt.show()

# test
drawSurveillanceDashboard()


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Maps: Global Species Occurrence (GBIF)
# ─────────────────────────────────────────────────────────────────────────────

occurrences = pd.read_csv('data/species_occurrences.csv').dropna(subset=['decimalLatitude','decimalLongitude'])
world = gpd.read_file('data/countries.geojson')  # ISO3166-1-Alpha-3 column

species_list = sorted(occurrences['species'].unique())
print(f"{len(occurrences)} occurrence records | {len(species_list)} species")
print(occurrences.head())

#T3.1
def drawSpeciesRichness(ax, occurrences, world):
    # Count distinct species per country
    richness = occurrences.groupby('country')['species'].nunique().reset_index()
    richness.columns = ['country', 'richness']

    # Merge with world GeoDataFrame
    world_richness = world.merge(richness, left_on='ISO3166-1-Alpha-3', right_on='country', how='left')

    # Reproject to Web Mercator for contextily compatibility
    world_richness = world_richness[world_richness['ISO3166-1-Alpha-3'] != 'ATA'].to_crs(epsg=3857)

    # Plot
    world_richness.plot(
        column='richness', 
        ax=ax, 
        cmap='YlOrRd',
        legend=True,
        alpha=0.7, 
        legend_kwds={'label': 'Species Richness', 'orientation': 'vertical', 'shrink': 0.5},
        missing_kwds={'color': 'lightgrey', 'label': 'No data'}
    )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    ax.set_title('Species Richness by Country', fontsize=14)
    ax.set_axis_off()

# Test
fig, ax = plt.subplots(figsize=(14, 7))
drawSpeciesRichness(ax, occurrences, world)
plt.tight_layout()
plt.savefig('species_richness.png', dpi=150)
plt.show()

#T3.2
def drawSpeciesOccurrences(ax, occurrences, species_name):
    
    # Filter for the species
    occur_filt = occurrences[occurrences['species'] == species_name].copy()

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        occur_filt,
        geometry=gpd.points_from_xy(occur_filt['decimalLongitude'], occur_filt['decimalLatitude']),
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    # Normalize year so more recent translates to a larger point (point size encodes recency)
    year_min, year_max = occur_filt['year'].min(), occur_filt['year'].max()
    if year_max > year_min:
        gdf['size'] = 10 + 40 * (occur_filt['year'] - year_min) / (year_max - year_min)
    else:
        gdf['size'] = 20

    # Color by basisOfRecord
    basis_types = gdf['basisOfRecord'].unique()
    palette = plt.cm.tab10.colors
    colour_map = {b: palette[i % len(palette)] for i, b in enumerate(basis_types)}
    gdf['colour'] = gdf['basisOfRecord'].map(colour_map)

    # Set map extent to bounding box of occurrences + margin
    margin = 500000
    minx, miny, maxx, maxy = gdf.total_bounds
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

    # Plot points per basisOfRecord category to get a legend
    for basis, colour in colour_map.items():
        subset = gdf[gdf['basisOfRecord'] == basis]
        ax.scatter(
            subset.geometry.x, subset.geometry.y,
            c=[colour], s=subset['size'],
            alpha=0.7, linewidths=0.3, edgecolors='k',
            label=basis
        )

    ax.set_title(f'Occurrences of {species_name}', fontsize=13)
    ax.set_axis_off()
    ax.legend(title='Basis of Record', loc='lower right', fontsize=8, title_fontsize=9)
    ax.annotate('Data source: GBIF', xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=8, color='grey')
    
#Test
fig, ax = plt.subplots(figsize=(10, 8))
drawSpeciesOccurrences(ax, occurrences, 'Panthera leo')
plt.tight_layout()
plt.savefig('species_occurrences_map.png', dpi=150)
plt.show()

#T3.3
def drawTemporalSpread(occurrences, species_name):
    # Filter species and drop missing years/coords
    df = occurrences[occurrences['species'] == species_name].copy()
    df = df.dropna(subset=['year', 'decimalLatitude', 'decimalLongitude'])
    df['decade'] = (df['year'] // 10 * 10).astype(int)

    decades = sorted(df['decade'].unique())
    n = len(decades)
    ncols = 4
    nrows = (n + ncols - 1) // ncols 

    # Convert full species set to GeoDataFrame for bounds
    gdf_all = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['decimalLongitude'], df['decimalLatitude']),
        crs='EPSG:4326'
    ).to_crs(epsg=3857)

    # Global bounding box to ensure consistent extent across panels
    margin = 500000 
    minx, miny, maxx, maxy = gdf_all.total_bounds

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, decade in enumerate(decades):
        ax = axes[i]
        subset = gdf_all[gdf_all['decade'] == decade]

        # Set consistent extent
        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

        # Basemap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=3)

        # Plot points
        ax.scatter(
            subset.geometry.x, subset.geometry.y,
            s=15, color='crimson', alpha=0.6,
            linewidths=0.2, edgecolors='darkred'
        )

        ax.set_title(f'{decade}s  (n={len(subset)})', fontsize=10, fontweight='bold')
        ax.set_axis_off()

    # Hide any unused panels
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Temporal Spread of {species_name} Occurrences by Decade',
                 fontsize=14, y=1.01)
    fig.text(0.5, -0.01, 'Data source: GBIF', ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig('temporal_spread.png', dpi=150, bbox_inches='tight')
    plt.show()

# Test
drawTemporalSpread(occurrences, 'Panthera leo')


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Network Graphs: Metabolic Pathways
# ─────────────────────────────────────────────────────────────────────────────

metabolites = pd.read_csv('data/metabolites.csv', index_col='metabolite_id')
reactions   = pd.read_csv('data/reactions.csv')

#T4.1
def buildMetabolicGraph(metabolites, reactions):
    G = nx.DiGraph()

    # Add metabolite nodes
    for metabolite_id, row in metabolites.iterrows():
        G.add_node(
            metabolite_id,
            name=row["name"],
            pathway=row["pathway"],
            formula=row["formula"]
        )

    # Add reaction edges
    for _, row in reactions.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            reaction_id=row["reaction_id"],
            enzyme=row["enzyme"],
            reversible=row["reversible"],
            delta_G=row["delta_G"]
        )

        # Add reverse edge if reaction is reversible
        if row["reversible"]:
            G.add_edge(
                row["target"],
                row["source"],
                reaction_id=row["reaction_id"],
                enzyme=row["enzyme"],
                reversible=row["reversible"],
                delta_G=row["delta_G"]
            )

    return G

#test
G = buildMetabolicGraph(metabolites, reactions)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

#T4.2
def drawMetabolicNetwork(ax, G, metabolites):
    # Layout
    pos = nx.kamada_kawai_layout(G)

    # Pathway colours
    pathway_colors = {
        "Glycolysis": "tab:blue",
        "TCA": "tab:green",
        "PPP": "tab:orange",
        "Shared": "tab:purple"
    }

    # Node colours
    node_colors = [
        pathway_colors[G.nodes[n]["pathway"]]
        for n in G.nodes()
    ]

    # Degree-scaled node sizes
    node_sizes = [
        300 + 200 * G.degree(n)
        for n in G.nodes()
    ]

    # Labels use metabolite names
    labels = {
        n: G.nodes[n]["name"]
        for n in G.nodes()
    }

    # Separate edges by style and ΔG sign
    solid_edges_neg = []
    solid_edges_pos = []
    dashed_edges_neg = []
    dashed_edges_pos = []

    for u, v, data in G.edges(data=True):

        reversible = data["reversible"]
        favourable = data["delta_G"] < 0

        if reversible:
            if favourable:
                dashed_edges_neg.append((u, v))
            else:
                dashed_edges_pos.append((u, v))
        else:
            if favourable:
                solid_edges_neg.append((u, v))
            else:
                solid_edges_pos.append((u, v))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=solid_edges_neg,
        edge_color="green",
        style="solid",
        arrows=True
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=solid_edges_pos,
        edge_color="red",
        style="solid",
        arrows=True
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=dashed_edges_neg,
        edge_color="green",
        style="dashed",
        arrows=True
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edgelist=dashed_edges_pos,
        edge_color="red",
        style="dashed",
        arrows=True
    )

    # Labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        ax=ax,
        font_size=8
    )

    # Legends
    pathway_legend = [
        mpatches.Patch(color=color, label=pathway)
        for pathway, color in pathway_colors.items()
    ]

    dg_legend = [
        mpatches.Patch(color="green", label="ΔG < 0 (favourable)"),
        mpatches.Patch(color="red", label="ΔG > 0 (unfavourable)")
    ]

    legend1 = ax.legend(
        handles=pathway_legend,
        title="Pathway",
        loc="upper left"
    )

    ax.add_artist(legend1)

    ax.legend(
        handles=dg_legend,
        title="Reaction ΔG",
        loc="upper right"
    )

    ax.set_title("Metabolic Network")
    ax.axis("off")

#test
fig, ax = plt.subplots(figsize=(14, 10))
drawMetabolicNetwork(ax, G, metabolites)
plt.tight_layout()
plt.savefig('metabolic_network.png', dpi=150)
plt.show()

#T4.3a
def centralMetabolites(G, metabolites):
    # Compute betweenness centrality
    bc = nx.betweenness_centrality(G)

    # Sort by descending centrality
    top5 = sorted(
        bc.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Convert IDs to metabolite names
    return [
        (metabolites.loc[node_id, "name"], centrality)
        for node_id, centrality in top5
    ]

#test
print("Top-5 central metabolites:")
for name, bc in centralMetabolites(G, metabolites):
    print(f"  {name}: {bc:.4f}")

#T4.3b
def findShortestPathway(G, metabolites, source_id, target_id):
    try:
        path_ids = nx.shortest_path(
            G,
            source=source_id,
            target=target_id
        )

        return [
            metabolites.loc[mid, "name"]
            for mid in path_ids
        ]

    except nx.NetworkXNoPath:
        return None

    except nx.NodeNotFound:
        return None
    
#test
# Test
path = findShortestPathway(G, metabolites, 'C00031', 'C00022')  # Glucose → Pyruvate
print(f"Glucose → Pyruvate: {' → '.join(path) if path else 'No path'}")

#T4.3c
def pathwayConnectivity(G, metabolites):
    connectivity = defaultdict(int)

    for u, v in G.edges():

        source_pathway = metabolites.loc[u, "pathway"]
        target_pathway = metabolites.loc[v, "pathway"]

        # Ignore Shared metabolites
        if source_pathway == "Shared" or target_pathway == "Shared":
            continue

        # Ignore edges within same pathway
        if source_pathway == target_pathway:
            continue

        pair = frozenset(
            [source_pathway, target_pathway]
        )

        connectivity[pair] += 1

    return dict(connectivity)

#test
print("Cross-pathway edges:")
for pair, n in pathwayConnectivity(G, metabolites).items():
    print(f"  {' ↔ '.join(pair)}: {n} edge(s)")
