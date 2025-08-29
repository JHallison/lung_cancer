import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import gseapy as gp
from tqdm import tqdm


def parse_rna(gene_counts):
    gene_counts = gene_counts.drop(columns=['Entrez_Gene_Id'])
    gene_counts.index = gene_counts['Hugo_Symbol']
    gene_counts = gene_counts.drop(columns=['Hugo_Symbol'])
    gene_counts = gene_counts.T
    gene_counts.index = [id + 'A' for id in gene_counts.index]
    gene_counts = gene_counts.loc[:, ~gene_counts.columns.duplicated()]

    return gene_counts


def parse_metadata(metadata_filepath, metadata_label_string):
    whole_cohort_meta = pd.read_csv(metadata_filepath)
    whole_cohort_meta = whole_cohort_meta.filter(items=['submitter_id.samples', metadata_label_string])
    whole_cohort_meta.index = whole_cohort_meta['submitter_id.samples']
    whole_cohort_meta = whole_cohort_meta.drop(columns=['submitter_id.samples'])
    return whole_cohort_meta


# Class for DESeq2 analysis

class Deseq2AnalysisClass:

    def __init__(self, gene_counts: pd.DataFrame, metadata: pd.DataFrame, save_path: str = None):

        """
        Constructs the appropriate dataframe to carry out DESeq2 analysis and create plots
        """

        self.gene_counts = self.construct_dataframe(gene_counts)
        self.metadata = metadata
        self.deseq_results = None
        self.save_path = save_path

    def construct_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Prepare the RNA counts DataFrame by rounding and converting to integers.
        """
        df = df.round()
        return df.astype(int)

    def perform_deseq2(self, metadata_column: str, group_1: str, group_2: str) -> pd.DataFrame:

        """
        Perform DESeq2 differential expression analysis.

        log2 fold change is calculated as log2(group_2/group_1)

        log2fc > 0 ---> upgregulated in group 2
        log2fc < 0 ---> downregulated in group 2

        """

        merged_df = self.gene_counts.merge(self.metadata, left_index=True, right_index=True, how='inner')
        count_data = merged_df.drop(columns=[metadata_column])
        metadata = merged_df[[metadata_column]]

        dds = DeseqDataSet(
            counts=count_data,
            metadata=metadata,
            design_factors=metadata_column,
            refit_cooks=True,
        )
        dds.deseq2()

        stat_res = DeseqStats(dds, contrast=[metadata_column, group_1, group_2])
        stat_res.summary()

        results_df = stat_res.results_df.dropna(subset=['padj']).sort_values(by='padj')
        self.deseq_results = results_df

        if self.save_path:
            results_df.to_csv(f"{self.save_path}/deseq2_results_{metadata_column}.csv")

        return results_df

    def volcano_plot(self, top_n: int = 20):

        """
        Generate a volcano plot from DESeq2 results.
        """

        if self.deseq_results is None:
            raise ValueError("You must run perform_deseq2() before plotting.")

        df = self.deseq_results.copy()
        df['Gene_name'] = df.index

        df['significance'] = 'Not Significant'
        df.loc[(df['padj'] < 0.05) & (df['log2FoldChange'] > 1), 'significance'] = 'Upregulated'
        df.loc[(df['padj'] < 0.05) & (df['log2FoldChange'] < -1), 'significance'] = 'Downregulated'

        top_genes = df.dropna(subset=['padj', 'Gene_name']).nsmallest(top_n, 'padj')

        volc_fig = plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x='log2FoldChange',
            y=-np.log10(df['padj']),
            hue='significance',
            palette={'Not Significant': 'gray', 'Upregulated': 'red', 'Downregulated': 'blue'},
            alpha=0.7,
            edgecolor=None
        )

        plt.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
        plt.axvline(1, color='red', linestyle='--', linewidth=1)
        plt.axvline(-1, color='blue', linestyle='--', linewidth=1)

        texts = []
        for _, row in top_genes.iterrows():
            text = plt.text(
                row['log2FoldChange'],
                -np.log10(row['padj']),
                row['Gene_name'],
                fontsize=8,
                ha='right' if row['log2FoldChange'] < 0 else 'left',
                va='bottom',
                color='black'
            )
            texts.append(text)
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        plt.title("Volcano plot with top {} most significant genes for late stage patients who survived".format(top_n))
        plt.xlabel('log2(Fold Change)')
        plt.ylabel('-log10(Adjusted p-value)')
        plt.legend(title='Gene Regulation', loc='upper right')
        plt.xlim(-12, 12)
        plt.tight_layout()

        return volc_fig

    def plot_pca(self, metadata_column: str):

        """
        Generate a PCA plot colored by a metadata column (e.g. EGFR mutation status").
        """

        # Scale and apply PCA
        X = self.gene_counts.values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        # Construct PCA DataFrame
        pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=self.gene_counts.index)
        pca_df[metadata_column] = self.metadata[metadata_column]

        # Plot
        pca_fig = plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x="PC1", y="PC2",
            hue=metadata_column,
            data=pca_df,
            palette="Set1",
            s=80,
            edgecolor="k"
        )
        plt.title(f"PCA of Samples Colored by {metadata_column}")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        plt.legend(title=metadata_column)
        plt.tight_layout()

        return pca_fig

    def plot_umap_colored(self, metadata_column: str):

        """
        UMAP of samples colored by metadata, using all genes.
        """

        X = self.gene_counts  # rows = samples and genes = columns

        # Fix: make all column names strings
        X.columns = X.columns.astype(str)

        # Match samples between gene counts and metadata
        common_samples = X.index.intersection(self.metadata.index)
        if common_samples.empty:
            raise ValueError("No overlapping sample indices between gene_counts and metadata.")

        X = X.loc[common_samples]
        metadata = self.metadata.loc[common_samples, metadata_column]

        # Scale all genes, no filtering
        X_scaled = StandardScaler().fit_transform(X)

        # UMAP
        embedding = umap.UMAP(random_state=42).fit_transform(X_scaled)

        # Prepare DataFrame for plotting
        plot_df = pd.DataFrame({
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            metadata_column: metadata.values
        }, index=common_samples)

        # Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=plot_df,
            x="UMAP1", y="UMAP2",
            hue=metadata_column,
            palette="Set1",
            s=80,
            alpha=0.85
        )

        plt.title(f"UMAP Colored by {metadata_column} ({len(common_samples)} Samples)")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend(title=metadata_column)
        plt.tight_layout()
        return plt.gcf()

from scipy.stats import chi2_contingency, mannwhitneyu


class dna_mutation_class:

    """
    A class to analyze co-occurring mutations with EGFR in cancer patient DNA samples.
    """

    def __init__(self, metadata: pd.DataFrame, dna_mutations: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize the EGFR mutation analysis object.

        Parameters:
        - metadata: DataFrame with EGFR-positive sample info. Must contain 'Sample_ID'.
        - dna_mutations: DataFrame with all DNA mutations. Must contain columns:
            'Sample_ID', 'gene', 'dna_vaf', etc.
        """

        self.metadata = metadata
        self.dna_mutations = dna_mutations
        self.egfr_samples = set(metadata.index.unique())
        self.results = pd.DataFrame()

    def compute_stats(self):

        """
        Computes co-occurrence statistics for each gene:
        - Chi-square test: presence/absence in EGFR+ vs EGFR− samples.
        - Mann–Whitney U test: distribution of 'dna_vaf' in EGFR+ vs EGFR− samples.
        Stores results in self.results DataFrame.
        """

        all_samples = set(self.dna_mutations.index.unique())
        egfr_positive = self.egfr_samples
        egfr_negative = all_samples - egfr_positive

        results = []

        all_genes = self.dna_mutations["gene"].unique()
        for gene in all_genes:
            if gene == "EGFR":
                continue  # Skip EGFR itself

            gene_data = self.dna_mutations[self.dna_mutations["gene"] == gene]

            gene_samples = set(gene_data["Sample_ID"])
            egfr_pos_with_gene = len(gene_samples & egfr_positive)
            egfr_pos_without_gene = len(egfr_positive) - egfr_pos_with_gene
            egfr_neg_with_gene = len(gene_samples & egfr_negative)
            egfr_neg_without_gene = len(egfr_negative) - egfr_neg_with_gene

            # Build 2x2 contingency table
            table = [
                [egfr_pos_with_gene, egfr_pos_without_gene],
                [egfr_neg_with_gene, egfr_neg_without_gene]
            ]
            chi2, chi2_p, _, _ = chi2_contingency(table)

            # MWU test (based on VAF values in gene_data)
            vaf_egfr_pos = gene_data[gene_data["Sample_ID"].isin(egfr_positive)]["dna_vaf"]
            vaf_egfr_neg = gene_data[gene_data["Sample_ID"].isin(egfr_negative)]["dna_vaf"]
            if len(vaf_egfr_pos) > 0 and len(vaf_egfr_neg) > 0:
                mwu_stat, mwu_p = mannwhitneyu(vaf_egfr_pos, vaf_egfr_neg, alternative="two-sided")
            else:
                mwu_p = 1.0  # Non-informative

            results.append({
                "gene": gene,
                "chi2_p": chi2_p,
                "chi2_neglogp": -np.log10(chi2_p) if chi2_p > 0 else np.nan,
                "mwu_p": mwu_p,
                "mwu_neglogp": -np.log10(mwu_p) if mwu_p > 0 else np.nan,
                "EGFR+_with_gene": egfr_pos_with_gene,
                "EGFR-_with_gene": egfr_neg_with_gene
            })

        self.results = pd.DataFrame(results).dropna()

    def plot_bar_charts(self, top_n=20):

        """
        Plot top co-occurring genes ranked by -log10(p-value) for each test.

        Args:
            top_n (int): Number of top genes to display.
        """

        df_chi = self.results.sort_values("chi2_neglogp", ascending=False).head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x="chi2_neglogp", y="gene", data=df_chi, palette="viridis")
        plt.xlabel("-log10(p-value) [Chi-square]")
        plt.ylabel("Gene")
        plt.title(f"Top {top_n} Co-occurring Genes with EGFR (Chi-square)")
        plt.tight_layout()
        plt.show()

        df_mwu = self.results.sort_values("mwu_neglogp", ascending=False).head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x="mwu_neglogp", y="gene", data=df_mwu, palette="magma")
        plt.xlabel("-log10(p-value) [Mann–Whitney U]")
        plt.ylabel("Gene")
        plt.title(f"Top {top_n} Co-occurring Genes with EGFR (MWU)")
        plt.tight_layout()
        plt.show()

    def plot_scatter(self):

        """
        Scatter plot comparing -log10(p-values) from Chi-square vs MWU tests.
        """

        # Filter to keep genes only significant for both tests

        sig_results = self.results[
            (self.results["chi2_p"] < 0.05) |
            (self.results["mwu_p"] < 0.05)
        ]

        sns.scatterplot(
            data=sig_results,
            x="chi2_neglogp", y="mwu_neglogp",
            hue="EGFR+_with_gene", palette="viridis"
        )

        plt.xlabel("-log10(p-value) Chi-square")
        plt.ylabel("-log10(p-value) MWU")
        plt.title("Chi-square vs MWU: Co-occurring Gene Significance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


class dna_mutation_class_simon:

    """
    A class to analyze co-occurring mutations with EGFR in cancer patient DNA samples.
    """

    def __init__(self, metadata: pd.DataFrame, dna_mutations: pd.DataFrame) -> pd.DataFrame:
        """
        Initialize the EGFR mutation analysis object.

        Parameters:
        - metadata: DataFrame with EGFR-positive sample info. Must contain 'Sample_ID'.
        - dna_mutations: DataFrame with all DNA mutations. Must contain columns:
            'Sample_ID', 'gene', 'dna_vaf', etc.
        """

        self.metadata = metadata
        self.dna_mutations = dna_mutations
        self.egfr_samples = set(metadata.index.unique())
        self.results = pd.DataFrame()

    def compute_stats(self):

        """
        Computes co-occurrence statistics for each gene:
        - Chi-square test: presence/absence in EGFR+ vs EGFR− samples.
        - Mann–Whitney U test: distribution of 'dna_vaf' in EGFR+ vs EGFR− samples.
        Stores results in self.results DataFrame.
        """

        all_samples = set(self.dna_mutations.index.unique())
        egfr_positive = self.egfr_samples
        egfr_negative = all_samples - egfr_positive

        results = []

        all_genes = self.dna_mutations["gene"].unique()
        for gene in all_genes:
            if gene == "EGFR":
                continue  # Skip EGFR itself

            gene_data = self.dna_mutations[self.dna_mutations["gene"] == gene]

            gene_samples = set(gene_data["Sample_ID"])
            egfr_pos_with_gene = len(gene_samples & egfr_positive)
            egfr_pos_without_gene = len(egfr_positive) - egfr_pos_with_gene
            egfr_neg_with_gene = len(gene_samples & egfr_negative)
            egfr_neg_without_gene = len(egfr_negative) - egfr_neg_with_gene

            # Build 2x2 contingency table
            table = [
                [egfr_pos_with_gene, egfr_pos_without_gene],
                [egfr_neg_with_gene, egfr_neg_without_gene]
            ]
            chi2, chi2_p, _, _ = chi2_contingency(table)

            # MWU test (based on VAF values in gene_data)
            vaf_egfr_pos = gene_data[gene_data["Sample_ID"].isin(egfr_positive)]["dna_vaf"]
            vaf_egfr_neg = gene_data[gene_data["Sample_ID"].isin(egfr_negative)]["dna_vaf"]
            if len(vaf_egfr_pos) > 0 and len(vaf_egfr_neg) > 0:
                mwu_stat, mwu_p = mannwhitneyu(vaf_egfr_pos, vaf_egfr_neg, alternative="two-sided")
            else:
                mwu_p = 1.0  # Non-informative

            results.append({
                "gene": gene,
                "chi2_p": chi2_p,
                "chi2_neglogp": -np.log10(chi2_p) if chi2_p > 0 else np.nan,
                "mwu_p": mwu_p,
                "mwu_neglogp": -np.log10(mwu_p) if mwu_p > 0 else np.nan,
                "EGFR+_with_gene": egfr_pos_with_gene,
                "EGFR-_with_gene": egfr_neg_with_gene
            })

        self.results = pd.DataFrame(results).dropna()

    def plot_bar_charts(self, top_n=20):

        """
        Plot top co-occurring genes ranked by -log10(p-value) for each test.

        Args:
            top_n (int): Number of top genes to display.
        """

        df_chi = self.results.sort_values("chi2_neglogp", ascending=False).head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x="chi2_neglogp", y="gene", data=df_chi, palette="viridis")
        plt.xlabel("-log10(p-value) [Chi-square]")
        plt.ylabel("Gene")
        plt.title(f"Top {top_n} Co-occurring Genes with EGFR (Chi-square)")
        plt.tight_layout()
        plt.show()

        df_mwu = self.results.sort_values("mwu_neglogp", ascending=False).head(top_n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x="mwu_neglogp", y="gene", data=df_mwu, palette="magma")
        plt.xlabel("-log10(p-value) [Mann–Whitney U]")
        plt.ylabel("Gene")
        plt.title(f"Top {top_n} Co-occurring Genes with EGFR (MWU)")
        plt.tight_layout()
        plt.show()

    def plot_scatter(self):

        """
        Scatter plot comparing -log10(p-values) from Chi-square vs MWU tests.
        """

        # Filter to keep genes only significant for both tests

        sig_results = self.results[
            (self.results["chi2_p"] < 0.05) |
            (self.results["mwu_p"] < 0.05)
        ]

        sns.scatterplot(
            data=sig_results,
            x="chi2_neglogp", y="mwu_neglogp",
            hue="EGFR+_with_gene", palette="viridis"
        )

        plt.xlabel("-log10(p-value) Chi-square")
        plt.ylabel("-log10(p-value) MWU")
        plt.title("Chi-square vs MWU: Co-occurring Gene Significance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def preranked_gsea(genes_df: pd.DataFrame,
                   threshold: int) -> pd.DataFrame:
    # Create direction column 1/-1
    genes_df["direction"] = np.sign(genes_df["Weight"])

    # Take the abs of log2foldchange
    genes_df["Weight"] = genes_df["Weight"].abs()

    # Sort values by size
    genes_df = genes_df.sort_values(by="Weight", ascending=False)

    # Threshold the genes
    genes_df = genes_df.head(threshold)

    # Multiple direction against weight to get the direction of fold change back
    genes_df['Weight'] = genes_df['Weight'] * genes_df['direction']

    # Drop direction column
    genes_df = genes_df.drop(columns=['direction'])

    return genes_df

    # Refactor add new column to indicate positive or negative fold change 1/-1
    # take absolute of log2fold change
    # Rank by log2fold change abs
    # Then threshold it
    # Multiply column by 1 or -1


def fullgenesets_scatter(gsea_df: gp.prerank,
                         plot_n_genesets: int,
                         title: str) -> tuple[pd.DataFrame, plt.Figure]:
    gsea_df = gsea_df.res2d.sort_index().head(plot_n_genesets)
    gsea_df = gsea_df.sort_values(by=['NES'])
    gsea_df['sig_at_fdr10'] = gsea_df['FDR q-val'] < 0.1
    edge_widths = np.where(gsea_df['sig_at_fdr10'], 3, 0.5)

    fig, ax = plt.subplots(figsize=(4, len(gsea_df) / 1.7))
    scatterplot = ax.scatter(y=gsea_df['Term'], x=gsea_df['NES'], c=gsea_df['FDR q-val'],
                             cmap='PiYG', linewidths=edge_widths, edgecolors='black', s=200)
    cbar = plt.colorbar(scatterplot, orientation="horizontal", pad=1 / len(gsea_df))
    cbar.set_label("False Discovery Rate")
    plt.vlines(0, 0, len(gsea_df) - 1, color='black', linestyles='--', linewidth=0.2)
    plt.title(title, size=12)
    plt.xlabel('Enrichment Score')

    return gsea_df, fig


def calculate_and_plot_genesets(raw_df: pd.DataFrame,
                                n_top_genes: int,
                                plot_n_genesets: int,
                                gene_set: str) -> tuple[pd.DataFrame, plt.Figure]:
    preranked_df = preranked_gsea(genes_df=raw_df, threshold=n_top_genes)

    print("Running GSEA for gene set:", gene_set)
    pre_res = gp.prerank(rnk=preranked_df,
                         gene_sets=gene_set,
                         processes=4,
                         min_size=5,
                         max_size=50,
                         permutation_num=200,
                         outdir=None)

    df_results, fig = fullgenesets_scatter(gsea_df=pre_res,
                                           plot_n_genesets=plot_n_genesets,
                                           title=f'{gene_set} For late stage patients who survived')

    return df_results, fig

def parse_gsea_df(results: pd.DataFrame) -> pd.DataFrame:
    gsea_df = results[['log2FoldChange']]
    gsea_df.index.name = 'Gene_name'
    gsea_df = gsea_df.reset_index()
    gsea_df.columns = ["Gene_name", "Weight"]

    return gsea_df


class DnaMutationNewAnalyser:

    """
    A class to analyze co-occurring mutations with EGFR in cancer patient DNA samples.
    """

    def __init__(self, metadata: pd.DataFrame,
                 dna_mutations: pd.DataFrame,
                 metadata_label_string,
                 metadata_label_group1,
                 metadata_label_group2,
                 experiment_name,
                 save_path) -> pd.DataFrame:
        """
        Initialize the EGFR mutation analysis object.

        Parameters:
        - metadata: DataFrame with EGFR-positive sample info. Must contain 'Sample_ID'.
        - dna_mutations: DataFrame with all DNA mutations. Must contain columns:
            'Sample_ID', 'gene', 'dna_vaf', etc.
        """

        self.metadata = metadata
        self.dna_mutations = dna_mutations
        self.metadata_label_string = metadata_label_string
        self.metadata_label_group1 = metadata_label_group1
        self.metadata_label_group2 = metadata_label_group2
        self.experiment_name = experiment_name
        self.save_path = save_path
        self.results = pd.DataFrame()



    def compute_stats(self):
        """
        Computes co-occurrence statistics for each gene:
        - Chi-square test: presence/absence in group1 vs group2 samples.
        - Mann–Whitney U test: distribution of mutation **counts** in group1 vs group2.
        Stores results in self.results DataFrame.
        """


        group_1 = self.metadata[self.metadata[self.metadata_label_string] == self.metadata_label_group1].index.tolist()
        group_2 = self.metadata[self.metadata[self.metadata_label_string] == self.metadata_label_group2].index.tolist()

        print(f"group 1: {len(group_1)}")
        print(f"group 2: {len(group_2)}")

        results = []

        all_genes = self.dna_mutations["gene"].unique()
        for gene in tqdm(all_genes):
            # if gene == "EGFR":
            #     continue  # Skip EGFR itself

            gene_data = self.dna_mutations[self.dna_mutations["gene"] == gene]
            gene_samples = set(gene_data["Sample_ID"].unique())

            group_1_with_mutated_gene = len(gene_samples & set(group_1))
            group_1_without_mutated_gene = len(group_1) - group_1_with_mutated_gene
            group_2_with_mutated_gene = len(gene_samples & set(group_2))
            group_2_without_mutated_gene = len(group_2) - group_2_with_mutated_gene

            try:
                table = [
                    [group_1_with_mutated_gene, group_1_without_mutated_gene],
                    [group_2_with_mutated_gene, group_2_without_mutated_gene]
                ]
                chi2, chi2_p, _, _ = chi2_contingency(table)
            except:
                chi2_p = 1.0

            # Mann–Whitney U test on mutation counts (not VAF)
            # Count number of mutations per sample in each group
            gene_counts = gene_data.groupby("Sample_ID").size()

            group_1_counts = gene_counts[gene_counts.index.isin(group_1)].values
            group_2_counts = gene_counts[gene_counts.index.isin(group_2)].values

            if len(group_1_counts) > 0 and len(group_2_counts) > 0:
                try:
                    mwu_stat, mwu_p = mannwhitneyu(group_1_counts, group_2_counts, alternative='two-sided')
                except:
                    mwu_p = 1.0
            else:
                mwu_p = 1.0

            results.append({
                "gene": gene,
                "group_1_with_mut": group_1_with_mutated_gene,
                "group_1_without_mut": group_1_without_mutated_gene,
                "group_2_with_mut": group_2_with_mutated_gene,
                "group_2_without_mut": group_2_without_mutated_gene,
                "chi2_p": chi2_p,
                "mwu_p": mwu_p,
                "chi2_neglogp": -np.log10(chi2_p) if chi2_p > 0 else np.nan,
                "mwu_neglogp": -np.log10(mwu_p) if mwu_p > 0 else np.nan,
            })

        self.results = pd.DataFrame(results)

        if self.save_path:
            self.results.to_csv(f"{self.save_path}/dna_mutation_results_{self.experiment_name}.csv", index=False)

        return self.results

    def plot_bar_charts(self, top_n=20):
        """
        Plot top co-occurring genes ranked by -log10(p-value) for each test.
    
        Args:
            top_n (int): Number of top genes to display.
    
        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]: 
            Figures for Chi-square and MWU bar plots.
        """
        # Chi-square bar plot
        df_chi = self.results.sort_values("chi2_neglogp", ascending=False).head(top_n)
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.barplot(x="chi2_neglogp", y="gene", data=df_chi, palette="viridis", ax=ax1)
        ax1.set_xlabel("-log10(p-value) [Chi-square]")
        ax1.set_ylabel("Gene")
        ax1.set_title(f"Top {top_n} Associated Mutated Genes with {self.experiment_name} (Chi-square)")
        fig1.tight_layout()
    
        # MWU bar plot
        df_mwu = self.results.sort_values("mwu_neglogp", ascending=False).head(top_n)
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.barplot(x="mwu_neglogp", y="gene", data=df_mwu, palette="magma", ax=ax2)
        ax2.set_xlabel("-log10(p-value) [Mann–Whitney U]")
        ax2.set_ylabel("Gene")
        ax2.set_title(f"Top {top_n} Associated Mutated Genes with {self.experiment_name} (MWU)")
        fig2.tight_layout()
    
        return fig1, fig2

    def plot_scatter(self):
        """
        Scatter plot comparing -log10(p-values) from Chi-square vs MWU tests.
        Returns:
            matplotlib.figure.Figure: The generated scatter plot figure.
        """
    
        sig_results = self.results[
            (self.results["chi2_p"] < 0.05) |
            (self.results["mwu_p"] < 0.05)
        ]
    
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=sig_results,
            x="chi2_neglogp", y="mwu_neglogp", ax=ax
        )
    
        ax.set_xlabel("-log10(p-value) Chi-square")
        ax.set_ylabel("-log10(p-value) MWU")
        ax.set_title("Chi-square vs MWU: Co-occurring Gene Significance")
        ax.grid(True)
        fig.tight_layout()
    
        return fig