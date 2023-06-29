#! /usr/bin/env python
# coding: utf-8

"""Create clustering model using HDBScan and AgglomerativeClustering models with embeddings"""

import os
import math
from tqdm.auto import tqdm
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch

from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from kneed import KneeLocator
import umap.umap_ as umap

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from hdbscan import HDBSCAN
import numpy as np

from loguru import logger
from lib.api import get_openai_embeddings, get_palm_embeddings
from lib.prompts import PROMPT_TEMPLATE_CLUSTER
from lib.api import get_openai_response_chat, get_palm_response
import settings


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Clustering Class
class ClusterTopics:
    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        min_cluster_size: int = 10,
        min_samples: Union[int, bool] = None,
        reduction_dims: Union[int, float] = 0,
        cluster_model: str = "hdbscan",
        cluster_description_model: Union[str, None] = None,
        cluster_categories: List[str] = None,
        use_elbow: bool = True,
        keep_outliers: bool = False,
        n_jobs: int = 6,
    ):
        self.embedding_model = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or round(math.sqrt(self.min_cluster_size))
        self.reduction_dims = reduction_dims
        self.cluster_model = cluster_model
        self.cluster_description_model = cluster_description_model

        self.cluster_categories = cluster_categories
        if cluster_categories:
            self.cluster_categories = [c for c in filter(len, list(set(cluster_categories)))]
    
        self.use_elbow = use_elbow
        self.keep_outliers = keep_outliers
        self.n_jobs = n_jobs
        self.embeddings = None
        self.corpus = None
        self.labels = None
        self.text_labels = None
        self.model = None
        self.model_data = None
        self.post_process = None


    def get_embeddings(self, sentences: List[str]) -> np.ndarray:

        """Converts text to embeddings"""

        if self.embedding_model == "openai":
            return get_openai_embeddings(sentences)
        elif self.embedding_model == "palm":
            return get_palm_embeddings(sentences)
        
        else:
            # Only do batching and progress if many embeddings
            if len(sentences) > 64:
                embeddings = SentenceTransformer(self.embedding_model).encode(
                    sentences, show_progress_bar=True, batch_size=64
                )
            else:
                embeddings = SentenceTransformer(self.embedding_model).encode(sentences)

            return np.asarray(embeddings)

    def get_reduced(self, embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:

        """Reduce dimensions using UMAP. This can reduce clustering time and memory,
        but at the expence of reduced infomration.
        Reducing to 2 dimensions is needed for plotting.
        TODO: Finding the right cluster terms (ngrams) breaks with dim reduction."""

        # No reduction
        if self.reduction_dims <= 0:
            return np.asarray(embeddings)

        if isinstance(self.reduction_dims, float):
            n_dims = math.ceil(embeddings.shape[1] * self.reduction_dims)
        else:
            n_dims = self.reduction_dims

        logger.info("Reducing embeddings to {} dims".format(n_dims))

        # returns np.ndarray
        return umap.UMAP(
            n_neighbors=3,
            n_components=n_dims,
            random_state=settings.RANDOM_SEED,
        ).fit_transform(embeddings)

    def get_elbow(self, embeddings: Union[torch.Tensor, np.ndarray]) -> float:

        """Gets the elbow or sorted inflection point of input data as float."""

        if self.use_elbow:
            k = self.min_samples
            nbrs = NearestNeighbors(
                n_neighbors=k, n_jobs=self.n_jobs, algorithm="auto"
            ).fit(embeddings)
            distances, _ = nbrs.kneighbors(embeddings)
            distances = [np.mean(d) for d in np.sort(distances, axis=0)]
            kneedle = KneeLocator(
                distances,
                list(range(len(distances))),
                online=True,
                interp_method="polynomial",
                curve="convex",
                direction="increasing",
            )
            epsilon = np.min(list(kneedle.all_elbows))
            if epsilon == 0.0:
                epsilon = np.mean(distances)
        else:
            epsilon = 0.5

        logger.info("Using epsilon value: {}".format(epsilon))

        return float(epsilon)

    def set_cluster_model(self, cluster_model: str) -> None:
        """Sets the cluster type for the class"""
        self.cluster_model = cluster_model

    def get_cluster_model(self, model_name: Union[str, None] = None) -> None:

        """Gets the correct clustering model and sets them up."""

        model_name = model_name or self.cluster_model

        logger.info("Cluster Model: {}".format(model_name))

        if model_name == "hdbscan":

            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            return HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                cluster_selection_epsilon=self.get_elbow(self.model_data),
                core_dist_n_jobs=self.n_jobs,
            )

        elif model_name == "agglomerative":

            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            # If we want to find the optimal number of clusters.
            # n_clusters = self.find_optimal_clusters_agglomerative(self.embeddings)
            n_clusters = len(self.cluster_categories) if isinstance(self.cluster_categories, list) else None
            distance_threshold = float(self.min_samples) if not isinstance(self.cluster_categories, list) else None
            
            return AgglomerativeClustering(
                n_clusters= n_clusters,
                compute_full_tree=True,
                distance_threshold=distance_threshold,
            )

        elif model_name == "optics":

            # Normalize embeddings
            self.model_data = self.embeddings / np.linalg.norm(
                self.embeddings, axis=1, keepdims=True
            )

            return OPTICS(
                min_samples=self.min_samples,
                eps=self.get_elbow(self.model_data),
                min_cluster_size=self.min_cluster_size,
                n_jobs=self.n_jobs,
            )

        else:
            logger.error("Only `hdbscan` and `agglomerative` are implemented.")
            raise NotImplementedError(
                "Only `hdbscan`, `optics`, and `agglomerative` are implemented."
            )

    def top_ngram_embeddings(
        self, corpus: List[str], min_df: Union[int, float] = 0.001
    ) -> tuple:

        """Returns 1-3 term ngrams and their embeddings"""
        
        # If a list of categories is given, use those
        if isinstance(self.cluster_categories, list):
            return (np.asarray(self.cluster_categories), self.get_embeddings(self.cluster_categories))
        
        features = None

        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=min_df)
            X1 = vectorizer.fit_transform(corpus)
            features = np.asarray(vectorizer.get_feature_names_out())
            sums = np.asarray(X1.sum(axis=0))[0]

        except ValueError as e:
            logger.error(
                "There was an error in finding top word embeddings: {}".format(str(e))
            )

        if features is not None:
            return (features, self.get_embeddings(features))

        return (np.array([]), np.array([]))

    def get_text_label_mapping(self, n_neighbors: int = 3) -> dict:

        """Finds the closest n-gram to a clusters centroid.
        Returns a dict to be used to map these to labels."""

        labels_idx, centroids = self.cluster_centroid_deduction()
        top_ngrams, top_ngram_embeddings = self.top_ngram_embeddings(self.corpus)

        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(top_ngram_embeddings)

        mapping = {-1: "<outliers>"}

        for i, label in tqdm(
            enumerate(labels_idx), desc="Finding labels", total=len(labels_idx)
        ):
            # skip matching if outlier
            if label == -1:
                continue

            top_text_idx = neigh.kneighbors(np.array([centroids[i]]))[1].flatten()
            mapping[label] = top_ngrams[top_text_idx][0]

        return mapping
    

    def get_text_label_mapping_llm(self) -> dict:
        """Gets explanations for each cluster using Palm or OpenAI LLM"""

        labels = np.sort(np.unique(self.labels))

        mapping = {-1: "<outliers>"}

        for label in tqdm(labels, desc="Finding labels", total=len(labels)):

            if label == -1:
                continue

            idx = np.where(self.labels == label)[0]
            samples = self.corpus[idx]

            # Random sample of 200 corpus texts
            if len(samples) > 200:
                samples = np.random.choice(samples, size=200, replace=False)

            # Get prompt
            prompt = PROMPT_TEMPLATE_CLUSTER.format(samples=samples)


            if self.cluster_description_model == "palm":
                explanation = get_palm_response(prompt)
            elif self.cluster_description_model == "openai":
                explanation = get_openai_response_chat(prompt, system_message="You are an expert at understanind the intent of Google searches.")
            else:
                raise NotImplementedError("Only `palm` and `openai` are implemented.")


            mapping[label] = explanation


        return mapping


    def cluster_centroid_deduction(self) -> np.ndarray:

        """Finds the centroids, or central point in space,
        for each cluster of texts."""

        centroids = []

        labels = np.sort(np.unique(self.labels))

        for label in labels:
            idx = np.where(self.labels == label)[0]
            centroid = np.mean(self.embeddings[idx], axis=0)
            centroids.append(centroid)

        return labels, np.array(centroids)

    def fish_additional_outliers(self) -> None:

        """Finds additional labels in the outliers getting the closest centroid
        and only returning ones with a good sillouhette score"""

        outliers_idx = np.where(self.labels == -1)[0]
        outlier_embeddings = self.embeddings[outliers_idx]
        labels, centroids = self.cluster_centroid_deduction()

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(centroids)
        assigned_idx = neigh.kneighbors(outlier_embeddings)[1].flatten()
        map_to_label = lambda x: labels[x]
        assigned = map_to_label(assigned_idx)

        if len(np.unique(assigned)) > 1:
            scores = silhouette_samples(outlier_embeddings, assigned)
            matched = np.where(scores > 0)[0]
            not_matched = np.where(scores <= 0)[0]

            self.labels[outliers_idx[matched]] = assigned[matched]
            self.labels[outliers_idx[not_matched]] = assigned[not_matched]

        return None

    def recluster_outliers(self) -> None:

        """This uses agglomerative clustering to recluster any remaining
        outliers.  One negative is that agglomerative doesn't produce
        outliers and some tokes (e.g. mispellings) SHOULD be outliers."""

        outliers_idx = np.where(self.labels == -1)[0]
        model = self.get_cluster_model(model_name="agglomerative")
        labels_idx = model.fit(self.model_data[outliers_idx]).labels_
        n = self.labels.max() + 1
        labels_idx = np.array([l + n for l in labels_idx])
        self.labels[outliers_idx] = labels_idx


    def fit(self, corpus: List[str]) -> tuple:

        """This is the main fitting function that does all the work."""

        self.corpus = np.array(corpus)

        logger.info("Getting embeddings.")
        self.embeddings = self.get_embeddings(self.corpus)

        self.model = self.get_cluster_model()

        logger.info("Fitting model.")
        self.model_data = self.get_reduced(self.model_data)
        self.labels = self.model.fit(self.model_data).labels_
        logger.info(
            "Initial Model. Unique Labels: {}".format(len(np.unique(self.labels)))
        )

        if -1 in list(self.labels):
            logger.info("Running post processes for outliers.")
            self.fish_additional_outliers()
            if not self.keep_outliers:
                self.recluster_outliers()
            logger.info(
                "Post Processing. Unique Labels: {}".format(len(np.unique(self.labels)))
            )

        logger.info("Finding names for cluster labels.")

        if self.cluster_description_model is not None:
            label_mapping = self.get_text_label_mapping_llm()
        else:
            label_mapping = self.get_text_label_mapping()

        self.text_labels = [label_mapping[l] for l in self.labels]

        return (self.labels, self.text_labels)