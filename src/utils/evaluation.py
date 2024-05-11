import networkx as nx
import torch
from src.utils.utils import transfer_to_pytorch_fun
import math
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm


class PerformanceEvaluation:
    def __init__(
        self,
        dataset,
        model,
        model_name,
        device,
        testing_data,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.testing_data = testing_data
        self.predicting_results = self.get_testing_set()

    def get_testing_set(self):
        predicted_results = []
        for id, sample_data in tqdm(enumerate(self.testing_data)):
            sample = sample_data.copy()
            data = transfer_to_pytorch_fun(sample)
            data.to(self.device)
            with torch.no_grad():
                if self.model_name == "Convspatial":
                    pred = self.model(data.x, data.edge_index, data.region)
                else:
                    pred = self.model(data.x, data.edge_index)

            pred_edge_feature = {e: k for e, k in zip(sample.edges(), pred.tolist())}
            nx.set_edge_attributes(sample, pred_edge_feature, "edge_attr")

            result = (pred > 0.5).float()
            pred_edge_label = {e: k for e, k in zip(sample.edges(), result.tolist())}

            nx.set_edge_attributes(sample, pred_edge_label, "edge_label")

            pos = {k: (v["x"][0][0], v["x"][0][1]) for k, v in sample.nodes(data=True)}
            data_distance = {
                edge: math.dist(pos[edge[0]], pos[edge[1]])
                for edge in sample_data.edges()
            }
            result_distance = {
                edge: math.dist(pos[edge[0]], pos[edge[1]]) for edge in sample.edges()
            }
            nx.set_edge_attributes(sample, result_distance, "weight")
            nx.set_edge_attributes(sample_data, data_distance, "weight")

            predicted_results.append(sample)

            del pred, data
        return predicted_results

    def performance_summary(self):
        (
            data_length,
            predict_length,
            data_node,
            predict_node,
            f1_scores,
            roc_scores,
            data_components,
            prediction_components,
        ) = ([], [], [], [], [], [], [], [])
        for data, prediction in tqdm(zip(self.testing_data, self.predicting_results)):
            data_length_sample, predict_length_sample = self.edge_length(
                data, prediction
            )
            data_node_sample, predict_node_sample = self.node_degree(data, prediction)
            f1_score_sample = self.get_f1_score(data, prediction)
            # roc_sample = self.roc_auc_value(data, prediction)
            roc_sample = 0
            (
                data_component_sample,
                prediction_component_sample,
            ) = self.get_num_of_components(data, prediction)

            data_length += data_length_sample
            predict_length += predict_length_sample
            data_node += data_node_sample
            predict_node += predict_node_sample
            f1_scores += [f1_score_sample]
            roc_scores += [roc_sample]
            data_components += [data_component_sample]
            prediction_components += [prediction_component_sample]

        return (
            data_length,
            predict_length,
            data_node,
            predict_node,
            f1_scores,
            roc_scores,
            data_components,
            prediction_components,
        )

    def get_f1_score(self, data, prediction):
        y_true = list(nx.get_edge_attributes(data, "edge_attr").values())
        y_predict = list(nx.get_edge_attributes(prediction, "edge_label").values())
        return f1_score(y_true, y_predict)

    def get_confusion_matrix(self, data, prediction):
        prediction = np.array(
            list(nx.get_edge_attributes(prediction, "edge_attr").values())
        )
        tn, fp, fn, tp = confusion_matrix(
            list(nx.get_edge_attributes(data, "edge_attr").values()), prediction
        ).ravel()
        return tn, fp, fn, tp

    def edge_length(self, data, prediction):
        data_edge = [
            edge
            for edge in data.edges()
            if nx.get_edge_attributes(data, "edge_attr")[edge] > 0.5
        ]
        data_distance = [
            nx.get_edge_attributes(data, "weight")[edge] for edge in data_edge
        ]

        prediction_edge = [
            edge
            for edge in prediction.edges()
            if nx.get_edge_attributes(prediction, "edge_attr")[edge] > 0.5
        ]
        result_distance = [
            nx.get_edge_attributes(prediction, "weight")[edge]
            for edge in prediction_edge
        ]

        return data_distance, result_distance

    def node_degree(self, data, prediction):
        data_node = [data.degree(weight="edge_attr")[n] for n in data.nodes()]
        predict_node = [
            prediction.degree(weight="edge_label")[n] for n in prediction.nodes()
        ]
        return data_node, predict_node

    def roc_auc_value(self, data, prediction):
        roc_score = roc_auc_score(
            list(nx.get_edge_attributes(data, "edge_attr").values()),
            list(nx.get_edge_attributes(prediction, "edge_attr").values()),
        )
        return roc_score

    def get_num_of_components(self, data, prediction):
        cleaned_data = data.copy()
        cleaned_prediction = prediction.copy()
        data_remove_edges = [
            e
            for e in data.edges()
            if nx.get_edge_attributes(data, "edge_attr")[e] < 0.5
        ]

        prediction_remove_edges = [
            e
            for e in prediction.edges()
            if nx.get_edge_attributes(prediction, "edge_attr")[e] < 0.5
        ]

        cleaned_data.remove_edges_from(data_remove_edges)
        cleaned_prediction.remove_edges_from(prediction_remove_edges)

        return nx.number_connected_components(
            cleaned_data
        ), nx.number_connected_components(cleaned_prediction)
