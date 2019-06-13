from . import EvaluationBase
from itertools import permutations
import numpy as np

class ScaleInvariantSDR(EvaluationBase):
    def __init__(self, true_sources_list, estimated_sources_list, 
                 compute_permutation=False, source_labels=None, scaling=True):
        self.true_sources_list = true_sources_list
        self.estimated_sources_list = estimated_sources_list
        self.compute_permutation = compute_permutation
        self.scaling = scaling
        
        if source_labels is None:
            source_labels = []
            for i, x in enumerate(self.true_sources_list):
                if x.path_to_input_file:
                    label = x.path_to_input_file
                else:
                    label = f'source_{i}'
                source_labels.append(label)
        self.source_labels = source_labels
        self.reference_array, self.estimated_array = self._preprocess_sources()
        
    def evaluate(self):
        num_sources = self.reference_array.shape[-1]
        num_channels = self.reference_array.shape[1]
        orderings = (
            list(permutations(range(num_sources))) 
            if self.compute_permutation 
            else [list(range(num_sources))]
        )
        results = np.empty((len(orderings), num_channels, num_sources, 3))
        
        for o, order in enumerate(orderings):
            for c in range(num_channels):
                for j in order:
                    SDR, SIR, SAR = self._compute_sdr(
                        self.estimated_array[:, c, j], self.reference_array[:, c, order], j, scaling=self.scaling
                    )
                    results[o, c, j, :] = [SDR, SIR, SAR]
        return self._populate_scores_dict(results, orderings)
    
    def _populate_scores_dict(self, results, orderings):
        best_permutation_by_sdr = np.argmax(results[:, :, :, 0].mean(axis=1).mean(axis=-1))
        results = results[best_permutation_by_sdr]
        best_permutation = orderings[best_permutation_by_sdr]
        scores = {'permutation': list(best_permutation)}
        for j in best_permutation:
            label = self.source_labels[j]
            scores[label] = {
                metric: results[:, j, m].tolist()
                for m, metric in enumerate(['SDR', 'SIR', 'SAR'])
            }
        return scores
    
    @staticmethod
    def _compute_sdr(estimated_signal, reference_signals, source_idx, scaling=True):
        references_projection = reference_signals.T @ reference_signals
        source = reference_signals[:, source_idx]
        scale = (source @ estimated_signal) / references_projection[source_idx, source_idx] if scaling else 1

        e_true = scale * source
        e_res = estimated_signal - e_true

        signal = (e_true ** 2).sum()
        noise = (e_res ** 2).sum()
        SDR = 10 * np.log10(signal / noise)
        
        references_onto_residual = np.dot(reference_signals.transpose(), e_res)
        b = np.linalg.solve(references_projection, references_onto_residual)

        e_interf = np.dot(reference_signals , b)
        e_artif = e_res - e_interf

        SIR = 10 * np.log10(signal / (e_interf**2).sum())
        SAR = 10 * np.log10(signal / (e_artif**2).sum())
        return SDR, SIR, SAR
    
    def _preprocess_sources(self):
        """
        Prepare the :ref:`audio_data` in the sources. Uses the format:
            (num_samples, num_channels, num_sources)
        Returns:
            (:obj:`np.ndarray`, :obj:`np.ndarray`) reference_source_array, estimated_source_array

        """
        reference_source_array = np.stack([np.copy(x.audio_data.T)
                                            for x in self.true_sources_list], axis=2)
        estimated_source_array = np.stack([np.copy(x.audio_data.T)
                                            for x in self.estimated_sources_list], axis=2)
        reference_source_array -= reference_source_array.mean(axis=0)
        estimated_source_array -= estimated_source_array.mean(axis=0)
        
        return reference_source_array, estimated_source_array