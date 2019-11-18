import numpy as np
import pandas as pd
import scipy

class NaiveBayes():
    def __init__(self,
                 smoothing: float = 1.0, # Laplace/Lidstone smoothing
                 fit_confidence: bool = True
                 ) -> None:
        self.smoothing  = smoothing
        self.fit_confidence = fit_confidence
        self.log_priors = None, # log_prob(y)
        self.log_conditionals = {} # log_prob(x_i|y)
        
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_weights: pd.Series = None,
            sample_weights: pd.Series = None
            ):
        
        self.feature_weights = pd.Series(data=1.0, index=X.columns) if not feature_weights else feature_weights
        self.sample_weights = pd.Series(data=1.0, index=X.index.values) if not sample_weights else sample_weights
        
        self._set_log_priors(y, self.sample_weights)
        self._set_log_conditionals(X, y, self.sample_weights)
        
        if self.fit_confidence and self.log_priors:
            self._set_cofidence(X, y, feature_weights)
        else:
            self.confidence = 1.0          
        return self
    
    def _set_log_priors(self, y, sample_weights):
        if not feature_weights:
            counts = y.value_counts()
        else:
            counts = w.groupby(y).sum()
        
        counts.sort_index(inplace=True)
        counts += self.smoothing
        self.log_priors= no.log(counts) - np.log(counts.sum())
    
    def _set_log_conditionals(self, X, y, sample_weights):
        conditionals = list(map(
                # calculate crosstable(w/wo weights), sum up, add smoothing
                lambda col: pd.crosstab(X[col], y, sample_weights, aggfunc=np.sum).sort_index(ascending=True, axis=0).sort_index(ascending=True, axis=1).fillna(0)+self.smoothing, 
                X.columns.values))
        log_conditionals = [np.log(c) - np.log(c.sum(axis=0)) for c in conditionals]
        self.log_conditionals = dict.fromkeys(X.columns.values, log_conditionals)
    
    def predict_log_prob(self, X, feature_weights):
        feature_weights = feature_weights.multiply(self.confidence)
        log_prob = pd.DataFrame(index=X.index, columns=self.log_priors.index, data=0.0)
        for col in X.columns:
            conditionals = pd.merge(X[col], 
                                    self.log_conditionals[col],
                                    how='left',
                                    left_on=col,
                                    right_index=True,
                                    suffixes=('test', None)
                                    ).fillna(0.0).drop(str(col)+'test', inplace=True)
            log_prob += feature_weights[col]*conditionals
        return log_prob
    
    def _set_confidence(self, X, y, sample_weights):
        self.confidence = 1.0
        pred_log_prob = self.predict_log_prob(X, self.feature_weights)
        prior_log_prob = pd.DataFrame(index=X.index, columns=self.log_priors.index, data=0.0)
        for y in self.log_priors.index:
            prior_log_prob[y] = self.log_priors[y]
        opt_result = scipy.optimize.minimize_scalar(
                            self._objective_function,
                            bracket = (0.0, 1.0),
                            method = "Brent",
                            args = (prior_log_prob, pred_log_prob, y, sample_weights)
                            )
        self.confidence = opt_result.x
    
    def _objective_function(self,
                            a: float,
                            priors: pd.Series,
                            posts: pd.Series,
                            y: pd.Series,
                            sample_weights,
        ) -> float:
        # negtive log-likelyhood, prob(y|x)
        probs = posts.multiply(a).add(priors)
        norm = scipy.special.logsumexp(output, axis=1)
        probs = probs.subtract(norm, axis=0)
        if sample_weights:
            probs = probs.multiply(sample_weights, axis=0)
        row_col = pd.DataFrame(data={"row":probs.index, "col":y}).dropna()
        return (-probs.lookup(row_col['row'], row_col['col'])).sum()

if __name__="__mains__":
    pass

    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        