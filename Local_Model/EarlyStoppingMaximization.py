import optuna

class EarlyStoppingCallbackMax:
    def __init__(self, n_warmup=10, min_improvement=0.05):
        self.n_warmup = n_warmup
        self.min_improvement = min_improvement
        self.best_score = None
        self.trial_scores = []
        
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        
        self.trial_scores.append(trial.value)
        
        # Wait for warmup period
        if len(self.trial_scores) < self.n_warmup:
            return
            
        # Update best score if needed
        if self.best_score is None or trial.value > self.best_score:
            self.best_score = trial.value
            
        # Calculate improvement over last n_warmup trials
        # recent_best = max(self.trial_scores[-self.n_warmup:])
        recent_best = max(self.trial_scores[-self.n_warmup:])
        if self.best_score > 0:  # Avoid division by zero
            improvement = (recent_best - self.best_score) / abs(self.best_score)
            
            # Stop if improvement is less than threshold
            if improvement < self.min_improvement:
                study.stop()