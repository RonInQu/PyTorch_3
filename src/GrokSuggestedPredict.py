"""
Main Causes of Spikiness

EMA_BLOOD_PRIOR_NEW = 0.2 → too much weight on raw GRU when in blood (allows sudden jumps into clot/wall).
EMA_EXIT_TO_BLOOD_NEW = 0.65 → quite fast exit from alert back to blood (creates downward spikes).
DA_LABEL_CONFIDENCE = 0.92 + loose override thresholds → DA can still cause sharp changes.
No final post-EMA low-pass filter.
"""

# Recommended Improved Version
# Replace your entire Bayesian / EMA section inside predict() with this cleaner, smoother version:


        # ── Step 3: EMA blending with tunable smoothness ──
        new_idx = np.argmax(probs)
        prior_idx = np.argmax(self.posterior)

        if prior_idx == 0:                                 # Currently in blood
            alpha_history = 0.90                           # higher = slower to leave blood
            alpha_new     = 0.10
        elif new_idx == 0:                                 # Exiting alert → blood
            alpha_history = 0.55                           # slower recovery (less spiky drop)
            alpha_new     = 0.45
        elif new_idx == prior_idx:                         # Staying in same alert class
            alpha_history = 0.97                           # very stable
            alpha_new     = 0.03
        else:                                              # Clot ↔ Wall switch
            alpha_history = 0.99                           # strongly resist flicker
            alpha_new     = 0.01

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        # ── Optional final gentle low-pass (highly recommended) ──
        # This smooths out any remaining sharp jumps after EMA + DA
        final_alpha = 0.92 if np.argmax(self.posterior) == 0 else 0.96
        self.posterior = final_alpha * self.posterior + (1 - final_alpha) * probs
		
		
# Suggested Parameter Values (start here)	
# At the top of the file, replace your current EMA_* constants with these:

EMA_BLOOD_PRIOR_HISTORY     = 0.90      # slower to trigger alerts
EMA_EXIT_TO_BLOOD_HISTORY   = 0.55      # slower recovery from alert
EMA_SAME_CLASS_HISTORY      = 0.97      # very stable when staying in clot/wall
EMA_CROSS_CLASS_HISTORY     = 0.99      # strongly resist clot↔wall flips

# Final gentle smoothing after everything
FINAL_SMOOTH_ALPHA_BLOOD    = 0.92
FINAL_SMOOTH_ALPHA_ALERT    = 0.96

# Full Recommended predict method (cleaned)
    @torch.no_grad()
    def predict(self, active_feats, da_label=None):
        # 1. Scale + build sequence + run GRU
        scaled = self.scaler.transform(active_feats.reshape(1, -1))[0]
        self.feat_history.append(scaled)

        if len(self.feat_history) < SEQ_LEN:
            pad = list(self.feat_history)[0] if self.feat_history else scaled
            seq_list = [pad] * (SEQ_LEN - len(self.feat_history)) + list(self.feat_history)
        else:
            seq_list = list(self.feat_history)

        seq = np.array(seq_list, dtype=np.float32)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(DEVICE)

        logits, self.hidden = self.model(x, self.hidden)
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

        probs = torch.softmax(logits / TEMPERATURE, 1).squeeze(0).cpu().numpy()

        # 2. DA override
        if da_label is not None:
            if da_label == 0:
                self.posterior = np.array([0.98, 0.01, 0.01], dtype=np.float32)
                self.hidden = None
                self.feat_history.clear()
                return self.posterior.copy()

            elif da_label in (1, 2):
                if self._da_should_override_gru(probs, da_label):
                    probs = self._make_da_probs(da_label)

        # 3. EMA blending
        new_idx = np.argmax(probs)
        prior_idx = np.argmax(self.posterior)

        if prior_idx == 0:
            alpha_history = EMA_BLOOD_PRIOR_HISTORY
            alpha_new     = 1 - alpha_history
        elif new_idx == 0:
            alpha_history = EMA_EXIT_TO_BLOOD_HISTORY
            alpha_new     = 1 - alpha_history
        elif new_idx == prior_idx:
            alpha_history = EMA_SAME_CLASS_HISTORY
            alpha_new     = 1 - alpha_history
        else:
            alpha_history = EMA_CROSS_CLASS_HISTORY
            alpha_new     = 1 - alpha_history

        self.posterior = alpha_history * self.posterior + alpha_new * probs

        # 4. Final gentle smoothing
        final_alpha = FINAL_SMOOTH_ALPHA_BLOOD if np.argmax(self.posterior) == 0 else FINAL_SMOOTH_ALPHA_ALERT
        self.posterior = final_alpha * self.posterior + (1 - final_alpha) * probs

        # 5. Post-EMA DA safety net
        if da_label in (1, 2):
            final_idx = np.argmax(self.posterior)
            if final_idx != da_label and self._da_should_override_gru(probs, da_label, strict=True):
                self.posterior = self._make_da_probs(da_label)

        return self.posterior.copy()
		