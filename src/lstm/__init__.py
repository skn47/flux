"""
LSTM price-reaction model (Phase F, final step).

Consumes `features_daily` (built by `features/`, see `features/README.md`)
and trains/evaluates a next-day adj_close regression LSTM in two feature
variants -- baseline (price/technical only) vs. flux-augmented (+
`flux_score`) -- per `progress/phase_f_lstm_decisions.md`. Does no
backtesting/trading-signal conversion; that is a separate later phase.
"""
