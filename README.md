# Equity Factors, Dividend Research and Warehouse Signals

Personal research archive covering residual momentum, valuation screening, dividend-index analysis and commodity warehouse signals. Historical scripts are retained for inspection, not silently corrected and presented as the originals.

## Research sequence
The internship summary describes screening the union of low-PB, low-PE and high-dividend candidates, then ranking residual momentum and testing monthly rebalancing. Related work examined CSI800, commodity warehouse changes and dividend-index behaviour. This does not establish that a complete commercial Barra model was implemented.

The later tutorial follows data availability, winsorization, cross-sectional standardization, industry/size neutralization, forward-return regression, Rank IC, portfolio construction, costs and holdout diagnostics. Its dividend-plus-ROE design and planted synthetic signal are teaching extensions, not the historical strategy.

## Known corrections
The old dividend chart script scales a 1,250-observation rolling standard deviation by sqrt(1250). Annualization for 250 trading days uses sqrt(250), regardless of the estimation-window length. Historical report figures require checking against original data; they are not independently verified performance.

The warehouse script's first two renamed columns disagree with the archived spreadsheet headers. Performance is blocked until the instrument, price, warehouse definition and release timestamp are confirmed. The original `barra.py` is a small simulated example and contains a shell installation command; it is stored as text, not advertised as a production risk model.

## Run
`python src/reproduce.py` runs the neutralization and return-metric checks. `MPLBACKEND=Agg python tutorials/factor_research_tutorial.py` runs the complete synthetic tutorial. Open `tutorials/Factor_Research_Tutorial_CN.ipynb` for explanations and saved figures.
